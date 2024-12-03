# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import argparse
from pathlib import Path

import yaml
import torch
from tqdm.auto import tqdm
from gluonts.dataset.field_names import FieldName
from gluonts.torch.util import lagged_sequence_values
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from tqdm import tqdm, trange

from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    get_next_file_num,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
)
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.sampler import (
    DDPMGuidance,
    DDIMGuidance,
)
import uncond_ts_diff.configs as diffusion_configs

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}

def create_model(config):
    model = TSDiff(
        **getattr(diffusion_configs, config["diffusion_config"]),
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization=config["normalization"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        lr=config["lr"],
        init_skip=config["init_skip"],
    )
    model.to(config["device"])
    return model

def load_model(config):
    model = TSDiff(
        **getattr(
            diffusion_configs,
            config.get("diffusion_config", "diffusion_small_config"),
        ),
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization="mean",
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        init_skip=config["init_skip"],
    )
    model.load_state_dict(
        torch.load(config["ckpt"], map_location="cpu"),
        strict=True,
    )
    model = model.to(config["device"])
    return model

def evaluate_one_step_generator(
    config, model, test_dataset, transformation, num_samples=100
):
    transformed_testdata = transformation.apply(
        test_dataset, is_train=False
    )
    test_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=config["prediction_length"],
        mode="test",
    )

    masking_transform = MaskInput(
        FieldName.TARGET,
        FieldName.OBSERVED_VALUES,
        config["context_length"],
        None,
        0,
    )
    test_transform = test_splitter + masking_transform

    predictor = sampler.get_predictor(
        test_transform,
        batch_size=1280 // num_samples,
        device=config["device"],
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_testdata,
        predictor=predictor,
        num_samples=num_samples,
    )
    forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
    tss = list(ts_it)
    evaluator = Evaluator()
    metrics, _ = evaluator(tss, forecasts)
    metrics = filter_metrics(metrics)
    results.append(dict(**missing_data_kwargs, **metrics))

    return results

def main(config: dict, log_dir: str):
    # set hyperparameter for distillation
    num_steps = 10_000
    gradient_clip_val = 0.5
    lr = 1.0e-3
    batch_size = 32
    masking_size = 24
    missing_data_kwargs = {
                "missing_scenario": "none",
                "missing_values": 0,
    }
    sampler_params = config["sampler_params"]
    
    # Read global parameters
    dataset_name = config["dataset"]
    freq = config["freq"]
    prediction_length = config["prediction_length"]
    num_samples = config["num_samples"]

    # Load dataset and model
    logger.info("Loading model")
    real_model = load_model(config)
    fake_model = load_model(config)
    one_step_model = load_model(config)
    device = next(real_model.backbone.parameters()).device

    # Set up for guidance
    sampler = DDPMGuidance(
            model=real_model,
            prediction_length=config["prediction_length"],
            num_samples=num_samples,
            **missing_data_kwargs,
            **sampler_params,
        )

    optimizer_one_step = torch.optim.Adam(sampler.parameters(), lr=lr)
    with trange(num_steps) as pbar:
        for step in pbar:
            optimizer_one_step.zero_grad()
            with torch.no_grad():
                # generating data
                gt_data = real_model.sample_n(batch_size, return_lags=True)
                masked_data = gt_data.clone() # (batch_size, seq_len, time_lags)
                
                # making observation mask
                prior_mask = torch.ones((batch_size, max(real_model.model.lags_seq)), device=device) # (batch_size, time_lags - 1)
                observation_mask = torch.ones_like(samples[:, :]) # (batch_size, seq_len)
                observation_mask[:,  -masking_size:] = 0.0
                lagged_mask = lagged_sequence_values(
                    real_model.model.lags_seq,
                    prior_mask,
                    observation_mask,
                    dim=1,
                ) # (batch_size, seq_len, time_lags - 1)
                observation_mask = torch.cat([observation_mask[:, :, None], lagged_mask], dim=-1) # (batch_size, seq_len, time_lags)

                # making prediction data
                masked_data[observation_mask != 1] = torch.randn_like(masked_data[observation_mask != 1])
                samples = sampler.guide(samples, observation_mask, None, 1.0)
                
            # calculating regression loss
            t = torch.full((batch_size,), 0, device=device, dtype=torch.long)
            one_step_samples = one_step_model.p_sample(masked_data, t, 0, features=None)

            loss = torch.nn.functional.mse_loss(one_step_samples, samples)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(one_step_model.parameters(), gradient_clip_val)

            optimizer_one_step.step()
            pbar.set_postfix({"loss": loss.item()})

    if False: # for evaluation code
        dataset = get_gts_dataset(dataset_name)
        assert dataset.metadata.freq == freq
        assert dataset.metadata.prediction_length == prediction_length

        # Setup data transformation and loading
        transformation = create_transforms(
            num_feat_dynamic_real=0,
            num_feat_static_cat=0,
            num_feat_static_real=0,
            time_features=real_model.time_features,
            prediction_length=prediction_length,
        )

        # Run guidance
        results = evaluate_one_step_generator(
            config, one_step_model, dataset.test, transformation, num_samples=num_samples
        )

        # Save results
        log_dir = Path(log_dir) / "guidance_logs"
        log_dir.mkdir(exist_ok=True, parents=True)
        base_filename = "results"
        run_num = get_next_file_num(
            base_filename, log_dir, file_type="yaml", separator="-"
        )
        save_path = log_dir / f"{base_filename}-{run_num}.yaml"

        with open(save_path, "w") as fp:
            yaml.safe_dump(
                {"config": config, "metrics": results},
                fp,
                default_flow_style=False,
                sort_keys=False,
            )


if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results", help="Path to results dir"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logger.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)

    main(config=config, log_dir=args.out_dir)
