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
from copy import deepcopy

from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    get_next_file_num,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
    extract
)
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.sampler import (
    DDPMGuidance,
    DDIMGuidance, 
)
import uncond_ts_diff.configs as diffusion_configs

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}

def fourier_loss(predicted, target):
    """
    Computes the loss between the predicted and target sequences in the frequency domain.

    Args:
        predicted (torch.Tensor): Predicted time series of shape (batch_size, seq_len, ...).
        target (torch.Tensor): Ground truth time series of shape (batch_size, seq_len, ...).

    Returns:
        torch.Tensor: Fourier-based loss value.
    """
    # Compute the Fourier transform of both sequences
    pred_fft = torch.fft.fft(predicted, dim=1)
    target_fft = torch.fft.fft(target, dim=1)

    # Compute the magnitude spectrum
    pred_magnitude = torch.abs(pred_fft)
    target_magnitude = torch.abs(target_fft)

    # Compute MSE on the magnitude spectrum
    return torch.nn.functional.mse_loss(pred_magnitude, target_magnitude)

@torch.no_grad()
def forward_diffusion_with_mask(model: TSDiff, x_start, t, mask, noise=None):
    device = next(model.backbone.parameters()).device
    if noise is None:
        noise = torch.randn_like(x_start, device=device)
    sqrt_alphas_cumprod_t = extract(
        model.sqrt_alphas_cumprod, t, x_start.shape
    )
    sqrt_one_minus_alphas_cumprod_t = extract(
        model.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    noise_x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    noise_x[mask == 1] = x_start[mask == 1]
    return noise_x

class NoOPGuidance(DDPMGuidance):
    @torch.no_grad()
    def _reverse_diffusion(
        self, observation, observation_mask, features, base_scale
    ):
        device = observation.device
        batch_size = observation.shape[0]

        seq = torch.randn_like(observation)
        seq[observation_mask == 1] = observation[observation_mask == 1]
        t = torch.full((batch_size,), self.model.timesteps - 1, device=device, dtype=torch.long)
        seq = self.model.fast_denoise(seq, t, features=None)

        return seq

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
    logger.info(f"Evaluating with {num_samples} samples.")
    results = []
    if config["setup"] == "forecasting":
        missing_data_kwargs_list = [
            {
                "missing_scenario": "none",
                "missing_values": 0,
            }
        ]
        config["missing_data_configs"] = missing_data_kwargs_list
    elif config["setup"] == "missing_values":
        missing_data_kwargs_list = config["missing_data_configs"]
    else:
        raise ValueError(f"Unknown setup {config['setup']}")

    Guidance = NoOPGuidance
    sampler_kwargs = config["sampler_params"]
    for missing_data_kwargs in missing_data_kwargs_list:
        logger.info(
            f"Evaluating scenario '{missing_data_kwargs['missing_scenario']}' "
            f"with {missing_data_kwargs['missing_values']:.1f} missing_values."
        )
        sampler = Guidance(
            model=model,
            prediction_length=config["prediction_length"],
            num_samples=num_samples,
            **missing_data_kwargs,
            **sampler_kwargs,
        )

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
            missing_data_kwargs["missing_scenario"],
            missing_data_kwargs["missing_values"],
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

def main(config: dict, log_dir: str, dataset_dir: str):
    # set hyperparameter for distillation
    num_steps = 12 # by micro batch, real batch size is 18 * 64. 60 for pre-computed dataset 
    gradient_clip_val = 0.1
    lr = 1e-4
    batch_size = 64
    masking_size = 24
    missing_data_kwargs = {
                "missing_scenario": "none",
                "missing_values": 0,
    }
    sampler_params = config["sampler_params"]
    sampling_batch_size = 64 * batch_size
    is_reg_loss = True
    is_fourier_loss = True
    is_dmd_loss = True
    reg_loss_lambda = 2.0
    fourier_loss_lambda = 2.0

    # Read global parameters
    dataset_name = config["dataset"]
    freq = config["freq"]
    prediction_length = config["prediction_length"]
    # num_samples = config["num_samples"]

    # Load dataset and model
    logger.info("Loading model")
    real_model = load_model(config)
    fake_model = load_model(config)
    one_step_model = load_model(config)
    device = next(real_model.backbone.parameters()).device

    # Set up for guidance
    sampler = DDIMGuidance(
            model=real_model,
            prediction_length=config["prediction_length"],
            num_samples=1,
            **missing_data_kwargs,
            **sampler_params,
        )

    optimizer_one_step = torch.optim.Adam(one_step_model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer_one_step, start_factor=0.01, total_iters=500)
    one_step_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_one_step, 50, gamma=0.99
        )
    one_step_scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[lr_scheduler, one_step_scheduler])
    if is_dmd_loss:
        optimizer_fake = torch.optim.Adam(fake_model.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer_fake, start_factor=0.01, total_iters=500)
        fake_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_fake, 50, gamma=0.99
        )
        fake_scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[lr_scheduler, fake_scheduler])
    
    best_reg_loss = float("inf")
    with trange(num_steps) as pbar:
        for step in pbar:
            # dataset file is not given, generate data on fly
            if dataset_dir is None:
                with torch.no_grad():
                    # generating data
                    gt_data = real_model.sample_n_grad(sampling_batch_size, return_lags=True)
                    # gt_data = torch.from_numpy(real_model.sample_n_grad(sampling_batch_size, return_lags=True))
                    # gt_data = gt_data.to(device)
                    masked_data = gt_data # (sampling_batch_size, seq_len, time_lags)
                    
                    # making observation mask
                    prior_mask = torch.ones((sampling_batch_size, max(real_model.lags_seq)), device=device) # (sampling_batch_size, time_lags - 1)
                    observation_mask = torch.ones_like(masked_data[:, :, 0]) # (sampling_batch_size, seq_len)
                    observation_mask[:,  -masking_size:] = 0.0
                    lagged_mask = lagged_sequence_values(
                        real_model.lags_seq,
                        prior_mask,
                        observation_mask,
                        dim=1,
                    ) # (sampling_batch_size, seq_len, time_lags - 1)
                    observation_mask = torch.cat([observation_mask[:, :, None], lagged_mask], dim=-1) # (sampling_batch_size, seq_len, time_lags)

                    # making prediction data
                    masked_data[observation_mask == 0] = torch.randn_like(masked_data[observation_mask == 0])
                    samples = sampler.guide(masked_data, observation_mask, None, sampler.scale, no_random_noise=True)

                masked_datas = masked_data.view(-1, batch_size, masked_data.shape[1], masked_data.shape[2]) # (sampling_batch_size // batch_size, batch_size, seq_len, time_lags)
                sampless = samples.view(-1, batch_size, samples.shape[1], samples.shape[2]) # (sampling_batch_size // batch_size, batch_size, seq_len, time_lags)
                observation_masks = observation_mask.view(-1, batch_size, observation_mask.shape[1], observation_mask.shape[2]) # (sampling_batch_size // batch_size, batch_size, seq_len, time_lags)
            # dataset file is given, load data from file
            else:
                masked_datas = torch.load(f"{dataset_dir}/z_{step}.pth").to(device)
                sampless = torch.load(f"{dataset_dir}/x_{step}.pth").to(device)

                # making observation mask
                prior_mask = torch.ones((masked_datas.shape[0] * masked_datas.shape[1], max(real_model.lags_seq)), device=device) # (sampling_batch_size, time_lags - 1)
                observation_mask = torch.ones_like(masked_datas.view(-1, masked_datas.shape[2], masked_datas.shape[3])[:, :, 0]) # (sampling_batch_size, seq_len)
                observation_mask[:,  -masking_size:] = 0.0
                lagged_mask = lagged_sequence_values(
                    real_model.lags_seq,
                    prior_mask,
                    observation_mask,
                    dim=1,
                ) # (sampling_batch_size, seq_len, time_lags - 1)
                observation_mask = torch.cat([observation_mask[:, :, None], lagged_mask], dim=-1) # (sampling_batch_size, seq_len, time_lags)
                observation_masks = observation_mask.view_as(masked_datas)

            loss_item_list = []
            dmd_loss_item_list = []
            for masked_data, samples, observation_mask in zip(tqdm(masked_datas, leave=False), sampless, observation_masks):
                optimizer_one_step.zero_grad()

                # calculating regression loss
                t = torch.full((batch_size,), one_step_model.timesteps - 1, device=device, dtype=torch.long)
                one_step_samples = one_step_model.fast_denoise(masked_data, t, features=None)
                loss = 0

                # reg_loss = torch.nn.functional.mse_loss(one_step_samples, samples)
                if is_reg_loss:
                    reg_loss = torch.nn.functional.mse_loss(one_step_samples[observation_mask == 0], samples[observation_mask == 0])
                    loss += reg_loss_lambda * reg_loss
                
                if is_fourier_loss:
                    fourier_reg_loss = fourier_loss(
                        one_step_samples[observation_mask == 0],
                        samples[observation_mask == 0]
                    )
                    loss += fourier_loss_lambda * fourier_reg_loss

                # calculating dmd loss
                if is_dmd_loss:
                    dmd_timestep = torch.randint(real_model.timesteps // 50, real_model.timesteps * 49 // 50, (batch_size,), device=device)
                    dmd_noise = masked_data.detach().clone()
                    dmd_noise[observation_mask == 0] = torch.randn_like(dmd_noise[observation_mask == 0])
                    dmd_x = one_step_model.fast_denoise(dmd_noise, t, features=None)
                    with torch.no_grad():
                        noisy_x = forward_diffusion_with_mask(real_model, dmd_x, dmd_timestep, observation_mask)
                        pred_real_timesteps = torch.cat([dmd_timestep[:, None], torch.zeros_like(dmd_timestep[:, None])], dim=-1)
                        # pred_real = sampler.guide(noisy_x, observation_mask, None, sampler.scale, timesteps=pred_real_timesteps, no_random_noise=True)
                        # pred_real = sampler.guide_fast(noisy_x, observation_mask, dmd_timestep, None, base_scale=sampler.scale)
                        # pred_real = sampler.guided_noise(noisy_x, observation_mask, dmd_timestep, None, base_scale=sampler.scale)
                        pred_real = real_model.backbone(noisy_x, dmd_timestep, None)
                        # denoise_real = real_model.fast_denoise(noisy_x, t, features=None)
                        
                        # pred_fake = fake_model.fast_denoise(noisy_x, dmd_timestep, features=None)
                        pred_fake = fake_model.backbone(noisy_x, dmd_timestep, None)
                        weighting_factor = (dmd_x[observation_mask == 0] - pred_real[observation_mask == 0]).view(batch_size, -1).abs().mean(dim=1) + 1e-9
                        # weighting_factor = (dmd_x[observation_mask == 0] - denoise_real[observation_mask == 0]).view(batch_size, -1).abs().mean(dim=1) + 1e-9
                        weighting_factor = weighting_factor.view(-1, 1, 1).expand_as(dmd_x)
                        # grad = (pred_fake - pred_real) / weighting_factor # original
                        grad = (pred_real - pred_fake) / weighting_factor # reversed
                        target = dmd_x - grad

                    # dmd_loss = 0.5 * torch.nn.functional.mse_loss(dmd_x, target)
                    dmd_loss = 0.5 * torch.nn.functional.mse_loss(dmd_x[observation_mask == 0], target[observation_mask == 0])
                    loss += dmd_loss
                    dmd_loss_item_list.append(dmd_loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(one_step_model.parameters(), gradient_clip_val)
                optimizer_one_step.step()
                one_step_scheduler.step()

                # calculating denoising loss for fake model
                if is_dmd_loss:
                    for _ in range(10):
                        dmd_timestep = torch.randint(0, fake_model.timesteps, (batch_size,), device=device)
                        dmd_x = dmd_x.detach()
                        with torch.no_grad():
                            noisy_x = forward_diffusion_with_mask(fake_model, dmd_x, dmd_timestep, observation_mask)
                        pred_fake = fake_model.fast_denoise(noisy_x, dmd_timestep, features=None)
                        denoising_loss = torch.nn.functional.mse_loss(pred_fake, dmd_x)
                        # denoising_loss = torch.nn.functional.mse_loss(pred_fake[observation_mask == 0], dmd_x[observation_mask == 0])
                        optimizer_fake.zero_grad()
                        denoising_loss.backward()
                        torch.nn.utils.clip_grad_norm_(fake_model.parameters(), gradient_clip_val)
                        optimizer_fake.step()
                        fake_scheduler.step()
                loss_item_list.append(loss.item())
                postfix = {'loss': loss.item()}
                if is_reg_loss: postfix['reg'] = reg_loss.item()
                if is_fourier_loss: postfix['fourier'] = fourier_reg_loss.item()
                if is_dmd_loss: postfix['dmd'] = dmd_loss.item()
                pbar.set_postfix(postfix)

            if is_dmd_loss:
                tqdm.write(f"Step {step}: loss {sum(loss_item_list) / len(loss_item_list)}, dmd_loss {sum(dmd_loss_item_list) / len(dmd_loss_item_list)}, reg_loss {sum(loss_item_list) / len(loss_item_list) - sum(dmd_loss_item_list) / len(dmd_loss_item_list)}")
                reg_loss = sum(loss_item_list) / len(loss_item_list) - sum(dmd_loss_item_list) / len(dmd_loss_item_list)
            else:
                tqdm.write(f"Step {step}: loss {sum(loss_item_list) / len(loss_item_list)}")
                reg_loss = sum(loss_item_list) / len(loss_item_list)

            if reg_loss < best_reg_loss:
                best_reg_loss = reg_loss
                # best_model = deepcopy(one_step_model.state_dict())

    # one_step_model.load_state_dict(best_model)
    if True: # for evaluation code
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
            config, one_step_model, dataset.test, transformation, num_samples=100
        )
        print(results)
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
                {"config": config, "metrics": results, "dmd_loss": is_dmd_loss},
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
    parser.add_argument(
        "--dataset_dir", type=str, default=None, help="Path to precomputed dataset dir"
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

    main(config=config, log_dir=args.out_dir, dataset_dir=args.dataset_dir)
