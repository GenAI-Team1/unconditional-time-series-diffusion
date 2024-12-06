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
import os
from torch.distributed import init_process_group, all_reduce, all_gather

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
    return model

def main(config: dict, log_dir: str):
    # set hyperparameter for distillation
    num_steps = 300 # by micro batch, real batch size is 18 * 64
    batch_size = 64
    masking_size = 24
    missing_data_kwargs = {
                "missing_scenario": "none",
                "missing_values": 0,
    }
    sampler_params = config["sampler_params"]
    sampling_batch_size = 64 * batch_size
    
    # set up DDP (distributed data parallel). torchrun sets this env variable
    is_ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

    # Load dataset and model
    logger.info("Loading model")
    real_model = load_model(config)

    if is_ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        
        # initialize DDP
        torch.distributed.init_process_group(backend='nccl')

        # parameters for DDP
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        real_model = real_model.to(device)
        # real_model = torch.nn.parallel.DistributedDataParallel(real_model, device_ids=[ddp_local_rank])        
        is_main_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        is_main_process = True
        seed_offset = 0

       # attempt to autodetect the device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
    
    torch.manual_seed(42 + ddp_local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + ddp_local_rank)
 
    # Set up for guidance
    sampler = DDIMGuidance(
            model=real_model,
            prediction_length=config["prediction_length"],
            num_samples=1,
            **missing_data_kwargs,
            **sampler_params,
        )
    
    x_list = []
    z_list = []
    idx = 0
    if is_main_process:
        print('start generating data')
    with trange(num_steps, disable=not is_main_process) as pbar:
        for step in pbar:
            with torch.no_grad():
                if is_ddp:
                    torch.distributed.barrier()
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
                if is_ddp:
                    masked_datas_gathered = [torch.zeros_like(masked_datas) for _ in range(ddp_world_size)]
                    sampless_gathered = [torch.zeros_like(sampless) for _ in range(ddp_world_size)]
                    all_gather(masked_datas_gathered, masked_datas)
                    all_gather(sampless_gathered, sampless)
                
                if is_main_process:
                    if is_ddp:
                        masked_datas = torch.cat(masked_datas_gathered, dim=0)
                        sampless = torch.cat(sampless_gathered, dim=0) 
                    z_list.append(masked_datas.detach().cpu())
                    x_list.append(sampless.detach().cpu())

                if is_main_process and (step + 1) % 5 == 0:
                    x = torch.cat(x_list, dim=0)
                    z = torch.cat(z_list, dim=0)
                    torch.save(x, f"data/x_{idx}.pth")
                    torch.save(z, f"data/z_{idx}.pth")
                    idx += 1
                    x_list = []
                    z_list = []

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
