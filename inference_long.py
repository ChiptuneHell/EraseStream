import argparse
import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)

from pipeline.causal_inference_long import CausalInferencePipeline as CausalInferencePipeline2

from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

#########################################################################################
from utils.utils import get_video_and_mask
from diffusers.image_processor import VaeImageProcessor
import torch.nn.functional as F
import sys
from model.wan_vae import AutoencoderKLWan
import numpy as np
from einops import rearrange
from torchvision.io import write_video
##########################################################################################

def prepare_mask_latents(mask, masked_image, device, vae):
    if mask is not None:
        mask = mask.to(device=device, dtype=vae.dtype)
        bs = 1
        new_mask = []
        for i in range(0, mask.shape[0], bs):
            mask_bs = mask[i : i + bs]
            mask_bs = vae.encode(mask_bs)[0]
            mask_bs = mask_bs.mode()
            new_mask.append(mask_bs)
        mask = torch.cat(new_mask, dim=0)

    if masked_image is not None:
        masked_image = masked_image.to(device=device, dtype=vae.dtype)
        bs = 1
        new_mask_pixel_values = []
        for i in range(0, masked_image.shape[0], bs):
            mask_pixel_values_bs = masked_image[i : i + bs]
            mask_pixel_values_bs = vae.encode(mask_pixel_values_bs)[0]
            mask_pixel_values_bs = mask_pixel_values_bs.mode()
            new_mask_pixel_values.append(mask_pixel_values_bs)
        masked_image_latents = torch.cat(new_mask_pixel_values, dim=0)
    else:
        masked_image_latents = None

    return mask, masked_image_latents

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="long_test_input/video")
    parser.add_argument("--mask_dir", type=str, default="long_test_input/mask")
    parser.add_argument("--output_folder", type=str, default="results_long")
    parser.add_argument("--prompt", type=str, default="Remove the specified object and all related effects, then restore a clean background")
    parser.add_argument("--config_path", type=str, default="configs/causal_forcing_dmd_chunkwise.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="../Erase-diffusion/logs/dmd_1step/checkpoint_model_001000/model.pt")
    parser.add_argument("--num_output_frames", type=int, default=63) # 可修改，让用户自己在页面改，比如用户想生成10秒的视频，这个值就是42

    # 21 = 5s; 42 = 10s; 63 = 15s 

    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--local_attn_size", type=int, default=12, help="Local attention size (-1 表示不开启滑动窗口)")
    parser.add_argument("--sink_size", type=int, default=3, help="保留的持久 Sink 帧数量")
    parser.add_argument("--window_rope", action="store_true", help="是否对 Rolling KV Window 进行 RoPE 重基准")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16
    set_seed(args.seed)
    vae = AutoencoderKLWan.from_pretrained("wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth").to(device=device, dtype=weight_dtype)
    vae.requires_grad_(False)

    image_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    local_attn_size = args.local_attn_size

    if hasattr(config, 'denoising_step_list'):
        pipeline = CausalInferencePipeline2(config, device=device)
    else:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)

    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        # key = 'generator_ema' if args.use_ema else 'generator'
        key = 'generator_ema'
        gen_sd = state_dict[key]
        try:
            pipeline.generator.load_state_dict(gen_sd)
        except RuntimeError:
            fixed = {}
            for k, v in gen_sd.items():
                if k.startswith("model._fsdp_wrapped_module."):
                    k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                fixed[k] = v
            pipeline.generator.load_state_dict(fixed, strict=False)

    pipeline = pipeline.to(dtype=weight_dtype)
    
    low_memory = get_cuda_free_memory_gb(gpu) < 40
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    torch.set_grad_enabled(False)
    os.makedirs(args.output_folder, exist_ok=True)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = sorted([f for f in os.listdir(args.video_dir) if f.lower().endswith(video_extensions)])
    
    print(f"找到 {len(video_files)} 个视频文件开始进行批量处理...")

    # 3. 循环批处理
    for video_name in tqdm(video_files):
        video_path = os.path.join(args.video_dir, video_name)
        mask_path = os.path.join(args.mask_dir, video_name) # 保持同名
        output_path = os.path.join(args.output_folder, video_name)

        if os.path.exists(output_path):
            print(f"视频 {video_name} 已经存在，跳过！")
            continue

        if not os.path.exists(mask_path):
            print(f"[WARN] 找不到匹配的 Mask 视频: {mask_path}，跳过该文件")
            continue

        try:
            target_pixel_frames = int(args.num_output_frames * 4 - 3)
            input_video, input_mask, _, _ = get_video_and_mask(
                input_video_path=video_path,
                input_mask_path=mask_path,
                video_length=target_pixel_frames,     
                sample_size=[480, 832]  
            )
        except Exception as e:
            print(f"读取视频对时出错 {video_name}: {e}")
            continue

        input_video = input_video.to(device=device)
        input_mask = input_mask.to(device=device)
        bs, _, video_length, height, width = input_video.size()

        init_video = image_processor.preprocess(rearrange(input_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
        init_video = init_video.to(dtype=torch.float32)
        init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)

        mask_condition = mask_processor.preprocess(rearrange(input_mask, "b c f h w -> (b f) c h w"), height=height, width=width) 
        mask_condition = mask_condition.to(dtype=torch.float32)
        mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

        masked_video = init_video

        _, masked_video_latents = prepare_mask_latents(
            None,
            masked_video,
            device,
            vae,
        )

        mask_condition = torch.concat(
            [
                torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2), 
                mask_condition[:, :, 1:]
            ], dim=2
        )
        mask_condition = mask_condition.view(bs, mask_condition.shape[2] // 4, 4, height, width)
        mask_condition = mask_condition.transpose(1, 2)
        mask_latents = resize_mask(1 - mask_condition, masked_video_latents, True).to(device, weight_dtype) 

        do_classifier_free_guidance = False
        mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
        masked_video_latents_input = (
            torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
        )
        y = torch.cat([mask_input, masked_video_latents_input], dim=1).to(device, weight_dtype)

        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

        video_out, latents = pipeline.inference(
            noise=sampled_noise,
            y=y,
            text_prompts=[args.prompt],
            return_latents=True,
            initial_latent=None,
        )

        current_video = rearrange(video_out, 'b t c h w -> b t h w c').cpu()
        video_final = 255.0 * current_video

        pipeline.vae.model.clear_cache()
        write_video(output_path, video_final[0], fps=16)

if __name__ == "__main__":
    main()