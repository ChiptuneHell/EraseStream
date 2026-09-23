"""Long-video web demo for StreamErase.

This is intentionally a separate entry point from ``web_demo.py``.  The
original demo remains available while this page follows the long-video
inference path in ``inference_long.py`` (21/42/63 latent frames).

Run from the repository root:

    python web_demo_long.py --host 0.0.0.0 --port 5001

The long-video pipeline publishes each decoded causal block as JPEG frames
while generation is running.  It still writes a final H.264 MP4 at the end so
the browser can switch from the live canvas to synchronized video playback.
"""

import argparse
import base64
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from einops import rearrange
from flask import Flask, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from omegaconf import OmegaConf
from torchvision.io import write_video

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from model.wan_vae import AutoencoderKLWan
from pipeline.causal_inference_long import CausalInferencePipeline
from utils.utils import get_video_and_mask
from utils.misc import set_seed


ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "long_test_input" / "video"
MASK_DIR = ROOT / "long_test_input" / "mask"
PREVIEW_DIR = ROOT / "long_test_input" / "preview"
RESULT_DIR = ROOT / "web" / "results_long"

app = Flask(
    __name__,
    template_folder=str(ROOT / "web" / "templates_long"),
    static_folder=str(ROOT / "web" / "static_long"),
)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.bfloat16
MODEL = None
PREPROCESS_VAE = None
MODEL_READY = False
MODEL_ERROR = None
PREP_STARTED_AT = None
PREP_FINISHED_AT = None
ACTIVE = False
JOB_LOCK = threading.Lock()
PREP_LOCK = threading.Lock()

CONFIG_PATH = ROOT / "configs" / "causal_forcing_dmd_chunkwise.yaml"
CHECKPOINT_PATH = ROOT / "model_weights" / "model.pt"
VAE_PATH = ROOT / "wan_models" / "Wan2.1-T2V-1.3B" / "Wan2.1_VAE.pth"
LOCAL_ATTN_SIZE = 12
SINK_SIZE = 3
WINDOW_ROPE = False
PROMPT = "Remove the specified object and all related effects, then restore a clean background"

ALLOWED_LATENT_FRAMES = (21, 42, 63)
OUTPUT_FPS = 16


def _video_names():
    return sorted(p.name for p in VIDEO_DIR.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}) if VIDEO_DIR.exists() else []


def prepare_mask_latents(masked_image, device, vae):
    """Encode the source video with the same VAE path as inference_long.py."""
    masked_image = masked_image.to(device=device, dtype=vae.dtype)
    encoded = []
    for start in range(0, masked_image.shape[0], 1):
        posterior = vae.encode(masked_image[start : start + 1])[0]
        encoded.append(posterior.mode())
    return torch.cat(encoded, dim=0)


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    _, _, _, height, width = mask.shape
    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first = F.interpolate(mask[:, :, 0:1], size=target_size, mode="trilinear", align_corners=False)

        target_size = list(latent_size[2:])
        target_size[0] -= 1
        if target_size[0] != 0:
            rest = F.interpolate(mask[:, :, 1:], size=target_size, mode="trilinear", align_corners=False)
            return torch.cat([first, rest], dim=2)
        return first

    return F.interpolate(mask, size=list(latent_size[2:]), mode="trilinear", align_corners=False)


def fit_video_length(tensor, target_frames):
    """Trim or repeat the final frame so video and mask have exact model length."""
    if tensor is None or tensor.shape[2] == 0:
        raise ValueError("The selected source or mask video contains no readable frames")
    if tensor.shape[2] >= target_frames:
        return tensor[:, :, :target_frames]
    missing = target_frames - tensor.shape[2]
    tail = tensor[:, :, -1:].expand(-1, -1, missing, -1, -1)
    return torch.cat([tensor, tail], dim=2)


def load_model():
    global MODEL, PREPROCESS_VAE
    if MODEL is not None and PREPROCESS_VAE is not None:
        return MODEL, PREPROCESS_VAE
    if DEVICE != "cuda":
        raise RuntimeError("The long-video demo requires a CUDA GPU")

    PREPROCESS_VAE = AutoencoderKLWan.from_pretrained(str(VAE_PATH)).to(device=DEVICE, dtype=WEIGHT_DTYPE)
    PREPROCESS_VAE.requires_grad_(False)

    config = OmegaConf.merge(
        OmegaConf.load(str(ROOT / "configs" / "default_config.yaml")),
        OmegaConf.load(str(CONFIG_PATH)),
    )
    MODEL = CausalInferencePipeline(
        config,
        device=DEVICE,
        local_attn_size=LOCAL_ATTN_SIZE,
        sink_size=SINK_SIZE,
        window_rope=WINDOW_ROPE,
    )

    state = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    weights = state.get("generator_ema", state.get("generator", state))
    try:
        MODEL.generator.load_state_dict(weights)
    except RuntimeError:
        fixed = {
            key.replace("model._fsdp_wrapped_module.", "model.", 1): value
            for key, value in weights.items()
        }
        MODEL.generator.load_state_dict(fixed, strict=False)

    MODEL.to(device=DEVICE, dtype=WEIGHT_DTYPE).eval()
    PREPROCESS_VAE.eval()
    if get_cuda_free_memory_gb(gpu) < 40:
        DynamicSwapInstaller.install_model(MODEL.text_encoder, device=gpu)
    else:
        MODEL.text_encoder.to(DEVICE)
    MODEL.generator.to(DEVICE)
    MODEL.vae.to(DEVICE)
    return MODEL, PREPROCESS_VAE


def prepare_model():
    global MODEL_READY, MODEL_ERROR, PREP_STARTED_AT, PREP_FINISHED_AT
    with PREP_LOCK:
        if MODEL_READY:
            return
        PREP_STARTED_AT = time.perf_counter()
        try:
            model, _ = load_model()
            with torch.inference_mode():
                model._initialize_kv_cache(batch_size=1, dtype=WEIGHT_DTYPE, device=DEVICE)
                model._initialize_crossattn_cache(batch_size=1, dtype=WEIGHT_DTYPE, device=DEVICE)
                torch.cuda.synchronize()
            MODEL_READY = True
        except Exception as exc:  # surfaced in /api/status and the page
            MODEL_ERROR = str(exc)
            print(f"Long-video model preparation failed: {exc}")
        finally:
            PREP_FINISHED_AT = time.perf_counter()


def run_job(video_name, latent_frames, job_id):
    global ACTIVE
    started = time.perf_counter()
    try:
        model, vae = load_model()
        target_pixel_frames = latent_frames * 4 - 3
        video_path = VIDEO_DIR / video_name
        mask_path = MASK_DIR / video_name
        if not mask_path.exists():
            raise FileNotFoundError(f"Matching mask video was not found: {mask_path.name}")

        socketio.emit("progress", {"job_id": job_id, "progress": 8, "message": "Reading source and mask"})
        input_video, input_mask, _, _ = get_video_and_mask(
            input_video_path=str(video_path),
            input_mask_path=str(mask_path),
            video_length=target_pixel_frames,
            sample_size=[480, 832],
        )
        input_video = fit_video_length(input_video, target_pixel_frames).to(DEVICE)
        input_mask = fit_video_length(input_mask, target_pixel_frames).to(DEVICE)
        _, _, video_length, height, width = input_video.shape

        image_processor = VaeImageProcessor(vae_scale_factor=8)
        mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        init_video = image_processor.preprocess(
            rearrange(input_video, "b c f h w -> (b f) c h w"),
            height=height,
            width=width,
        )
        init_video = rearrange(init_video.float(), "(b f) c h w -> b c f h w", f=video_length)
        mask_condition = mask_processor.preprocess(
            rearrange(input_mask, "b c f h w -> (b f) c h w"),
            height=height,
            width=width,
        )
        mask_condition = rearrange(mask_condition.float(), "(b f) c h w -> b c f h w", f=video_length)

        socketio.emit("progress", {"job_id": job_id, "progress": 18, "message": "Encoding removal condition"})
        masked_video_latents = prepare_mask_latents(init_video, DEVICE, vae)
        mask_condition = torch.cat(
            [torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2), mask_condition[:, :, 1:]],
            dim=2,
        )
        mask_condition = mask_condition.view(1, mask_condition.shape[2] // 4, 4, height, width).transpose(1, 2)
        mask_latents = resize_mask(1 - mask_condition, masked_video_latents, True).to(DEVICE, WEIGHT_DTYPE)
        y = torch.cat([mask_latents, masked_video_latents], dim=1).to(DEVICE, WEIGHT_DTYPE)

        noise = torch.randn(
            [1, latent_frames, 16, 60, 104],
            device=DEVICE,
            dtype=WEIGHT_DTYPE,
        )
        socketio.emit("progress", {"job_id": job_id, "progress": 25, "message": "Generating long video"})
        socketio.emit(
            "stream_started",
            {
                "job_id": job_id,
                "fps": OUTPUT_FPS,
                "total_frames": target_pixel_frames,
                "latent_frames": latent_frames,
            },
        )
        generation_started = time.perf_counter()

        stream_frame_cursor = 0
        num_blocks = max(
            1,
            (latent_frames + max(1, int(getattr(model, "num_frame_per_block", 1))) - 1)
            // max(1, int(getattr(model, "num_frame_per_block", 1))),
        )

        def emit_stream_block(pixel_block, block_index, _latent_start, is_last):
            """Encode one decoded VAE block and publish ordered JPEG frames."""
            nonlocal stream_frame_cursor
            frames = (
                pixel_block[0]
                .detach()
                .float()
                .mul(255.0)
                .clamp(0, 255)
                .round()
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            for frame in frames:
                # The VAE exposes RGB tensors while OpenCV encodes BGR.
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 86],
                )
                if not encoded_ok:
                    raise RuntimeError("Could not encode a streamed output frame")
                socketio.emit(
                    "stream_frame",
                    {
                        "job_id": job_id,
                        "frame_index": stream_frame_cursor,
                        "jpeg": base64.b64encode(encoded.tobytes()).decode("ascii"),
                        "fps": OUTPUT_FPS,
                    },
                )
                stream_frame_cursor += 1

            # The generation bar measures model generation and streamed VAE/JPEG
            # work only.  It reaches 100% when the final block is available;
            # writing the final MP4 below is reported as a separate status.
            stream_progress = 25 + round(75 * (block_index + 1) / num_blocks)
            stream_seconds = time.perf_counter() - generation_started
            stream_fps = stream_frame_cursor / max(stream_seconds, 1e-6)
            socketio.emit(
                "progress",
                {
                    "job_id": job_id,
                    "progress": min(100, stream_progress),
                    "message": "Streaming output" if not is_last else "Finalizing output",
                    "stream_frame": stream_frame_cursor,
                    "fps": round(stream_fps, 2),
                    "realtime": round(stream_fps / OUTPUT_FPS, 2),
                    "elapsed": round(time.perf_counter() - started, 2),
                },
            )
            if is_last:
                socketio.emit(
                    "stream_finished",
                    {"job_id": job_id, "frames": stream_frame_cursor},
                )

        with torch.inference_mode():
            video_out = model.inference(
                noise=noise,
                y=y,
                text_prompts=[PROMPT],
                return_latents=False,
                report_timing=True,
                block_callback=emit_stream_block,
            )
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started

        # MP4 muxing happens after generation has completed and must not make
        # the generation progress bar appear to stall below completion.
        socketio.emit("progress", {"job_id": job_id, "progress": 100, "message": "Saving result video"})
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULT_DIR / f"{job_id}.mp4"
        temporary_path = RESULT_DIR / f"{job_id}.tmp.mp4"
        # ``causal_inference_long`` returns [B, T, C, H, W], while
        # torchvision.write_video expects [T, H, W, C].  Keeping the channel
        # dimension in the second position produces the ``Unexpected numpy
        # array shape (3, 480, 832)`` error when torchvision converts frames.
        frames_out = (
            rearrange(video_out[0], "t c h w -> t h w c") * 255.0
        ).clamp(0, 255).round().to(torch.uint8).cpu()
        write_video(str(temporary_path), frames_out, fps=OUTPUT_FPS)
        os.replace(temporary_path, output_path)
        if hasattr(model.vae.model, "clear_cache"):
            model.vae.model.clear_cache()

        elapsed = time.perf_counter() - started
        frames = int(frames_out.shape[0])
        generation_fps = frames / max(generation_seconds, 1e-6)
        socketio.emit(
            "complete",
            {
                "job_id": job_id,
                "url": f"long_results/{output_path.name}",
                "elapsed": round(elapsed, 2),
                "generation_seconds": round(generation_seconds, 2),
                "fps": round(float(generation_fps), 2),
                "realtime": round(float(generation_fps / OUTPUT_FPS), 2),
                "frames": frames,
                "duration": round(frames / OUTPUT_FPS, 2),
                "latent_frames": latent_frames,
            },
        )
    except Exception as exc:
        print(f"Long-video job failed: {exc}")
        socketio.emit("job_error", {"job_id": job_id, "message": str(exc)})
    finally:
        ACTIVE = False


@app.get("/")
def index():
    presets = [name for name in _video_names() if (MASK_DIR / name).exists()]
    return render_template("long_index.html", presets=presets, latent_frames=ALLOWED_LATENT_FRAMES)


@app.get("/long_input/video/<path:name>")
def long_input_video(name):
    return send_from_directory(str(VIDEO_DIR), name)


@app.get("/long_preview/<path:name>")
def long_preview(name):
    # Preview files are generated by preprocess_preview.py.  Falling back to
    # the source keeps the page usable before the optional visual asset exists.
    preview_path = PREVIEW_DIR / f"{name}.preview.mp4"
    directory = PREVIEW_DIR if preview_path.exists() else VIDEO_DIR
    return send_from_directory(str(directory), preview_path.name if preview_path.exists() else name)


@app.get("/long_preview/poster/<path:name>")
def long_preview_poster(name):
    poster_path = PREVIEW_DIR / f"{name}.poster.jpg"
    if poster_path.exists():
        return send_from_directory(str(PREVIEW_DIR), poster_path.name)
    # A missing poster is harmless; the video element will show its first frame
    # once the preview asset is available.
    return ("", 404)


@app.get("/long_input/mask/<path:name>")
def long_input_mask(name):
    return send_from_directory(str(MASK_DIR), name)


@app.get("/long_results/<path:name>")
def long_results(name):
    return send_from_directory(str(RESULT_DIR), name)


@app.get("/api/status")
def status():
    prep_seconds = None
    if PREP_STARTED_AT is not None:
        prep_seconds = round((PREP_FINISHED_AT or time.perf_counter()) - PREP_STARTED_AT, 2)
    return jsonify(
        {
            "active": ACTIVE,
            "device": DEVICE,
            "cuda": torch.cuda.is_available(),
            "model_ready": MODEL_READY,
            "model_error": MODEL_ERROR,
            "preparation_seconds": prep_seconds,
            "presets": _video_names(),
            "latent_frames": ALLOWED_LATENT_FRAMES,
            "output_fps": OUTPUT_FPS,
        }
    )


@socketio.on("start")
def start(data):
    global ACTIVE
    with JOB_LOCK:
        if ACTIVE:
            emit("job_error", {"message": "Another generation is already running"})
            return
        video_name = str(data.get("video", ""))
        try:
            latent_frames = int(data.get("latent_frames", 21))
        except (TypeError, ValueError):
            latent_frames = 21
        if video_name not in _video_names() or not (MASK_DIR / video_name).exists():
            emit("job_error", {"message": "The selected source/mask pair was not found"})
            return
        if latent_frames not in ALLOWED_LATENT_FRAMES:
            emit("job_error", {"message": "Unsupported output length"})
            return
        if not MODEL_READY:
            emit("job_error", {"message": MODEL_ERROR or "The GPU model is still preparing"})
            return
        ACTIVE = True
        job_id = uuid.uuid4().hex
        # Announce the ID before the worker can publish progress or frames.
        emit("started", {"job_id": job_id, "latent_frames": latent_frames})
        threading.Thread(
            target=run_job,
            args=(video_name, latent_frames, job_id),
            daemon=True,
            name=f"long-generation-{job_id[:8]}",
        ).start()


def main():
    global VIDEO_DIR, MASK_DIR, PREVIEW_DIR, CONFIG_PATH, CHECKPOINT_PATH, VAE_PATH
    global LOCAL_ATTN_SIZE, SINK_SIZE, WINDOW_ROPE, PROMPT

    parser = argparse.ArgumentParser(description="StreamErase long-video web demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--video_dir", default=str(VIDEO_DIR))
    parser.add_argument("--mask_dir", default=str(MASK_DIR))
    parser.add_argument("--preview_dir", default=str(PREVIEW_DIR))
    parser.add_argument("--config_path", default=str(CONFIG_PATH))
    parser.add_argument("--checkpoint_path", default=str(CHECKPOINT_PATH))
    parser.add_argument("--vae_path", default=str(VAE_PATH))
    parser.add_argument("--local_attn_size", type=int, default=12)
    parser.add_argument("--sink_size", type=int, default=3)
    parser.add_argument("--window_rope", action="store_true")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    VIDEO_DIR = Path(args.video_dir).resolve()
    MASK_DIR = Path(args.mask_dir).resolve()
    PREVIEW_DIR = Path(args.preview_dir).resolve()
    CONFIG_PATH = Path(args.config_path).resolve()
    CHECKPOINT_PATH = Path(args.checkpoint_path).resolve()
    VAE_PATH = Path(args.vae_path).resolve()
    LOCAL_ATTN_SIZE = args.local_attn_size
    SINK_SIZE = args.sink_size
    WINDOW_ROPE = args.window_rope
    PROMPT = args.prompt
    set_seed(args.seed)

    print(f"Long-video demo: http://{args.host}:{args.port}")
    print(f"Input: {VIDEO_DIR}")
    print(f"Lengths: {', '.join(str(v) for v in ALLOWED_LATENT_FRAMES)} latent frames")
    threading.Thread(target=prepare_model, daemon=True, name="long-model-preload").start()
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
