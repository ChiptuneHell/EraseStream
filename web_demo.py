"""Small local web demo for StreamErase.

Run with ``python web_demo.py`` and open http://127.0.0.1:5001.
The model is loaded lazily when the first job is started.
"""
import argparse, os, threading, time, uuid
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video
from diffusers.image_processor import VaeImageProcessor
import torch.nn.functional as F

from pipeline import CausalInferencePipeline, CausalDiffusionInferencePipeline
from model.wan_vae import AutoencoderKLWan
from utils.utils import get_video_and_mask
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

ROOT = Path(__file__).parent
app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None
VAE = None
JOB_LOCK = threading.Lock()
ACTIVE = False


def load_model():
    global MODEL, VAE
    if MODEL is not None:
        return MODEL, VAE
    if DEVICE != "cuda":
        raise RuntimeError("需要 CUDA GPU 才能运行擦除模型")
    dtype = torch.bfloat16
    VAE = AutoencoderKLWan.from_pretrained(str(ROOT / "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")).to(DEVICE, dtype=dtype)
    VAE.requires_grad_(False)
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "configs/default_config.yaml"),
                          OmegaConf.load(ROOT / "configs/causal_forcing_dmd_chunkwise.yaml"))
    MODEL = CausalInferencePipeline(cfg, device=DEVICE)
    state = torch.load(ROOT / "model_weights/model.pt", map_location="cpu")
    weights = state.get("generator_ema", state.get("generator", state))
    try:
        MODEL.generator.load_state_dict(weights)
    except RuntimeError:
        MODEL.generator.load_state_dict({k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in weights.items()}, strict=False)
    MODEL.to(device=DEVICE, dtype=dtype).eval()
    if get_cuda_free_memory_gb(gpu) < 40:
        DynamicSwapInstaller.install_model(MODEL.text_encoder, device=gpu)
    else:
        MODEL.text_encoder.to(DEVICE)
    MODEL.generator.to(DEVICE); MODEL.vae.to(DEVICE)
    return MODEL, VAE


def resize_mask(mask, latent):
    size = list(latent.shape[2:])
    first = F.interpolate(mask[:, :, :1], size=[1, size[1], size[2]], mode="trilinear", align_corners=False)
    rest = F.interpolate(mask[:, :, 1:], size=[max(size[0]-1, 0), size[1], size[2]], mode="trilinear", align_corners=False) if size[0] > 1 else None
    return torch.cat([first, rest], dim=2) if rest is not None else first


def run_job(video_name, sio_job):
    global ACTIVE
    started = time.perf_counter()
    try:
        model, vae = load_model()
        socketio.emit("progress", {"job_id": sio_job, "progress": 8, "message": "读取视频和 mask"})
        vp = ROOT / "test_input/video" / video_name
        mp = ROOT / "test_input/mask" / video_name
        video, mask, _, _ = get_video_and_mask(str(vp), video_length=81, sample_size=[480, 832], input_mask_path=str(mp))
        video, mask = video.to(DEVICE), mask.to(DEVICE)
        _, _, frames, height, width = video.shape
        image_proc = VaeImageProcessor(vae_scale_factor=8)
        mask_proc = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        init = rearrange(image_proc.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width), "(b f) c h w -> b c f h w", f=frames).float()
        m = rearrange(mask_proc.preprocess(rearrange(mask, "b c f h w -> (b f) c h w"), height=height, width=width), "(b f) c h w -> b c f h w", f=frames).float()
        socketio.emit("progress", {"job_id": sio_job, "progress": 18, "message": "编码条件和 mask"})
        masked_latents = vae.encode(init.to(dtype=vae.dtype))[0].mode()
        m = torch.cat([torch.repeat_interleave(m[:, :, :1], 4, dim=2), m[:, :, 1:]], dim=2)
        m = m.view(1, m.shape[2] // 4, 4, height, width).transpose(1, 2)
        mask_latents = resize_mask(1 - m, masked_latents).to(DEVICE, torch.bfloat16)
        y = torch.cat([mask_latents, masked_latents], dim=1)
        noise = torch.randn([1, 21, 16, 60, 104], device=DEVICE, dtype=torch.bfloat16)
        socketio.emit("progress", {"job_id": sio_job, "progress": 25, "message": "开始因果分块生成"})
        with torch.inference_mode():
            out = model.inference(noise=noise, y=y, text_prompts=["Remove the specified object and all related effects, then restore a clean background"], return_latents=False)
        # Keep progress visible while the GPU call finishes; the output is then immediately playable.
        socketio.emit("progress", {"job_id": sio_job, "progress": 92, "message": "写出结果视频"})
        result_dir = ROOT / "web/results"; result_dir.mkdir(parents=True, exist_ok=True)
        out_path = result_dir / f"{sio_job}.mp4"
        frames_out = (rearrange(out, "b t c h w -> b t h w c")[0] * 255).clamp(0, 255).to(torch.uint8).cpu()
        write_video(str(out_path), frames_out, fps=16)
        elapsed = time.perf_counter() - started
        fps = frames_out.shape[0] / elapsed
        socketio.emit("complete", {"job_id": sio_job, "url": f"/results/{out_path.name}", "elapsed": round(elapsed, 2), "fps": round(float(fps), 2), "frames": int(frames_out.shape[0]), "realtime": round(float(fps / 16), 2)})
    except Exception as exc:
        socketio.emit("job_error", {"job_id": sio_job, "message": str(exc)})
    finally:
        ACTIVE = False


@app.get("/")
def index():
    presets = sorted(p.name for p in (ROOT / "test_input/video").glob("*.mp4"))
    return render_template("index.html", presets=presets)

@app.get("/results/<path:name>")
def results(name):
    return send_from_directory(ROOT / "web/results", name)

@app.get("/test_input/<path:name>")
def test_input(name):
    return send_from_directory(ROOT / "test_input", name)

@app.get("/api/status")
def status():
    return jsonify({"active": ACTIVE, "device": DEVICE, "cuda": torch.cuda.is_available()})

@socketio.on("start")
def start(data):
    global ACTIVE
    with JOB_LOCK:
        if ACTIVE:
            emit("job_error", {"message": "已有任务正在运行"}); return
        name = data.get("video")
        if name not in [p.name for p in (ROOT / "test_input/video").glob("*.mp4")]:
            emit("job_error", {"message": "找不到所选视频"}); return
        ACTIVE = True; job = uuid.uuid4().hex
        threading.Thread(target=run_job, args=(name, job), daemon=True).start()
        emit("started", {"job_id": job})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args(); print(f"Demo: http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
