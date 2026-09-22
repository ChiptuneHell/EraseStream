"""Pre-render orange mask previews. No GPU or model weights are required.

Run ``python preprocess_preview.py``; unchanged inputs reuse the cached MP4
and poster. These assets are for display only, never model conditioning.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = ROOT / "long_test_input" / "video"
DEFAULT_MASK_DIR = ROOT / "long_test_input" / "mask"
DEFAULT_OUTPUT_DIR = ROOT / "long_test_input" / "preview"
PREVIEW_SIZE = (832, 480)
PREVIEW_FPS = 16
PREVIEW_FRAMES = 63 * 4 - 3
PREVIEW_VERSION = 2
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ORANGE = np.array([74.0, 155.0, 255.0], dtype=np.float32)  # BGR: #ff9b4a


def preview_paths(directory: Path, name: str):
    # Keep the original extension in the cache key to distinguish clip.mov
    # from clip.mp4, but always publish an actual H.264 MP4.
    return (
        directory / f"{name}.preview.mp4",
        directory / f"{name}.poster.jpg",
        directory / f"{name}.json",
    )


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Preview encoding requires ffmpeg or pip install imageio-ffmpeg") from exc


def overlay_frame(source_bgr: np.ndarray, mask_bgr: np.ndarray) -> np.ndarray:
    # Match get_video_and_mask: linear resize, then grayscale threshold 240.
    source = cv2.resize(source_bgr, PREVIEW_SIZE)
    mask = cv2.resize(mask_bgr, PREVIEW_SIZE)
    region = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) >= 240
    result = source.copy()
    result[region] = (source[region].astype(np.float32) * 0.38 + ORANGE * 0.62).astype(np.uint8)
    edge = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    result[edge] = (result[edge].astype(np.float32) * 0.18 + ORANGE * 0.82).astype(np.uint8)
    return result


def input_signature(video_path: Path, mask_path: Path) -> dict:
    def stamp(path):
        info = path.stat()
        return [str(path.resolve()), info.st_size, info.st_mtime_ns]
    return {
        "version": PREVIEW_VERSION, "size": list(PREVIEW_SIZE),
        "fps": PREVIEW_FPS, "frames": PREVIEW_FRAMES,
        "source": stamp(video_path), "mask": stamp(mask_path),
    }


def preprocess_pair(video_path: Path, mask_path: Path, output_dir: Path, overwrite=False) -> bool:
    signature = input_signature(video_path, mask_path)
    video_out, poster_out, manifest = preview_paths(output_dir, video_path.name)
    if not overwrite and video_out.is_file() and poster_out.is_file() and manifest.is_file():
        try:
            if json.loads(manifest.read_text(encoding="utf-8")) == signature:
                print(f"[cached] {video_out.name}", flush=True)
                return False
        except (ValueError, OSError):
            pass

    ffmpeg = find_ffmpeg()
    source = cv2.VideoCapture(str(video_path))
    mask = cv2.VideoCapture(str(mask_path))
    encoder = None
    print(f"[prepare] {video_path.name}: orange overlay + poster", flush=True)
    try:
        if not source.isOpened() or not mask.isOpened():
            raise RuntimeError(f"Could not open source/mask pair: {video_path.name}")
        source_fps, mask_fps = source.get(cv2.CAP_PROP_FPS), mask.get(cv2.CAP_PROP_FPS)
        if abs(source_fps - mask_fps) > 0.01:
            raise ValueError(f"Source/mask FPS differ ({source_fps:g}/{mask_fps:g}): align them before preprocessing")
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="preview-", dir=output_dir) as scratch:
            temporary_video = Path(scratch) / "overlay.mp4"
            temporary_poster = Path(scratch) / "poster.jpg"
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "832x480",
                "-r", str(PREVIEW_FPS), "-i", "-", "-an", "-c:v", "libx264",
                "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(temporary_video),
            ]
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
            last_source = last_mask = None
            try:
                for index in range(PREVIEW_FRAMES):
                    source_ok, source_frame = source.read()
                    mask_ok, mask_frame = mask.read()
                    if source_ok:
                        last_source = source_frame
                    if mask_ok:
                        last_mask = mask_frame
                    if last_source is None or last_mask is None:
                        raise ValueError(f"Source/mask contains no readable frames: {video_path.name}")
                    # Same frame indices and last-frame padding as web_demo_long.
                    frame = overlay_frame(last_source, last_mask)
                    if index == 0 and not cv2.imwrite(str(temporary_poster), frame):
                        raise RuntimeError("Could not write the preview poster")
                    encoder.stdin.write(frame.tobytes())
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError("FFmpeg could not encode the H.264 preview")
            finally:
                if encoder.poll() is None:
                    encoder.kill()
                    encoder.wait()
                if not encoder.stdin.closed:
                    encoder.stdin.close()
            temporary_video.replace(video_out)
            temporary_poster.replace(poster_out)
            manifest.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    finally:
        source.release()
        mask.release()
    print(f"[done] {video_out.name}: {PREVIEW_FRAMES} frames at {PREVIEW_FPS} FPS", flush=True)
    return True


def prepare_previews(video_dir=DEFAULT_VIDEO_DIR, mask_dir=DEFAULT_MASK_DIR,
                     output_dir=DEFAULT_OUTPUT_DIR, overwrite=False):
    videos = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise ValueError(f"No videos found in {video_dir}")
    for video_path in videos:
        mask_path = mask_dir / video_path.name
        if not mask_path.is_file():
            print(f"[skip] no matching mask for {video_path.name}")
            continue
        preprocess_pair(video_path, mask_path, output_dir, overwrite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--mask_dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_previews(args.video_dir, args.mask_dir, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
