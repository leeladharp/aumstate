from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from video_agent import VideoSettings


DEFAULT_OUTPUT_WIDTH = 1080
DEFAULT_OUTPUT_HEIGHT = 1920
DEFAULT_OUTPUT_FPS = 30


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def ensure_ffmpeg_available() -> None:
    if not ffmpeg_exists():
        raise RuntimeError(
            "FFmpeg is not installed or not on PATH. Install ffmpeg to render videos, "
            "for example with: sudo apt install -y ffmpeg"
        )


def calculate_frame_count(duration_seconds: float, fps: int = DEFAULT_OUTPUT_FPS) -> int:
    return max(1, round(duration_seconds * fps))


def build_cover_filter(output_width: int, output_height: int) -> str:
    return (
        "scale="
        f"'if(gte(iw/ih,{output_width}/{output_height}),-2,{output_width})':"
        f"'if(gte(iw/ih,{output_width}/{output_height}),{output_height},-2)',"
        f"crop={output_width}:{output_height}:(iw-{output_width})/2:(ih-{output_height})/2"
    )


def build_scene_filter(settings: VideoSettings, duration_seconds: float) -> str:
    del duration_seconds
    return (
        f"{build_cover_filter(settings.output_width, settings.output_height)},"
        f"fps={settings.frame_rate},"
        "setsar=1,"
        "format=yuv420p"
    )


def build_generation_output_dir(base_dir: Path | str = "outputs") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"video_{timestamp}_{uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run_ffmpeg(command: list[str]) -> None:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        error_output = process.stderr.strip() or process.stdout.strip() or "Unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg command failed: {error_output}")


def build_scene_render_command(
    image_path: Path,
    output_path: Path,
    duration_seconds: float,
    settings: VideoSettings,
) -> list[str]:
    frame_count = calculate_frame_count(duration_seconds=duration_seconds, fps=settings.frame_rate)
    filter_chain = build_scene_filter(settings=settings, duration_seconds=duration_seconds)
    return [
        "ffmpeg",
        "-y",
        "-framerate",
        str(settings.frame_rate),
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        str(duration_seconds),
        "-vf",
        filter_chain,
        "-r",
        str(settings.frame_rate),
        "-fps_mode",
        "cfr",
        "-frames:v",
        str(frame_count),
        "-an",
        str(output_path),
    ]


def render_scene_clip(
    image_path: Path,
    output_path: Path,
    duration_seconds: float,
    settings: VideoSettings,
) -> Path:
    ensure_ffmpeg_available()
    run_ffmpeg(
        build_scene_render_command(
            image_path=image_path,
            output_path=output_path,
            duration_seconds=duration_seconds,
            settings=settings,
        )
    )
    return output_path


def write_concat_manifest(clip_paths: list[Path], manifest_path: Path) -> Path:
    lines = [f"file '{clip_path.name}'" for clip_path in clip_paths]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def concat_scene_clips(clip_paths: list[Path], output_path: Path) -> Path:
    ensure_ffmpeg_available()
    manifest_path = output_path.parent / "concat_manifest.txt"
    write_concat_manifest(clip_paths=clip_paths, manifest_path=manifest_path)

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest_path),
        "-c",
        "copy",
        str(output_path),
    ]

    run_ffmpeg(command)
    return output_path


def attach_audio(video_path: Path, audio_path: Path, output_path: Path, settings: VideoSettings) -> Path:
    ensure_ffmpeg_available()
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-af",
        "apad",
        "-t",
        str(settings.total_duration_seconds),
        str(output_path),
    ]
    run_ffmpeg(command)
    return output_path


def render_video(
    image_paths: list[Path],
    scene_durations: list[int],
    output_dir: Path,
    settings: VideoSettings,
    audio_path: Path | None = None,
) -> Path:
    if len(image_paths) != len(scene_durations):
        raise ValueError("Image path count must match scene duration count.")

    clip_paths: list[Path] = []
    for index, (image_path, duration_seconds) in enumerate(zip(image_paths, scene_durations, strict=True), start=1):
        clip_path = output_dir / f"scene_clip_{index:02d}.mp4"
        render_scene_clip(
            image_path=image_path,
            output_path=clip_path,
            duration_seconds=duration_seconds,
            settings=settings,
        )
        clip_paths.append(clip_path)

    silent_video_path = output_dir / "video_no_audio.mp4"
    concat_scene_clips(clip_paths=clip_paths, output_path=silent_video_path)

    if audio_path is not None and audio_path.exists():
        final_output_path = output_dir / "final_video.mp4"
        return attach_audio(
            video_path=silent_video_path,
            audio_path=audio_path,
            output_path=final_output_path,
            settings=settings,
        )

    return silent_video_path
