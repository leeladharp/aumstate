from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from video_agent import VideoPlan, VideoScene, VideoSettings
from video_providers import build_narration_audio_path, build_scene_image_path, inspect_narration_audio_file
from video_renderer import (
    attach_audio,
    concat_scene_clips,
    ensure_ffmpeg_available,
    ensure_ffprobe_available,
    run_ffmpeg,
    run_subprocess,
)


KLING_ASSISTED_MODE = "kling_assisted"
BASIC_MOTION_MODE = "basic_motion"
KLING_DURATION_TOLERANCE_SECONDS = 0.15
KLING_IMPORT_STATE_FILENAME = "kling_import_state.json"
KLING_IMPORTS_DIRNAME = "kling_imports"
KLING_NORMALIZED_DIRNAME = "kling_normalized"
KLING_PACKAGE_NAME_PREFIX = "kling_project_"
SIMPLE_LIFE_STYLES = {"Simple Life Story", "Quiet Cinematic Animation"}
STATUS_NOT_UPLOADED = "Not uploaded"
STATUS_UPLOADED = "Uploaded"
STATUS_VALIDATING = "Validating"
STATUS_VALID = "Valid"
STATUS_INVALID = "Invalid"
STATUS_READY = "Ready"
ACCEPTED_VIDEO_EXTENSIONS = {".mp4"}


@dataclass
class KlingSceneAsset:
    scene_number: int
    image_path: str
    prompt_path: str
    details_path: str
    expected_filename: str
    required_duration: float


@dataclass
class KlingImportEntry:
    generation_id: str
    scene_number: int
    source_filename: str
    stored_source_path: str
    assigned_scene: int | None
    source_duration: float | None
    required_duration: float
    trim_start: float
    validation_status: str
    normalized_output_path: str
    error_summary: str
    updated_timestamp: str
    source_width: int | None = None
    source_height: int | None = None
    source_frame_rate: float | None = None
    source_rotation: int | None = None
    source_pixel_format: str | None = None
    has_audio_stream: bool | None = None
    source_file_size: int | None = None
    source_mtime: float | None = None
    trim_applied: bool = False


def build_kling_prompt(scene: VideoScene, settings: VideoSettings) -> str:
    lines = [
        "Scene objective:",
        scene.visual_prompt,
        "",
        "Subject movement:",
        scene.motion_prompt,
        "",
        "Natural motion:",
        "- subtle facial expressions",
        "- restrained head and eye movement",
        "- natural body mechanics",
        "- gentle cloth and hair movement",
        "- soft environmental motion where appropriate",
        "- physically plausible movement",
        "- calm pacing",
        "",
        "Camera:",
        "- stable camera",
        "- gentle cinematic movement only",
        "- no handheld shake",
        "- no rapid zoom",
        "- no sudden pan",
        "- no extreme reframing",
        "- preserve the original composition",
        "",
        "Continuity:",
        "- use the uploaded image as the visual source of truth",
        "- preserve character identity",
        "- preserve face, clothing, proportions, and age",
        "- preserve background layout",
        "- preserve lighting and color palette",
        "- do not introduce new characters or major objects",
        "",
        "Negative constraints:",
        "- no jitter",
        "- no flicker",
        "- no morphing",
        "- no warped face",
        "- no duplicated limbs",
        "- no extra fingers",
        "- no unstable background",
        "- no sudden lighting changes",
        "- no text",
        "- no logos",
        "- no watermark",
        "",
        f"Visual style: {settings.visual_style}",
        f"Content type: {settings.content_type}",
        f"Aspect ratio: {settings.aspect_ratio}",
        f"Target visible action duration: approximately {scene.duration_seconds} seconds.",
        "Use the uploaded image as the visual source of truth and preserve continuity exactly.",
    ]

    if settings.visual_style in SIMPLE_LIFE_STYLES:
        lines.extend(
            [
                "",
                "Simple-life style rules:",
                "- ordinary everyday actions",
                "- subtle emotion",
                "- muted natural colors",
                "- soft daylight",
                "- restrained character movement",
                "- gentle environmental movement",
                "- stable observational camera",
                "- realistic pacing",
                "- no exaggerated cartoon motion",
                "- no neon nursery styling",
                "- no magical sparkles",
                "- no dramatic camera circles",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def expected_kling_filename(scene_number: int) -> str:
    return f"scene_{scene_number:02d}_kling.mp4"


def build_scene_details(scene: VideoScene, settings: VideoSettings) -> dict[str, Any]:
    return {
        "scene_number": scene.scene_number,
        "required_duration_seconds": scene.duration_seconds,
        "aspect_ratio": settings.aspect_ratio,
        "output_width": settings.output_width,
        "output_height": settings.output_height,
        "frame_rate": settings.frame_rate,
        "narration": scene.narration,
        "motion_prompt": scene.motion_prompt,
        "expected_imported_filename": expected_kling_filename(scene.scene_number),
    }


def build_kling_manifest(plan: VideoPlan, settings: VideoSettings, generation_id: str) -> dict[str, Any]:
    return {
        "generation_id": generation_id,
        "title": plan.title,
        "video_mode": settings.video_mode,
        "total_duration_seconds": settings.total_duration_seconds,
        "scene_count": settings.scene_count,
        "aspect_ratio": settings.aspect_ratio,
        "output_width": settings.output_width,
        "output_height": settings.output_height,
        "frame_rate": settings.frame_rate,
        "narration_enabled": settings.narration_enabled,
        "language": settings.language,
        "visual_style": settings.visual_style,
        "content_type": settings.content_type,
        "scene_order": [scene.scene_number for scene in plan.scenes],
        "scenes": [
            {
                "scene_number": scene.scene_number,
                "expected_imported_filename": expected_kling_filename(scene.scene_number),
                "required_duration_seconds": scene.duration_seconds,
            }
            for scene in plan.scenes
        ],
    }


def build_kling_instructions() -> str:
    return "\n".join(
        [
            "1. Open the Kling website.",
            "2. Select image-to-video.",
            "3. Upload scene_01.png.",
            "4. Paste scene_01_kling_prompt.txt.",
            "5. Select an appropriate generation mode.",
            "6. Generate the clip.",
            "7. Download it.",
            "8. Rename it to scene_01_kling.mp4.",
            "9. Repeat for every scene.",
            "10. Return to AumState and upload all clips.",
            "11. Keep scene numbers unchanged.",
            "",
            "Recommendations:",
            "- generate and inspect Scene 1 before generating all scenes",
            "- avoid adding automatic Kling text or subtitles",
            "- avoid extending the clip unnecessarily",
            "- preserve the uploaded image composition",
            "- use the same Kling model/settings across all scenes where possible",
            "- do not change character appearance between scenes",
        ]
    ) + "\n"


def build_kling_package(
    plan: VideoPlan,
    settings: VideoSettings,
    output_dir: Path,
    generation_id: str,
) -> Path:
    if settings.video_mode != KLING_ASSISTED_MODE:
        raise ValueError("Kling package export requires kling_assisted mode.")

    for scene in plan.scenes:
        image_path = build_scene_image_path(output_dir=output_dir, scene_number=scene.scene_number)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing scene image for scene {scene.scene_number}: {image_path}")

    narration_path = build_narration_audio_path(output_dir=output_dir)
    zip_path = output_dir / f"{KLING_PACKAGE_NAME_PREFIX}{generation_id}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "project_manifest.json",
            json.dumps(build_kling_manifest(plan=plan, settings=settings, generation_id=generation_id), indent=2),
        )
        archive.writestr("kling_instructions.txt", build_kling_instructions())

        narration_dir = "narration"
        archive.writestr(f"{narration_dir}/narration.txt", plan.narration)
        if settings.narration_enabled:
            if not narration_path.exists():
                raise FileNotFoundError(f"Missing narration file: {narration_path}")
            archive.write(narration_path, arcname=f"{narration_dir}/{narration_path.name}")

        for scene in plan.scenes:
            scene_dir = f"scene_{scene.scene_number:02d}"
            image_path = build_scene_image_path(output_dir=output_dir, scene_number=scene.scene_number)
            prompt_text = build_kling_prompt(scene=scene, settings=settings)
            details = build_scene_details(scene=scene, settings=settings)
            archive.write(image_path, arcname=f"{scene_dir}/{image_path.name}")
            archive.writestr(f"{scene_dir}/scene_{scene.scene_number:02d}_kling_prompt.txt", prompt_text)
            archive.writestr(f"{scene_dir}/scene_{scene.scene_number:02d}_details.json", json.dumps(details, indent=2))

    return zip_path


def normalize_filename_for_mapping(filename: str) -> str:
    return re.sub(r"[^a-z0-9]", "", Path(filename).name.lower())


def match_scene_number_from_filename(filename: str) -> int | None:
    normalized = normalize_filename_for_mapping(filename)
    patterns = [
        r"scene0*([1-9]\d*)klingmp4$",
        r"scene0*([1-9]\d*)mp4$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))
    return None


def build_imports_dir(output_dir: Path) -> Path:
    path = output_dir / KLING_IMPORTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_normalized_dir(output_dir: Path) -> Path:
    path = output_dir / KLING_NORMALIZED_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_import_state_path(output_dir: Path) -> Path:
    return output_dir / KLING_IMPORT_STATE_FILENAME


def load_import_state(output_dir: Path) -> dict[int, KlingImportEntry]:
    state_path = build_import_state_path(output_dir=output_dir)
    if not state_path.exists():
        return {}
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    entries: dict[int, KlingImportEntry] = {}
    for item in payload.get("entries", []):
        entry = KlingImportEntry(**item)
        entries[entry.scene_number] = entry
    return entries


def save_import_state(output_dir: Path, generation_id: str, entries: dict[int, KlingImportEntry]) -> Path:
    state_path = build_import_state_path(output_dir=output_dir)
    payload = {
        "generation_id": generation_id,
        "entries": [asdict(entries[key]) for key in sorted(entries)],
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return state_path


def store_uploaded_clip(
    output_dir: Path,
    scene_number: int,
    source_filename: str,
    file_bytes: bytes,
) -> Path:
    imports_dir = build_imports_dir(output_dir=output_dir)
    target_path = imports_dir / f"scene_{scene_number:02d}_original.mp4"
    if target_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = imports_dir / f"scene_{scene_number:02d}_original_{timestamp}.mp4"
        shutil.move(str(target_path), str(archive_path))
    target_path.write_bytes(file_bytes)
    return target_path


def probe_media(path: Path) -> dict[str, Any]:
    ensure_ffprobe_available()
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    result = run_subprocess(command)
    return json.loads(result.stdout)


def parse_frame_rate(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        if float(denominator) == 0:
            return None
        return float(numerator) / float(denominator)
    return float(rate)


def parse_rotation_value(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    try:
        normalized = int(round(float(raw_value))) % 360
    except (TypeError, ValueError):
        return None

    normalized_map = {
        0: 0,
        90: 90,
        180: 180,
        270: 270,
    }
    return normalized_map.get(normalized, normalized)


def extract_rotation(stream: dict[str, Any]) -> int:
    tags = stream.get("tags", {}) or {}
    side_data_list = stream.get("side_data_list", []) or []

    for item in side_data_list:
        rotation = parse_rotation_value(item.get("rotation"))
        if rotation is not None:
            return rotation

    rotation = parse_rotation_value(tags.get("rotate"))
    if rotation is not None:
        return rotation

    return 0


def validate_clip(
    clip_path: Path,
    required_duration: float,
    trim_start: float = 0.0,
    tolerance_seconds: float = KLING_DURATION_TOLERANCE_SECONDS,
) -> tuple[str, dict[str, Any], str]:
    if clip_path.suffix.lower() not in ACCEPTED_VIDEO_EXTENSIONS:
        return STATUS_INVALID, {}, "Only MP4 clips are accepted."

    try:
        payload = probe_media(path=clip_path)
    except Exception as error:
        return STATUS_INVALID, {}, f"Corrupted or unreadable MP4: {error}"

    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video_stream is None:
        return STATUS_INVALID, {}, "Imported clip has no readable video stream."

    source_duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    width = int(video_stream.get("width", 0) or 0)
    height = int(video_stream.get("height", 0) or 0)
    frame_rate = parse_frame_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    rotation = extract_rotation(video_stream)
    pixel_format = str(video_stream.get("pix_fmt", "") or "")
    sample_aspect_ratio = str(video_stream.get("sample_aspect_ratio", "") or "")
    details = {
        "source_duration": source_duration,
        "width": width,
        "height": height,
        "frame_rate": frame_rate,
        "rotation": rotation,
        "pixel_format": pixel_format,
        "sample_aspect_ratio": sample_aspect_ratio,
        "has_audio_stream": audio_stream is not None,
    }

    if source_duration <= 0:
        return STATUS_INVALID, details, "Imported clip duration could not be measured."
    if source_duration + tolerance_seconds < required_duration:
        return STATUS_INVALID, details, (
            f"Clip is shorter than required. Required {required_duration:.2f}s, got {source_duration:.2f}s. "
            f"Tolerance: {tolerance_seconds:.2f}s."
        )
    if trim_start < 0:
        return STATUS_INVALID, details, "Trim start must be greater than or equal to 0."
    if trim_start + required_duration > source_duration + tolerance_seconds:
        return STATUS_INVALID, details, (
            f"Trim start exceeds clip bounds. Required {required_duration:.2f}s from {trim_start:.2f}s, "
            f"source {source_duration:.2f}s, tolerance {tolerance_seconds:.2f}s."
        )

    status = STATUS_VALID
    if trim_start > 0 or source_duration > required_duration + tolerance_seconds:
        status = STATUS_READY
    return status, details, ""


def build_normalize_command(
    source_path: Path,
    output_path: Path,
    settings: VideoSettings,
    required_duration: float,
    trim_start: float,
) -> list[str]:
    vf_parts = [
        "scale="
        f"'if(gte(iw/ih,{settings.output_width}/{settings.output_height}),-2,{settings.output_width})':"
        f"'if(gte(iw/ih,{settings.output_width}/{settings.output_height}),{settings.output_height},-2)'",
        f"crop={settings.output_width}:{settings.output_height}:(iw-{settings.output_width})/2:(ih-{settings.output_height})/2",
        f"fps={settings.frame_rate}",
        "setsar=1",
        "format=yuv420p",
    ]
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{trim_start:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{required_duration:.3f}",
        "-an",
        "-vf",
        ",".join(vf_parts),
        "-r",
        str(settings.frame_rate),
        "-fps_mode",
        "cfr",
        str(output_path),
    ]


def normalize_clip(
    source_path: Path,
    output_dir: Path,
    scene_number: int,
    settings: VideoSettings,
    required_duration: float,
    trim_start: float,
) -> Path:
    ensure_ffmpeg_available()
    normalized_dir = build_normalized_dir(output_dir=output_dir)
    output_path = normalized_dir / f"scene_{scene_number:02d}_normalized.mp4"
    run_ffmpeg(
        build_normalize_command(
            source_path=source_path,
            output_path=output_path,
            settings=settings,
            required_duration=required_duration,
            trim_start=trim_start,
        )
    )
    if not output_path.exists() or not output_path.is_file():
        raise FileNotFoundError(f"Normalized clip was not created for scene {scene_number}: {output_path}")
    return output_path


def needs_renormalization(entry: KlingImportEntry, source_path: Path) -> bool:
    if not entry.normalized_output_path:
        return True
    normalized_path = Path(entry.normalized_output_path)
    if not normalized_path.exists():
        return True
    stat = source_path.stat()
    return (
        entry.source_file_size != stat.st_size
        or entry.source_mtime != stat.st_mtime
    )


def update_entry_from_validation(
    entry: KlingImportEntry,
    source_path: Path,
    status: str,
    details: dict[str, Any],
    error_summary: str,
) -> KlingImportEntry:
    stat = source_path.stat()
    entry.source_duration = details.get("source_duration")
    entry.validation_status = status
    entry.error_summary = error_summary
    entry.source_width = details.get("width")
    entry.source_height = details.get("height")
    entry.source_frame_rate = details.get("frame_rate")
    entry.source_rotation = details.get("rotation")
    entry.source_pixel_format = details.get("pixel_format")
    entry.has_audio_stream = details.get("has_audio_stream")
    entry.source_file_size = stat.st_size
    entry.source_mtime = stat.st_mtime
    entry.updated_timestamp = datetime.now().isoformat()
    return entry


def create_import_entry(
    generation_id: str,
    scene_number: int,
    source_filename: str,
    stored_source_path: Path,
    required_duration: float,
    trim_start: float = 0.0,
) -> KlingImportEntry:
    return KlingImportEntry(
        generation_id=generation_id,
        scene_number=scene_number,
        source_filename=source_filename,
        stored_source_path=str(stored_source_path),
        assigned_scene=scene_number,
        source_duration=None,
        required_duration=required_duration,
        trim_start=trim_start,
        validation_status=STATUS_UPLOADED,
        normalized_output_path="",
        error_summary="",
        updated_timestamp=datetime.now().isoformat(),
    )


def resolve_narration_path(output_dir: Path, stored_narration_path: Path | None) -> Path | None:
    if stored_narration_path is not None and stored_narration_path.exists():
        return stored_narration_path

    fallback_path = build_narration_audio_path(output_dir=output_dir)
    if fallback_path.exists():
        return fallback_path

    return stored_narration_path


def assemble_kling_video(
    plan: VideoPlan,
    settings: VideoSettings,
    output_dir: Path,
    import_entries: dict[int, KlingImportEntry],
    narration_path: Path | None,
) -> Path:
    missing_scenes = [
        scene.scene_number
        for scene in plan.scenes
        if scene.scene_number not in import_entries or import_entries[scene.scene_number].validation_status not in {STATUS_VALID, STATUS_READY}
    ]
    if missing_scenes:
        raise ValueError(f"Cannot assemble final video. Missing valid clips for scenes: {missing_scenes}")

    normalized_paths: list[Path] = []
    for scene in plan.scenes:
        entry = import_entries[scene.scene_number]
        normalized_path = Path(entry.normalized_output_path)
        if not normalized_path.exists():
            raise FileNotFoundError(f"Missing normalized clip for scene {scene.scene_number}: {normalized_path}")
        normalized_paths.append(normalized_path)

    silent_output = output_dir / "final_kling_assisted_silent.mp4"
    concat_scene_clips(clip_paths=normalized_paths, output_path=silent_output)

    if settings.narration_enabled:
        if narration_path is None or not narration_path.exists():
            raise FileNotFoundError(
                "Narration is enabled, but the generated narration file could not be found. "
                "Regenerate narration before assembling."
            )
        inspect_narration_audio_file(
            audio_path=narration_path,
            provider_name="",
            model_name="",
            require_audible_audio=True,
        )
        final_output = output_dir / "final_kling_assisted.mp4"
        return attach_audio(
            video_path=silent_output,
            audio_path=narration_path,
            output_path=final_output,
            settings=settings,
        )

    return silent_output
