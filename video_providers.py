from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import subprocess
import textwrap
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from video_agent import VideoPlan, VideoScene, VideoSettings, shorten_narration_once
from video_renderer import ensure_ffmpeg_available, ensure_ffprobe_available, run_subprocess


OPENAI_IMAGE_MODEL = "gpt-image-1"
DEV_MODE_ENV = "VIDEO_DEV_MODE"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_TTS_MODEL_ENV = "OPENAI_TTS_MODEL"
DEFAULT_OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
AUDIBLE_AUDIO_THRESHOLD_DB = -60.0
QUALITY_DRAFT = "Draft"
QUALITY_STANDARD = "Standard"
QUALITY_FINAL = "Final"
DEFAULT_QUALITY_LABEL = QUALITY_DRAFT
QUALITY_LABEL_TO_OPENAI = {
    QUALITY_DRAFT: "low",
    QUALITY_STANDARD: "medium",
    QUALITY_FINAL: "high",
}
ASPECT_RATIO_TO_OPENAI_SIZE = {
    "9:16": "1024x1536",
    "16:9": "1536x1024",
    "1:1": "1024x1024",
}
logger = logging.getLogger(__name__)
_CONFIG_LOADED = False

VOICE_LABEL_TO_OPENAI = {
    "Warm Female": "coral",
    "Warm Male": "ash",
    "Neutral": "alloy",
}
SPEAKING_SPEED_TO_OPENAI = {
    "Slow": 0.9,
    "Normal": 1.0,
    "Fast": 1.1,
}


@dataclass
class NarrationAudioInfo:
    path: Path
    exists: bool
    file_size_bytes: int
    duration_seconds: float
    has_audio_stream: bool
    contains_audible_audio: bool
    codec_name: str
    provider_name: str
    model_name: str
    max_volume_db: float | None = None


@dataclass
class ProviderResult:
    path: Path | None
    used_fallback: bool
    message: str


class ImageProvider(Protocol):
    def generate_scene_image(self, scene: VideoScene, output_path: Path) -> ProviderResult:
        ...


class SpeechProvider(Protocol):
    def generate_narration_audio(self, narration: str, output_path: Path) -> ProviderResult:
        ...

    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def requires_audible_audio(self) -> bool:
        ...


def load_app_config(force: bool = False, dotenv_path: Path | None = None) -> None:
    global _CONFIG_LOADED
    if _CONFIG_LOADED and not force and dotenv_path is None:
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)
    _CONFIG_LOADED = True


load_app_config()


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_scene_image_path(output_dir: Path, scene_number: int) -> Path:
    return output_dir / f"scene_{scene_number:02d}.png"


def build_narration_audio_path(output_dir: Path) -> Path:
    return output_dir / "narration.wav"


def build_placeholder_palette(scene_number: int) -> tuple[str, str]:
    palettes = {
        1: ("#F6E7C0", "#E07A5F"),
        2: ("#D9F0FF", "#3D5A80"),
        3: ("#E7F7D4", "#8D6A9F"),
    }
    return palettes.get(scene_number, ("#F4EBD0", "#4F6D7A"))


def render_placeholder_image(
    scene: VideoScene,
    plan: VideoPlan,
    output_path: Path,
    settings: VideoSettings,
) -> Path:
    ensure_directory(output_path)
    background_color, accent_color = build_placeholder_palette(scene.scene_number)

    image = Image.new("RGB", (settings.output_width, settings.output_height), color=background_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width = settings.output_width
    height = settings.output_height

    draw.rounded_rectangle(
        [(70, 90), (width - 70, height - 90)],
        radius=48,
        outline=accent_color,
        width=10,
    )
    draw.ellipse([(130, 140), (width - 130, int(height * 0.48))], outline=accent_color, width=8)
    draw.rectangle([(140, int(height * 0.57)), (width - 140, int(height * 0.86))], outline=accent_color, width=8)

    text_blocks = [
        f"{plan.title}",
        f"Scene {scene.scene_number}  |  {scene.duration_seconds}s",
        f"Narration: {scene.narration}",
        f"Visual: {scene.visual_prompt}",
        f"Motion: {scene.motion_prompt}",
    ]

    y_position = 190
    for index, block in enumerate(text_blocks):
        wrapped = textwrap.fill(block, width=38)
        fill_color = accent_color if index < 2 else "#1F2933"
        draw.multiline_text(
            (130, y_position),
            wrapped,
            fill=fill_color,
            font=font,
            spacing=8,
        )
        y_position += 210 if index == 0 else 250

    image.save(output_path, format="PNG")
    return output_path


def create_silent_wav(output_path: Path, duration_seconds: float) -> Path:
    ensure_directory(output_path)

    sample_rate = 22050
    frame_count = max(1, int(sample_rate * duration_seconds))

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)

    return output_path


def is_video_dev_mode() -> bool:
    return get_config_value(DEV_MODE_ENV, "false").strip().lower() == "true"


def get_streamlit_secret(key: str) -> str:
    try:
        import streamlit as st
    except Exception:
        return ""

    try:
        value = st.secrets.get(key, "")
    except Exception:
        return ""

    return str(value).strip()


def get_config_value(key: str, default: str = "") -> str:
    value = os.getenv(key, "").strip()
    if value:
        return value

    secret_value = get_streamlit_secret(key)
    if secret_value:
        return secret_value

    return default


def get_openai_image_size(settings: VideoSettings) -> str:
    return ASPECT_RATIO_TO_OPENAI_SIZE.get(settings.aspect_ratio, "1024x1536")


def build_openai_image_prompt(plan: VideoPlan, scene: VideoScene) -> str:
    continuity_lines = [
        "Shared continuity instructions:",
        "- Preserve the requested style_lock without adding extra aesthetics that were not requested.",
        "- No text, logos, watermarks, captions, extra limbs, duplicate characters, or distorted anatomy.",
        f"- Respect the selected aspect ratio: {plan.aspect_ratio}.",
    ]
    if scene.continuity_mode == "independent":
        continuity_lines.extend(
            [
                "- This scene is independent. Do not inherit subjects or symbols from earlier scenes unless they are explicitly in this scene prompt.",
                "- Generate a fresh image for this scene without forcing continuity from earlier scenes.",
            ]
        )
    else:
        continuity_lines.extend(
            [
                f"- This scene uses continuity_mode={scene.continuity_mode} and continuity_group={scene.continuity_group}.",
                "- Preserve exact recurring character or world identity for this continuity group only.",
                "- If a reference image is provided, treat that group's approved reference image as authoritative.",
                "- Preserve face, hair, clothing, body proportions, accessories, environment continuity, and lighting continuity for this continuity group.",
                "- Preserve the exact same hairstyle, hair length, hairline, hair volume, and hair color unless the user explicitly requested a change.",
            ]
        )
    return (
        "\n".join(continuity_lines)
        + "\n\n"
        f"Style lock:\n{plan.style_lock}\n\n"
        f"Scene instructions:\n{scene.visual_prompt}\n\n"
        "Scene metadata:\n"
        f"- Scene number: {scene.scene_number}\n"
        f"- Duration: {scene.duration_seconds} seconds\n"
        f"- Continuity mode: {scene.continuity_mode}\n"
        f"- Continuity group: {scene.continuity_group or 'none'}\n"
        f"- Motion intent for later animation: {scene.motion_prompt}\n\n"
        "Image requirements:\n"
        "- Use the closest supported OpenAI image size for the selected aspect ratio.\n"
        "- Save the final image as PNG.\n"
        "- Do not stretch the generated image.\n"
        "- If a reference image is provided, keep the same hair and face identity as the reference image.\n"
        "- If the API returns a different supported size, preserve aspect ratio and let the existing renderer handle final video sizing."
    )


def get_openai_api_key() -> str:
    return get_config_value(OPENAI_API_KEY_ENV)


def get_openai_tts_model() -> str:
    return get_config_value(OPENAI_TTS_MODEL_ENV, DEFAULT_OPENAI_TTS_MODEL)


def build_openai_client() -> Any:
    from openai import OpenAI

    return OpenAI(api_key=get_openai_api_key())


def normalize_quality_label(quality_label: str) -> str:
    normalized = (quality_label or "").strip()
    if normalized in QUALITY_LABEL_TO_OPENAI:
        return normalized
    return DEFAULT_QUALITY_LABEL


def map_quality_label_to_openai(quality_label: str) -> str:
    return QUALITY_LABEL_TO_OPENAI[normalize_quality_label(quality_label)]


def get_provider_name(provider: ImageProvider) -> str:
    return provider.__class__.__name__


def select_image_provider(
    plan: VideoPlan,
    settings: VideoSettings,
    image_provider: ImageProvider | None = None,
) -> ImageProvider:
    if image_provider is not None:
        return image_provider
    if is_video_dev_mode():
        return PlaceholderImageProvider(plan=plan, settings=settings)
    return OpenAIImageProvider(plan=plan, settings=settings)


class PlaceholderImageProvider:
    def __init__(self, plan: VideoPlan | None = None, settings: VideoSettings | None = None) -> None:
        self.plan = plan
        self.settings = settings

    def generate_scene_image(self, scene: VideoScene, output_path: Path) -> ProviderResult:
        placeholder_plan = self.plan or VideoPlan(
            title="AUM State Video Studio",
            content_type="placeholder",
            duration_seconds=15,
            aspect_ratio="9:16",
            narration="",
            style_lock="",
            scenes=[],
            settings=self.settings or VideoSettings(15, 5, [5, 5, 5], 3, "basic_motion", 30, "9:16", 1080, 1920, "nursery", "3D Nursery Animation", "Draft", "Still", True, "English", "Warm Female", "Warm", "Normal"),
        )
        settings = self.settings or placeholder_plan.settings
        render_placeholder_image(scene=scene, plan=placeholder_plan, output_path=output_path, settings=settings)
        return ProviderResult(
            path=output_path,
            used_fallback=True,
            message="Generated placeholder scene image with Pillow.",
        )


class OpenAIImageProvider:
    def __init__(self, plan: VideoPlan, settings: VideoSettings, client: Any | None = None) -> None:
        self.plan = plan
        self.settings = settings
        self.api_key = get_openai_api_key()
        self.quality_label = normalize_quality_label(settings.image_quality)
        self.openai_quality = map_quality_label_to_openai(self.quality_label)
        self.client = client
        self.continuity_references: dict[str, Path] = {}

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def update_continuity_reference(self, scene: VideoScene, image_path: Path) -> None:
        if scene.continuity_group and scene.continuity_mode != "independent":
            self.continuity_references[scene.continuity_group] = image_path

    def _generate_from_reference(
        self,
        client: Any,
        scene: VideoScene,
        prompt: str,
    ) -> Any:
        reference_path = None
        if scene.continuity_group and scene.continuity_mode != "independent":
            reference_path = self.continuity_references.get(scene.continuity_group)

        if reference_path is None:
            return client.images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=prompt,
                size=get_openai_image_size(self.settings),
                quality=self.openai_quality,
            )

        logger.info(
            "OpenAI image edit provider=%s model=%s size=%s quality=%s scene_number=%s reference_image=%s",
            self.__class__.__name__,
            OPENAI_IMAGE_MODEL,
            get_openai_image_size(self.settings),
            self.openai_quality,
            scene.scene_number,
            reference_path.name,
        )
        with reference_path.open("rb") as reference_image:
            return client.images.edit(
                model=OPENAI_IMAGE_MODEL,
                image=reference_image,
                prompt=prompt,
                size=get_openai_image_size(self.settings),
                quality=self.openai_quality,
                input_fidelity="high",
            )

    def generate_scene_image(self, scene: VideoScene, output_path: Path) -> ProviderResult:
        if not self.is_configured():
            raise ValueError(
                "OPENAI_API_KEY is missing. Set OPENAI_API_KEY to enable real image generation, "
                f"or set {DEV_MODE_ENV}=true to use placeholder images during development."
            )

        prompt = build_openai_image_prompt(plan=self.plan, scene=scene)
        client = self.client or build_openai_client()
        logger.info(
            "OpenAI image request provider=%s model=%s size=%s quality=%s scene_number=%s",
            self.__class__.__name__,
            OPENAI_IMAGE_MODEL,
            get_openai_image_size(self.settings),
            self.openai_quality,
            scene.scene_number,
        )
        response = self._generate_from_reference(client=client, scene=scene, prompt=prompt)

        image_bytes = extract_openai_image_bytes(response)
        if not image_bytes:
            raise RuntimeError(f"Scene {scene.scene_number} returned an empty image response.")

        save_png_bytes(image_bytes=image_bytes, output_path=output_path)
        self.update_continuity_reference(scene=scene, image_path=output_path)
        return ProviderResult(
            path=output_path,
            used_fallback=False,
            message=(
                f"Generated scene {scene.scene_number} image with the OpenAI Images API "
                f"using {self.quality_label} quality."
            ),
        )


def extract_openai_image_bytes(response: Any) -> bytes:
    data = getattr(response, "data", None)
    if not data:
        raise RuntimeError("OpenAI Images API returned no image data.")

    first_item = data[0]
    b64_json = getattr(first_item, "b64_json", None)
    if not b64_json and isinstance(first_item, dict):
        b64_json = first_item.get("b64_json")

    if not b64_json:
        raise RuntimeError("OpenAI Images API response did not include base64 image content.")

    return base64.b64decode(b64_json)


def save_png_bytes(image_bytes: bytes, output_path: Path) -> Path:
    ensure_directory(output_path)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.save(output_path, format="PNG")
    return output_path


class SilentSpeechProvider:
    provider_name = "Development fallback"
    model_name = "silent_wav"
    requires_audible_audio = False

    def generate_narration_audio(self, narration: str, output_path: Path) -> ProviderResult:
        word_count = max(1, len(narration.split()))
        duration_seconds = max(15.0, word_count / 2.5)
        create_silent_wav(output_path=output_path, duration_seconds=duration_seconds)
        return ProviderResult(
            path=output_path,
            used_fallback=True,
            message="Created silent narration fallback audio.",
        )


class OpenAISpeechProvider:
    provider_name = "OpenAI"
    requires_audible_audio = True

    def __init__(self, settings: VideoSettings, client: Any | None = None) -> None:
        self.settings = settings
        self.api_key = get_openai_api_key()
        self.model_name = get_openai_tts_model()
        self.client = client

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def build_voice(self) -> str:
        return VOICE_LABEL_TO_OPENAI.get(self.settings.voice, "alloy")

    def build_instructions(self) -> str:
        return (
            f"Language: {self.settings.language}. "
            f"Speaking style: {self.settings.speaking_style}. "
            f"Voice tone: {self.settings.voice}. "
            "Deliver clear, natural narration with stable pacing."
        )

    def build_speed(self) -> float:
        return SPEAKING_SPEED_TO_OPENAI.get(self.settings.speaking_speed, 1.0)

    def generate_narration_audio(self, narration: str, output_path: Path) -> ProviderResult:
        if not self.is_configured():
            raise ValueError("OPENAI_API_KEY is missing. Configure it before generating real narration.")

        client = self.client or build_openai_client()
        response = client.audio.speech.create(
            input=narration,
            model=self.model_name,
            voice=self.build_voice(),
            instructions=self.build_instructions(),
            response_format="wav",
            speed=self.build_speed(),
        )

        ensure_directory(output_path)
        response.write_to_file(output_path)
        return ProviderResult(
            path=output_path,
            used_fallback=False,
            message=f"Generated narration audio with OpenAI using model {self.model_name}.",
        )


def select_speech_provider(
    settings: VideoSettings,
    speech_provider: SpeechProvider | None = None,
) -> SpeechProvider:
    if speech_provider is not None:
        return speech_provider
    if is_video_dev_mode():
        return SilentSpeechProvider()
    return OpenAISpeechProvider(settings=settings)


def probe_audio_file(audio_path: Path) -> dict[str, Any]:
    ensure_ffprobe_available()
    result = run_subprocess(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(audio_path),
        ]
    )
    return json.loads(result.stdout)


def get_audio_max_volume_db(audio_path: Path) -> float | None:
    ensure_ffmpeg_available()
    process = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = process.stderr
    match = re.search(r"max_volume:\s*([-\w\.]+)\s*dB", output)
    if match is None:
        raise RuntimeError("Could not determine narration audio loudness.")
    raw_value = match.group(1).lower()
    if raw_value == "-inf":
        return None
    return float(raw_value)


def audio_contains_audible_signal(audio_path: Path) -> bool:
    max_volume_db = get_audio_max_volume_db(audio_path)
    if max_volume_db is None:
        return False
    return max_volume_db > AUDIBLE_AUDIO_THRESHOLD_DB


def inspect_narration_audio_file(
    audio_path: Path,
    provider_name: str = "",
    model_name: str = "",
    require_audible_audio: bool = True,
) -> NarrationAudioInfo:
    if not audio_path.exists():
        raise FileNotFoundError(f"Narration file was not created: {audio_path}")

    file_size_bytes = audio_path.stat().st_size
    if file_size_bytes <= 0:
        raise ValueError(f"Narration file is empty: {audio_path}")

    payload = probe_audio_file(audio_path)
    streams = payload.get("streams", [])
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if audio_stream is None:
        raise ValueError(f"Narration file does not contain a readable audio stream: {audio_path}")

    duration_value = payload.get("format", {}).get("duration", 0.0) or 0.0
    duration_seconds = float(duration_value)
    if duration_seconds <= 0:
        raise ValueError(f"Narration file duration is invalid: {audio_path}")

    max_volume_db = get_audio_max_volume_db(audio_path)
    contains_audible_audio = max_volume_db is not None and max_volume_db > AUDIBLE_AUDIO_THRESHOLD_DB
    if require_audible_audio and not contains_audible_audio:
        raise ValueError("Narration was generated, but the audio is silent or invalid. Please retry narration generation.")

    return NarrationAudioInfo(
        path=audio_path,
        exists=True,
        file_size_bytes=file_size_bytes,
        duration_seconds=duration_seconds,
        has_audio_stream=True,
        contains_audible_audio=contains_audible_audio,
        codec_name=str(audio_stream.get("codec_name", "") or ""),
        provider_name=provider_name,
        model_name=model_name,
        max_volume_db=max_volume_db,
    )


def generate_scene_images(
    plan: VideoPlan,
    output_dir: Path,
    settings: VideoSettings,
    image_provider: ImageProvider | None = None,
) -> tuple[list[Path], list[str]]:
    selected_quality = normalize_quality_label(settings.image_quality)
    provider = select_image_provider(plan=plan, settings=settings, image_provider=image_provider)
    image_paths: list[Path] = []
    messages: list[str] = [
        f"Image provider: {get_provider_name(provider)}",
        f"Image quality: {selected_quality}",
    ]
    failures: list[str] = []

    for scene in plan.scenes:
        output_path = build_scene_image_path(output_dir=output_dir, scene_number=scene.scene_number)
        try:
            result = provider.generate_scene_image(scene=scene, output_path=output_path)
        except Exception as error:
            failures.append(f"Scene {scene.scene_number} failed: {error}")
            messages.append(f"Scene {scene.scene_number} image generation failed: {error}")
            continue

        if result.path is None:
            failures.append(f"Scene {scene.scene_number} failed: provider returned no file path.")
            messages.append(f"Scene {scene.scene_number} image generation failed: provider returned no file path.")
            continue

        image_paths.append(result.path)
        messages.append(result.message)

    if failures:
        raise RuntimeError("One or more scene images failed to generate. " + " | ".join(failures))

    return image_paths, messages


def measure_wav_duration(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        if frame_rate <= 0:
            raise ValueError("Audio frame rate must be positive.")
        return frame_count / frame_rate


def generate_narration_audio(
    plan: VideoPlan,
    output_dir: Path,
    settings: VideoSettings,
    speech_provider: SpeechProvider | None = None,
) -> tuple[Path | None, str]:
    if not settings.narration_enabled:
        return None, "Narration disabled."

    provider = select_speech_provider(settings=settings, speech_provider=speech_provider)
    output_path = build_narration_audio_path(output_dir=output_dir)
    available_duration_seconds = max(0.5, settings.total_duration_seconds - 0.5)

    result = provider.generate_narration_audio(plan.narration, output_path)

    try:
        narration_info = inspect_narration_audio_file(
            audio_path=result.path,
            provider_name=provider.provider_name,
            model_name=provider.model_name,
            require_audible_audio=provider.requires_audible_audio,
        )
    except ValueError as error:
        if provider.requires_audible_audio:
            raise ValueError("Narration was generated, but the audio is silent or invalid. Please retry narration generation.") from error
        raise

    narration_duration = narration_info.duration_seconds

    if narration_duration <= available_duration_seconds:
        return result.path, result.message

    shortened_narration = shorten_narration_once(
        narration=plan.narration,
        settings=settings,
        available_duration_seconds=available_duration_seconds,
    )

    retry_path = output_dir / "narration_retry.wav"
    retry_result = provider.generate_narration_audio(shortened_narration, retry_path)
    try:
        retry_info = inspect_narration_audio_file(
            audio_path=retry_result.path,
            provider_name=provider.provider_name,
            model_name=provider.model_name,
            require_audible_audio=provider.requires_audible_audio,
        )
    except ValueError as error:
        if provider.requires_audible_audio:
            raise ValueError("Narration was generated, but the audio is silent or invalid. Please retry narration generation.") from error
        raise
    retry_duration = retry_info.duration_seconds
    if retry_duration > available_duration_seconds:
        return retry_result.path, (
            f"{retry_result.message} Narration still exceeds the available duration after one retry."
        )

    return retry_result.path, (
        f"{retry_result.message} Narration was shortened once to fit the selected duration."
    )
