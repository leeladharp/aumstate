from __future__ import annotations

import base64
import io
import json
import logging
import os
import textwrap
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw, ImageFont

from video_agent import VideoPlan, VideoScene, VideoSettings, shorten_narration_once


OPENAI_IMAGE_MODEL = "gpt-image-1"
DEV_MODE_ENV = "VIDEO_DEV_MODE"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
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
    return os.getenv(DEV_MODE_ENV, "").strip().lower() == "true"


def get_openai_image_size(settings: VideoSettings) -> str:
    return ASPECT_RATIO_TO_OPENAI_SIZE.get(settings.aspect_ratio, "1024x1536")


def build_openai_image_prompt(plan: VideoPlan, scene: VideoScene) -> str:
    return (
        "Shared continuity instructions:\n"
        "- Preserve the exact same main character design across all scenes.\n"
        "- Treat the reference character design as authoritative for every later scene.\n"
        "- Preserve clothing, face, proportions, colors, environment style, and lighting.\n"
        "- Preserve the exact same hairstyle, hair length, hairline, hair volume, and hair color across all scenes.\n"
        "- Do not restyle, lengthen, shorten, braid, curl, straighten, recolor, or otherwise alter the hair unless the user explicitly requested that transformation.\n"
        "- Create a polished 3D nursery animation aesthetic.\n"
        "- Use large expressive eyes, soft rounded shapes, bright pastel colors, warm cinematic lighting, and toddler-friendly composition.\n"
        "- No text, logos, watermarks, captions, extra limbs, duplicate characters, or distorted anatomy.\n"
        "- Vertical composition suitable for a 9:16 short-form video.\n\n"
        f"Style lock:\n{plan.style_lock}\n\n"
        f"Scene instructions:\n{scene.visual_prompt}\n\n"
        "Scene metadata:\n"
        f"- Scene number: {scene.scene_number}\n"
        f"- Duration: {scene.duration_seconds} seconds\n"
        f"- Motion intent for later animation: {scene.motion_prompt}\n\n"
        "Image requirements:\n"
        "- Vertical 9:16 composition.\n"
        "- Use the closest supported OpenAI image size for portrait output.\n"
        "- Save the final image as PNG.\n"
        "- Do not stretch the generated image.\n"
        "- If a reference image is provided, keep the same hair and face identity as the reference image.\n"
        "- If the API returns a different supported portrait size, preserve aspect ratio and let the existing renderer handle final 1080x1920 video sizing."
    )


def get_openai_api_key() -> str:
    return os.getenv(OPENAI_API_KEY_ENV, "").strip()


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
            settings=self.settings or VideoSettings(15, 5, [5, 5, 5], 3, 30, "9:16", 1080, 1920, "nursery", "3D Nursery Animation", "Draft", "Still", True, "English", "Warm Female", "Warm", "Normal"),
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
        self.continuity_reference_path: Path | None = None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def update_continuity_reference(self, image_path: Path) -> None:
        if self.continuity_reference_path is None:
            self.continuity_reference_path = image_path

    def _generate_from_reference(
        self,
        client: Any,
        scene: VideoScene,
        prompt: str,
    ) -> Any:
        if self.continuity_reference_path is None:
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
            self.continuity_reference_path.name,
        )
        with self.continuity_reference_path.open("rb") as reference_image:
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
        self.update_continuity_reference(output_path)
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
    def generate_narration_audio(self, narration: str, output_path: Path) -> ProviderResult:
        word_count = max(1, len(narration.split()))
        duration_seconds = max(15.0, word_count / 2.5)
        create_silent_wav(output_path=output_path, duration_seconds=duration_seconds)
        return ProviderResult(
            path=output_path,
            used_fallback=True,
            message="Created silent narration fallback audio.",
        )


class EnvSpeechProvider:
    def __init__(self) -> None:
        self.api_url = os.getenv("AUMSTATE_SPEECH_API_URL", "").strip()
        self.api_key = os.getenv("AUMSTATE_SPEECH_API_KEY", "").strip()

    def is_configured(self) -> bool:
        return bool(self.api_url and self.api_key)

    def generate_narration_audio(self, narration: str, output_path: Path) -> ProviderResult:
        if not self.is_configured():
            raise ValueError(
                "Speech generation credentials are missing. Set AUMSTATE_SPEECH_API_URL and "
                "AUMSTATE_SPEECH_API_KEY or use the silent audio fallback."
            )

        payload = json.dumps({"text": narration}).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                audio_bytes = response.read()
        except urllib.error.URLError as error:
            raise RuntimeError(f"Speech generation request failed: {error}") from error

        ensure_directory(output_path)
        output_path.write_bytes(audio_bytes)
        return ProviderResult(
            path=output_path,
            used_fallback=False,
            message="Generated narration audio using the configured API provider.",
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

    provider = speech_provider or EnvSpeechProvider()
    output_path = build_narration_audio_path(output_dir=output_dir)
    available_duration_seconds = max(0.5, settings.total_duration_seconds - 0.5)

    try:
        result = provider.generate_narration_audio(plan.narration, output_path)
    except Exception as error:
        result = SilentSpeechProvider().generate_narration_audio(plan.narration, output_path)
        return result.path, f"Speech generation unavailable. Using silent fallback audio. Details: {error}"

    try:
        narration_duration = measure_wav_duration(result.path)
    except Exception as error:
        return result.path, f"{result.message} Audio duration could not be measured. Details: {error}"

    if narration_duration <= available_duration_seconds:
        return result.path, result.message

    shortened_narration = shorten_narration_once(
        narration=plan.narration,
        settings=settings,
        available_duration_seconds=available_duration_seconds,
    )

    retry_path = output_dir / "narration_retry.wav"
    retry_result = provider.generate_narration_audio(shortened_narration, retry_path)
    retry_duration = measure_wav_duration(retry_result.path)
    if retry_duration > available_duration_seconds:
        return retry_result.path, (
            f"{retry_result.message} Narration still exceeds the available duration after one retry."
        )

    return retry_result.path, (
        f"{retry_result.message} Narration was shortened once to fit the selected duration."
    )
