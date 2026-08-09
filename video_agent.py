from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import ollama

from creative_models import CreativeResult


OLLAMA_MODEL = "qwen3:8b"
DEFAULT_TOTAL_DURATION_SECONDS = 15
DEFAULT_SCENE_DURATION_SECONDS = 5
DEFAULT_FRAME_RATE = 30
DEFAULT_ASPECT_RATIO_LABEL = "Vertical 9:16"
DEFAULT_VISUAL_STYLE = "Quiet Cinematic Animation"
DEFAULT_IMAGE_QUALITY = "Draft"
DEFAULT_MOTION_LEVEL = "Still"
DEFAULT_VIDEO_MODE = "basic_motion"
DEFAULT_NARRATION_ENABLED = True
DEFAULT_LANGUAGE = "English"
DEFAULT_VOICE = "Warm Female"
DEFAULT_SPEAKING_STYLE = "Warm"
DEFAULT_SPEAKING_SPEED = "Normal"

VIDEO_DURATION_OPTIONS = [10, 15, 30, 45, 60]
SCENE_DURATION_OPTIONS = [3, 4, 5, 6]
FRAME_RATE_OPTIONS = [24, 30]
ASPECT_RATIO_OPTIONS = {
    "Vertical 9:16": ("9:16", 1080, 1920),
    "Landscape 16:9": ("16:9", 1920, 1080),
    "Square 1:1": ("1:1", 1080, 1080),
}
CONTENT_TYPE_LABELS = {
    "nursery": "Nursery",
    "simple_life": "Simple Life",
    "humor": "Humor",
    "psychology": "Psychology",
    "philosophy": "Philosophy",
    "human_behavior": "Human Behavior",
    "spiritual_reflection": "Spiritual Reflection",
    "education": "Educational",
    "story": "Story",
    "explainer": "Explainer",
}
CONTENT_TYPE_OPTIONS = list(CONTENT_TYPE_LABELS.keys())
CONTENT_TYPE_DISPLAY_OPTIONS = [CONTENT_TYPE_LABELS[key] for key in CONTENT_TYPE_OPTIONS]
CONTENT_TYPE_LABEL_TO_VALUE = {label: value for value, label in CONTENT_TYPE_LABELS.items()}
VISUAL_STYLE_OPTIONS = [
    "Quiet Cinematic Animation",
    "Animated Realism",
    "Hand-Painted Storybook",
    "Minimal Illustration",
    "Soft 3D Animation",
    "3D Nursery Animation",
]
LEGACY_VISUAL_STYLE_ALIASES = {
    "Simple Life Story": "Quiet Cinematic Animation",
    "Soft Watercolor": "Hand-Painted Storybook",
    "Storybook Illustration": "Hand-Painted Storybook",
}


@dataclass
class VideoSettings:
    total_duration_seconds: int
    preferred_scene_duration_seconds: int
    scene_durations: list[int]
    scene_count: int
    video_mode: str
    frame_rate: int
    aspect_ratio: str
    output_width: int
    output_height: int
    content_type: str
    visual_style: str
    image_quality: str
    motion_level: str
    narration_enabled: bool
    language: str
    voice: str
    speaking_style: str
    speaking_speed: str


@dataclass
class VideoScene:
    scene_number: int
    duration_seconds: int
    narration: str
    visual_prompt: str
    motion_prompt: str


@dataclass
class VideoPlan:
    title: str
    content_type: str
    duration_seconds: int
    aspect_ratio: str
    narration: str
    style_lock: str
    scenes: list[VideoScene]
    settings: VideoSettings


def normalize_content_type(content_type: str) -> str:
    cleaned = (content_type or "").strip()
    if not cleaned:
        return "nursery"
    lowered = cleaned.lower()
    lowered = lowered.replace("-", "_").replace(" ", "_")
    if lowered in CONTENT_TYPE_OPTIONS:
        return lowered
    if cleaned in CONTENT_TYPE_LABEL_TO_VALUE:
        return CONTENT_TYPE_LABEL_TO_VALUE[cleaned]
    for value, label in CONTENT_TYPE_LABELS.items():
        if lowered == label.lower().replace(" ", "_"):
            return value
    return "nursery"


def normalize_visual_style(visual_style: str, content_type: str) -> str:
    cleaned = (visual_style or "").strip()
    if cleaned in LEGACY_VISUAL_STYLE_ALIASES:
        return LEGACY_VISUAL_STYLE_ALIASES[cleaned]
    if cleaned in VISUAL_STYLE_OPTIONS:
        return cleaned
    if normalize_content_type(content_type) == "nursery":
        return "3D Nursery Animation"
    return DEFAULT_VISUAL_STYLE


def build_scene_durations(total_duration_seconds: int, preferred_scene_duration_seconds: int) -> list[int]:
    if total_duration_seconds <= 0:
        raise ValueError("Total duration must be positive.")
    if preferred_scene_duration_seconds <= 0:
        raise ValueError("Preferred scene duration must be positive.")

    scene_durations: list[int] = []
    remaining = total_duration_seconds

    while remaining > preferred_scene_duration_seconds:
        scene_durations.append(preferred_scene_duration_seconds)
        remaining -= preferred_scene_duration_seconds

    if remaining <= 0:
        raise ValueError("Scene duration calculation produced a zero-length final scene.")

    scene_durations.append(remaining)
    return scene_durations


def build_video_settings(
    total_duration_seconds: int = DEFAULT_TOTAL_DURATION_SECONDS,
    preferred_scene_duration_seconds: int = DEFAULT_SCENE_DURATION_SECONDS,
    video_mode: str = DEFAULT_VIDEO_MODE,
    frame_rate: int = DEFAULT_FRAME_RATE,
    aspect_ratio_label: str = DEFAULT_ASPECT_RATIO_LABEL,
    content_type: str = "nursery",
    visual_style: str = DEFAULT_VISUAL_STYLE,
    image_quality: str = DEFAULT_IMAGE_QUALITY,
    motion_level: str = DEFAULT_MOTION_LEVEL,
    narration_enabled: bool = DEFAULT_NARRATION_ENABLED,
    language: str = DEFAULT_LANGUAGE,
    voice: str = DEFAULT_VOICE,
    speaking_style: str = DEFAULT_SPEAKING_STYLE,
    speaking_speed: str = DEFAULT_SPEAKING_SPEED,
) -> VideoSettings:
    normalized_total_duration = int(total_duration_seconds)
    normalized_scene_duration = int(preferred_scene_duration_seconds)
    normalized_frame_rate = int(frame_rate)

    if normalized_total_duration not in VIDEO_DURATION_OPTIONS:
        normalized_total_duration = DEFAULT_TOTAL_DURATION_SECONDS
    if normalized_scene_duration not in SCENE_DURATION_OPTIONS:
        normalized_scene_duration = DEFAULT_SCENE_DURATION_SECONDS
    if normalized_frame_rate not in FRAME_RATE_OPTIONS:
        normalized_frame_rate = DEFAULT_FRAME_RATE

    normalized_content_type = normalize_content_type(content_type)
    normalized_visual_style = normalize_visual_style(visual_style, normalized_content_type)

    aspect_ratio, output_width, output_height = ASPECT_RATIO_OPTIONS.get(
        aspect_ratio_label,
        ASPECT_RATIO_OPTIONS[DEFAULT_ASPECT_RATIO_LABEL],
    )

    scene_durations = build_scene_durations(
        total_duration_seconds=normalized_total_duration,
        preferred_scene_duration_seconds=normalized_scene_duration,
    )

    return VideoSettings(
        total_duration_seconds=normalized_total_duration,
        preferred_scene_duration_seconds=normalized_scene_duration,
        scene_durations=scene_durations,
        scene_count=len(scene_durations),
        video_mode=video_mode.strip() or DEFAULT_VIDEO_MODE,
        frame_rate=normalized_frame_rate,
        aspect_ratio=aspect_ratio,
        output_width=output_width,
        output_height=output_height,
        content_type=normalized_content_type,
        visual_style=normalized_visual_style,
        image_quality=image_quality.strip() or DEFAULT_IMAGE_QUALITY,
        motion_level=motion_level.strip() or DEFAULT_MOTION_LEVEL,
        narration_enabled=bool(narration_enabled),
        language=language.strip() or DEFAULT_LANGUAGE,
        voice=voice.strip() or DEFAULT_VOICE,
        speaking_style=speaking_style.strip() or DEFAULT_SPEAKING_STYLE,
        speaking_speed=speaking_speed.strip() or DEFAULT_SPEAKING_SPEED,
    )


def settings_snapshot(settings: VideoSettings) -> str:
    return json.dumps(asdict(settings), sort_keys=True)


def settings_changed(previous_snapshot: str, settings: VideoSettings) -> bool:
    if not previous_snapshot:
        return False
    return previous_snapshot != settings_snapshot(settings)


def total_duration_seconds(scenes: list[VideoScene]) -> int:
    return sum(scene.duration_seconds for scene in scenes)


def normalize_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def parse_plan_json(raw_text: str) -> dict[str, Any]:
    return json.loads(normalize_json_text(raw_text))


def apply_style_lock_to_visual(style_lock: str, visual_prompt: str) -> str:
    style_lock_clean = style_lock.strip()
    visual_clean = visual_prompt.strip()
    if not style_lock_clean:
        return visual_clean
    if style_lock_clean.lower() in visual_clean.lower():
        return visual_clean
    return f"{style_lock_clean}. {visual_clean}"


def narration_word_target(settings: VideoSettings) -> tuple[int, int] | None:
    if settings.language.lower() != "english":
        return None

    ranges = {
        10: (15, 22),
        15: (25, 35),
        30: (55, 70),
        45: (80, 100),
        60: (105, 135),
    }
    return ranges.get(settings.total_duration_seconds, (25, 35))


def validate_video_plan_data(plan_data: dict[str, Any], settings: VideoSettings) -> VideoPlan:
    required_fields = [
        "title",
        "content_type",
        "duration_seconds",
        "aspect_ratio",
        "narration",
        "style_lock",
        "scenes",
    ]
    missing_fields = [field for field in required_fields if field not in plan_data]
    if missing_fields:
        raise ValueError(f"Missing required video plan fields: {', '.join(missing_fields)}")

    raw_scenes = plan_data["scenes"]
    if not isinstance(raw_scenes, list):
        raise ValueError("Video plan scenes must be a list.")
    if len(raw_scenes) != settings.scene_count:
        raise ValueError(f"Video plan must contain exactly {settings.scene_count} scenes.")

    scenes: list[VideoScene] = []
    style_lock = str(plan_data["style_lock"]).strip()
    for index, scene_data in enumerate(raw_scenes, start=1):
        if not isinstance(scene_data, dict):
            raise ValueError(f"Scene {index} must be an object.")

        scene_number = int(scene_data.get("scene_number", index))
        duration_seconds = int(scene_data.get("duration_seconds", 0))
        narration = str(scene_data.get("narration", "")).strip()
        visual_prompt = str(scene_data.get("visual_prompt", "")).strip()
        motion_prompt = str(scene_data.get("motion_prompt", "")).strip()
        expected_duration = settings.scene_durations[index - 1]

        if scene_number != index:
            raise ValueError(f"Scene {index} must use scene_number {index}.")
        if duration_seconds != expected_duration:
            raise ValueError(
                f"Scene {index} duration must be {expected_duration} seconds, got {duration_seconds}."
            )
        if duration_seconds <= 0:
            raise ValueError(f"Scene {index} duration must be positive.")
        if not narration:
            raise ValueError(f"Scene {index} narration is required.")
        if not visual_prompt:
            raise ValueError(f"Scene {index} visual prompt is required.")
        if not motion_prompt:
            raise ValueError(f"Scene {index} motion prompt is required.")

        scenes.append(
            VideoScene(
                scene_number=scene_number,
                duration_seconds=duration_seconds,
                narration=narration,
                visual_prompt=apply_style_lock_to_visual(style_lock, visual_prompt),
                motion_prompt=motion_prompt,
            )
        )

    plan = VideoPlan(
        title=str(plan_data["title"]).strip(),
        content_type=normalize_content_type(str(plan_data["content_type"]).strip()),
        duration_seconds=int(plan_data["duration_seconds"]),
        aspect_ratio=str(plan_data["aspect_ratio"]).strip(),
        narration=str(plan_data["narration"]).strip(),
        style_lock=style_lock,
        scenes=scenes,
        settings=settings,
    )

    if not plan.title:
        raise ValueError("Video plan title is required.")
    if not plan.content_type:
        raise ValueError("Video plan content_type is required.")
    if not plan.narration:
        raise ValueError("Video plan narration is required.")
    if plan.aspect_ratio != settings.aspect_ratio:
        raise ValueError(f"Video plan aspect ratio must be {settings.aspect_ratio}.")
    if plan.duration_seconds != settings.total_duration_seconds:
        raise ValueError(f"Video plan duration must be {settings.total_duration_seconds} seconds.")
    if total_duration_seconds(plan.scenes) != settings.total_duration_seconds:
        raise ValueError("Scene durations must sum exactly to the selected total duration.")

    return plan


def build_fallback_style_lock(idea: str, settings: VideoSettings) -> str:
    base = (
        f"{settings.visual_style}, preserve one recurring protagonist identity, "
        "stable face, hair, clothing, proportions, accessories, palette, environment, and lighting."
    )
    if settings.content_type == "nursery":
        return f"{base} Child-safe warmth and playful clarity based on: {idea}"
    return f"{base} Match the requested tone and world based on: {idea}"


def build_fallback_plan(idea: str, settings: VideoSettings) -> VideoPlan:
    cleaned_idea = " ".join(idea.split()).strip() or "A simple visual story"
    title_base = cleaned_idea[:60].strip().rstrip(".")
    title = title_base.title() if title_base else "Storyboard MVP"
    style_lock = build_fallback_style_lock(cleaned_idea, settings)

    scenes: list[VideoScene] = []
    for index, duration_seconds in enumerate(settings.scene_durations, start=1):
        if index == 1:
            narration = f"Scene {index} opens the central idea: {cleaned_idea}."
            visual_prompt = "Opening frame that establishes the protagonist, setting, and emotional tone."
        elif index == settings.scene_count:
            narration = f"Scene {index} lands the final turn and leaves a clear closing image."
            visual_prompt = "Closing frame that resolves the central idea while preserving the same world and character."
        else:
            narration = f"Scene {index} develops the contradiction or progression in one concise beat."
            visual_prompt = "Middle frame showing the next visual beat, with continuity from the earlier scene."

        motion_prompt = "Subtle cinematic camera movement that fits a still-image animation workflow."
        scenes.append(
            VideoScene(
                scene_number=index,
                duration_seconds=duration_seconds,
                narration=narration,
                visual_prompt=apply_style_lock_to_visual(style_lock, visual_prompt),
                motion_prompt=motion_prompt,
            )
        )

    return VideoPlan(
        title=title,
        content_type=settings.content_type,
        duration_seconds=settings.total_duration_seconds,
        aspect_ratio=settings.aspect_ratio,
        narration=" ".join(scene.narration for scene in scenes),
        style_lock=style_lock,
        scenes=scenes,
        settings=settings,
    )


def build_planning_prompt(idea: str, settings: VideoSettings, creative_brief: str = "") -> str:
    scene_examples = ",\n".join(
        (
            "    {\n"
            f'      "scene_number": {index},\n'
            f'      "duration_seconds": {duration_seconds},\n'
            '      "narration": "Scene narration",\n'
            '      "visual_prompt": "Image prompt",\n'
            '      "motion_prompt": "Centered zoom instructions"\n'
            "    }"
        )
        for index, duration_seconds in enumerate(settings.scene_durations, start=1)
    )
    narration_target = narration_word_target(settings)
    if narration_target is not None:
        narration_guidance = f"- Keep the full narration around {narration_target[0]} to {narration_target[1]} words.\n"
    else:
        narration_guidance = (
            f"- Keep the narration natural for {settings.language} and short enough to finish about 0.5 seconds before the video ends.\n"
        )

    creative_section = ""
    if creative_brief.strip():
        creative_section = f"\nCreative intelligence brief:\n{creative_brief.strip()}\n"

    return f"""
You are a video storyboard planner for a Streamlit app.

Return only valid JSON with this exact schema:
{{
  "title": "Short title",
  "content_type": "{settings.content_type}",
  "duration_seconds": {settings.total_duration_seconds},
  "aspect_ratio": "{settings.aspect_ratio}",
  "narration": "Complete narration text for the full video",
  "style_lock": "A shared visual style description that every scene prompt must reuse",
  "scenes": [
{scene_examples}
  ]
}}

Rules:
- Create exactly {settings.scene_count} scenes.
- Use these exact scene durations in order: {settings.scene_durations}.
- Total duration must be exactly {settings.total_duration_seconds} seconds.
- Every scene_number must be sequential starting at 1.
- Aspect ratio must be {settings.aspect_ratio}.
- Content type must be {settings.content_type}.
- Visual style should align with: {settings.visual_style}.
- The same protagonist and same world must appear across every scene when the concept needs a protagonist.
- The style_lock must describe the recurring protagonist or recurring world precisely enough to preserve continuity.
- The style_lock must explicitly lock hairstyle, hair length, hair color, face shape, body proportions, clothing, palette, species if relevant, accessories, and lighting.
- Keep continuity stable across all scenes.
- Do not change hair length or hairstyle between scenes unless the user explicitly requests a transformation.
- Narration enabled: {"yes" if settings.narration_enabled else "no"}.
- Language: {settings.language}.
- Speaking style: {settings.speaking_style}.
{narration_guidance}- Use visual storytelling where possible.
- Do not add markdown fences.
- Do not add explanatory text.
{creative_section}
Video idea: {idea}
"""


def request_video_plan_from_ollama(idea: str, settings: VideoSettings, creative_brief: str = "") -> str:
    prompt = build_planning_prompt(idea=idea, settings=settings, creative_brief=creative_brief)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": idea},
        ],
    )
    return response["message"]["content"]


def shorten_narration_once(narration: str, settings: VideoSettings, available_duration_seconds: float) -> str:
    narration_target = narration_word_target(settings)
    word_guidance = ""
    if narration_target is not None:
        word_guidance = f"Target about {narration_target[0]} to {narration_target[1]} words."

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You shorten narration to fit a target video duration without changing the meaning. "
                    "Return plain text only.\n"
                    f"Video duration available for speech: about {available_duration_seconds:.1f} seconds.\n"
                    f"Language: {settings.language}. Speaking style: {settings.speaking_style}. {word_guidance}"
                ),
            },
            {"role": "user", "content": narration},
        ],
    )
    return response["message"]["content"].strip()


def build_creative_brief(result: CreativeResult) -> str:
    sections = [
        f"Director intent: {result.director.content_intent}",
        f"Director tone: {result.director.emotional_tone}",
        f"Narrative shape: {result.director.narrative_shape}",
        f"Story focus: {result.director.story_focus}",
        f"Selected specialists: {', '.join(result.selected_specialists)}",
        f"Final premise: {result.final_story.premise}",
        f"Conflict: {result.final_story.conflict}",
        f"Progression: {result.final_story.progression}",
        f"Emotional turn: {result.final_story.emotional_turn}",
        f"Ending: {result.final_story.ending}",
        "Scene beats:",
    ]
    sections.extend(f"- {beat}" for beat in result.final_story.scene_beats)
    if result.psychology:
        sections.append(f"Psychology contradiction: {result.psychology.contradiction}")
    if result.philosophy:
        sections.append(f"Philosophical question: {result.philosophy.central_question}")
        sections.append(f"Source meaning: {result.philosophy.source_meaning}")
        sections.append(f"Modern reflection: {result.philosophy.modern_reflection}")
    if result.ambiguity:
        sections.append(f"Ambiguity to preserve: {result.ambiguity.what_not_to_explain}")
    if result.humor:
        sections.append(f"Humor setup: {result.humor.setup}")
        sections.extend(f"- humor candidate: {candidate}" for candidate in result.humor.punchline_candidates)
    sections.append(f"Critic notes: {result.critic.notes}")
    sections.extend(f"- critic edit: {instruction}" for instruction in result.critic.edit_instructions)
    return "\n".join(sections)


def build_video_plan(idea: str, settings: VideoSettings) -> tuple[VideoPlan, str | None]:
    try:
        raw_response = request_video_plan_from_ollama(idea=idea, settings=settings)
        plan_data = parse_plan_json(raw_response)
        return validate_video_plan_data(plan_data, settings=settings), None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        fallback_plan = build_fallback_plan(idea=idea, settings=settings)
        return fallback_plan, f"Ollama returned an invalid plan. Using fallback storyboard. Details: {error}"


def build_video_plan_from_creative_result(
    idea: str,
    creative_result: CreativeResult,
    settings: VideoSettings,
) -> tuple[VideoPlan, str | None]:
    creative_brief = build_creative_brief(creative_result)
    try:
        raw_response = request_video_plan_from_ollama(
            idea=idea,
            settings=settings,
            creative_brief=creative_brief,
        )
        plan_data = parse_plan_json(raw_response)
        return validate_video_plan_data(plan_data, settings=settings), None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        fallback_plan = build_fallback_plan(idea=idea, settings=settings)
        fallback_plan.content_type = settings.content_type
        return fallback_plan, (
            "Creative pipeline succeeded, but final storyboard JSON was invalid. "
            f"Using fallback storyboard. Details: {error}"
        )
