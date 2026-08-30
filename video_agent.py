from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import ollama

from creative_models import CreativeResult


logger = logging.getLogger(__name__)

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
GENERIC_SCENE_TEXT_FRAGMENTS = (
    "develops the story",
    "next visual beat",
    "introduces the idea",
    "resolves the concept",
    "establishes the protagonist",
    "develops the contradiction",
    "middle frame showing",
    "opening frame that establishes",
    "closing frame that resolves",
    "specific middle image showing",
    "concrete opening image",
    "concrete closing image",
)
FORBIDDEN_RECURRING_SYMBOLS = (
    "glowing orb",
    "magical light",
    "floating symbol",
    "spiritual particle effect",
)
CONTINUITY_STYLE_FRAGMENTS = (
    "keep the same ",
    "same protagonist",
    "same character",
    "same world",
    "preserve one recurring protagonist",
    "stable face",
    "hairstyle",
    "hair length",
    "hair color",
    "face shape",
    "body proportions",
    "clothing",
    "accessories",
)
HUMAN_BEAT_KEYWORDS = (
    "man",
    "woman",
    "person",
    "people",
    "human",
    "couple",
    "child",
    "mother",
    "father",
    "he ",
    "she ",
    "they ",
    "his ",
    "her ",
    "their ",
)
WORLD_BEAT_KEYWORDS = (
    "room",
    "street",
    "house",
    "apartment",
    "office",
    "market",
    "temple",
    "classroom",
)
SOURCE_TEXT_MARKERS = (
    "bhagavad gita",
    "verse",
    "scripture",
    "sutra",
    "upanishad",
    "quran",
    "bible",
    "source text",
)
SOURCE_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "within",
    "inside",
    "using",
    "enclosing",
    "covered",
    "covering",
    "enclosed",
    "life",
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
    continuity_mode: str = "independent"
    continuity_group: str | None = None
    scene_purpose: str = "transition"
    source_anchor_id: str | None = None


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
    used_fallback: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class SourceAnchor:
    canonical_text: str
    concept_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class RequiredSourceAnchor:
    id: str
    meaning: str
    required_objects: tuple[str, ...]
    source_order: int
    allowed_depictions: tuple[str, ...] = ()
    forbidden_replacements: tuple[str, ...] = ()

    @property
    def description(self) -> str:
        return self.meaning


@dataclass(frozen=True)
class SourceSceneAssignment:
    scene_number: int
    source_anchor_id: str | None
    source_anchor: str
    scene_purpose: str


class SourceFidelityValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        scene_number: int | None = None,
        source_anchor_id: str | None = None,
        source_anchor: str | None = None,
        issue_type: str = "drift",
    ) -> None:
        super().__init__(message)
        self.scene_number = scene_number
        self.source_anchor_id = source_anchor_id
        self.source_anchor = source_anchor
        self.issue_type = issue_type


class StoryboardSceneFieldError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        scene_number: int,
        missing_field: str,
    ) -> None:
        super().__init__(message)
        self.scene_number = scene_number
        self.missing_field = missing_field


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


def is_generic_scene_text(text: str) -> bool:
    lowered = " ".join((text or "").lower().split())
    return any(fragment in lowered for fragment in GENERIC_SCENE_TEXT_FRAGMENTS)


def has_authorized_symbol(symbol: str, idea: str, creative_result: CreativeResult | None = None) -> bool:
    haystacks = [idea.lower()]
    if creative_result is not None:
        haystacks.extend(
            [
                creative_result.final_story.premise.lower(),
                creative_result.final_story.progression.lower(),
                creative_result.final_story.emotional_turn.lower(),
                creative_result.final_story.ending.lower(),
                " ".join(creative_result.final_story.scene_beats).lower(),
            ]
        )
        if creative_result.philosophy:
            haystacks.extend(
                [
                    creative_result.philosophy.source_meaning.lower(),
                    creative_result.philosophy.modern_reflection.lower(),
                    creative_result.philosophy.deeper_meaning.lower(),
                ]
            )
    return any(symbol in haystack for haystack in haystacks)


def infer_scene_continuity(scene_beats: list[str]) -> list[tuple[str, str | None]]:
    hints: list[tuple[str, str | None]] = []
    shared_character_group: str | None = None
    shared_world_group: str | None = None

    for beat in scene_beats:
        lowered = f" {(beat or '').lower()} "
        mentions_human = any(keyword in lowered for keyword in HUMAN_BEAT_KEYWORDS)
        mentions_world = any(keyword in lowered for keyword in WORLD_BEAT_KEYWORDS)

        if mentions_human:
            shared_character_group = shared_character_group or "human_a"
            hints.append(("character", shared_character_group))
            continue
        if mentions_world:
            shared_world_group = shared_world_group or "world_a"
            hints.append(("world", shared_world_group))
            continue
        hints.append(("independent", None))

    return hints


def extract_explicit_source_metaphors(idea: str) -> list[str]:
    cleaned = " ".join((idea or "").split()).strip()
    if not cleaned:
        return []

    lowered = cleaned.lower()
    captured = ""
    using_match = re.search(r"\busing\s+(.+?)(?:\.\s+use\b|$)", cleaned, flags=re.IGNORECASE)
    if using_match:
        captured = using_match.group(1).strip(" .")
    elif "smoke covering fire" in lowered and "dust covering a mirror" in lowered:
        captured = cleaned

    if not captured:
        return []

    normalized = captured.replace(", and ", ", ").replace(" and ", ", ")
    parts = [part.strip(" .") for part in normalized.split(",") if part.strip(" .")]
    metaphors: list[str] = []
    for part in parts:
        lowered_part = part.lower()
        if "womb" in lowered_part and "unborn" in lowered_part:
            metaphors.append("unborn life enclosed within the womb")
        else:
            metaphors.append(lowered_part)
    return metaphors


def is_source_based_reflection(idea: str, content_type: str) -> bool:
    lowered = (idea or "").lower()
    return (
        any(marker in lowered for marker in SOURCE_TEXT_MARKERS) or bool(extract_explicit_source_metaphors(idea))
    ) and content_type in {"philosophy", "spiritual_reflection"}


def metaphor_keywords(metaphor: str) -> list[str]:
    words = re.findall(r"[a-z]+", metaphor.lower())
    keywords = [word for word in words if word not in SOURCE_STOPWORDS and len(word) > 2]
    return list(dict.fromkeys(keywords))


def build_required_source_anchors(source_metaphors: list[str]) -> list[RequiredSourceAnchor]:
    anchors: list[RequiredSourceAnchor] = []
    for index, metaphor in enumerate(source_metaphors, start=1):
        normalized = build_source_anchor(metaphor)
        if normalized.canonical_text == "smoke covering fire":
            anchors.append(
                RequiredSourceAnchor(
                    id=f"anchor_{index}",
                    meaning=normalized.canonical_text,
                    required_objects=("smoke", "fire"),
                    source_order=index,
                    forbidden_replacements=("glowing orb",),
                )
            )
        elif normalized.canonical_text == "dust covering a mirror":
            anchors.append(
                RequiredSourceAnchor(
                    id=f"anchor_{index}",
                    meaning=normalized.canonical_text,
                    required_objects=("dust", "mirror"),
                    source_order=index,
                    forbidden_replacements=("pool of water",),
                )
            )
        elif normalized.canonical_text == "unborn life enclosed within the womb":
            anchors.append(
                RequiredSourceAnchor(
                    id=f"anchor_{index}",
                    meaning=normalized.canonical_text,
                    required_objects=("unborn life", "womb"),
                    source_order=index,
                    allowed_depictions=(
                        "warm abstract womb environment",
                        "fetal silhouette",
                        "unborn life enclosed in a maternal form",
                        "non-medical symbolic depiction",
                    ),
                    forbidden_replacements=("seed", "clay jar", "cocoon", "glowing orb", "egg", "closed flower", "container"),
                )
            )
        else:
            anchors.append(
                RequiredSourceAnchor(
                    id=f"anchor_{index}",
                    meaning=normalized.canonical_text,
                    required_objects=tuple(metaphor_keywords(metaphor)),
                    source_order=index,
                )
            )
    return anchors


def build_source_anchor(metaphor: str) -> SourceAnchor:
    lowered = metaphor.lower()
    if "smoke" in lowered and "fire" in lowered:
        return SourceAnchor(
            canonical_text="smoke covering fire",
            concept_groups=(("smoke",), ("fire", "flame", "blaze", "ember")),
        )
    if "dust" in lowered and "mirror" in lowered:
        return SourceAnchor(
            canonical_text="dust covering a mirror",
            concept_groups=(("dust",), ("mirror", "looking glass")),
        )
    if "womb" in lowered and "unborn" in lowered:
        return SourceAnchor(
            canonical_text="unborn life enclosed within the womb",
            concept_groups=(("womb",), ("unborn", "child", "baby", "infant", "embryo", "fetus", "foetus")),
        )
    keywords = tuple((keyword,) for keyword in metaphor_keywords(metaphor))
    return SourceAnchor(canonical_text=metaphor.lower(), concept_groups=keywords)


def text_matches_source_anchor(text: str, anchor: SourceAnchor) -> bool:
    lowered = " ".join((text or "").lower().replace("-", " ").split())
    if not lowered:
        return False
    return all(any(term in lowered for term in group) for group in anchor.concept_groups)


def build_source_scene_slots(required_source_anchors: list[RequiredSourceAnchor], scene_count: int) -> list[SourceSceneAssignment]:
    if not required_source_anchors:
        return []
    if scene_count < len(required_source_anchors):
        raise SourceFidelityValidationError(
            "Storyboard does not contain enough source-metaphor scenes to preserve the explicit source metaphors.",
            issue_type="missing",
        )

    assignments: list[SourceSceneAssignment] = []
    for anchor in required_source_anchors:
        assignments.append(
            SourceSceneAssignment(
                scene_number=anchor.source_order,
                source_anchor_id=anchor.id,
                source_anchor=anchor.meaning,
                scene_purpose="source_metaphor",
            )
        )
    return assignments


def build_anchor_lookup(required_source_anchors: list[RequiredSourceAnchor]) -> dict[str, RequiredSourceAnchor]:
    return {anchor.id: anchor for anchor in required_source_anchors}


def log_anchor_trace(stage: str, required_source_anchors: list[RequiredSourceAnchor], scene_slots: list[SourceSceneAssignment]) -> None:
    anchor_by_id = build_anchor_lookup(required_source_anchors)
    for slot in scene_slots:
        anchor = anchor_by_id.get(slot.source_anchor_id or "")
        logger.info(
            "video anchor trace stage=%s scene=%s purpose=%s anchor_id=%s anchor=%s order=%s",
            stage,
            slot.scene_number,
            slot.scene_purpose,
            slot.source_anchor_id,
            anchor.description if anchor else "",
            anchor.source_order if anchor else "",
        )


def build_scene_slot_plan(required_source_anchors: list[RequiredSourceAnchor], scene_count: int) -> list[SourceSceneAssignment]:
    assignments = build_source_scene_slots(required_source_anchors=required_source_anchors, scene_count=scene_count)
    assignment_by_scene = {assignment.scene_number: assignment for assignment in assignments}
    source_scene_count = len(assignments)

    scene_purposes: list[SourceSceneAssignment] = []
    for scene_number in range(1, scene_count + 1):
        if scene_number in assignment_by_scene:
            scene_purposes.append(assignment_by_scene[scene_number])
        elif required_source_anchors and scene_number == scene_count:
            scene_purposes.append(
                SourceSceneAssignment(
                    scene_number=scene_number,
                    source_anchor_id=None,
                    source_anchor="",
                    scene_purpose="conclusion",
                )
            )
        elif required_source_anchors and scene_number > source_scene_count:
            scene_purposes.append(
                SourceSceneAssignment(
                    scene_number=scene_number,
                    source_anchor_id=None,
                    source_anchor="",
                    scene_purpose="modern_reflection",
                )
            )
        else:
            scene_purposes.append(
                SourceSceneAssignment(
                    scene_number=scene_number,
                    source_anchor_id=None,
                    source_anchor="",
                    scene_purpose="transition",
                )
            )
    return scene_purposes


def compact_scene_imagery(scene: VideoScene) -> str:
    imagery = " ".join(part.strip() for part in (scene.narration, scene.visual_prompt) if part.strip())
    imagery = re.sub(r"\s+", " ", imagery).strip()
    return imagery[:160].rstrip(" .,")


def scene_to_plan_dict(scene: VideoScene) -> dict[str, Any]:
    return {
        "scene_number": scene.scene_number,
        "duration_seconds": scene.duration_seconds,
        "narration": scene.narration,
        "visual_prompt": scene.visual_prompt,
        "motion_prompt": scene.motion_prompt,
        "scene_purpose": scene.scene_purpose,
        "source_anchor_id": scene.source_anchor_id,
        "story_anchor_id": scene.source_anchor_id,
        "continuity_mode": scene.continuity_mode,
        "continuity_group": scene.continuity_group,
    }


def apply_scene_slot_metadata(plan_data: dict[str, Any], scene_slots: list[SourceSceneAssignment]) -> dict[str, Any]:
    slot_by_scene = {slot.scene_number: slot for slot in scene_slots}
    for scene_data in plan_data.get("scenes", []):
        scene_number = int(scene_data.get("scene_number", 0))
        slot = slot_by_scene.get(scene_number)
        if slot is None:
            continue
        scene_data["scene_purpose"] = slot.scene_purpose
        scene_data["source_anchor_id"] = slot.source_anchor_id
        scene_data["story_anchor_id"] = slot.source_anchor_id
    return plan_data


def merge_model_storyboard_with_scene_slots(
    plan_data: dict[str, Any],
    scene_slots: list[SourceSceneAssignment],
) -> dict[str, Any]:
    raw_scenes = plan_data.get("scenes")
    if not isinstance(raw_scenes, list):
        return plan_data
    slot_by_scene = {slot.scene_number: slot for slot in scene_slots}
    merged_scenes: list[dict[str, Any]] = []
    for index, scene_data in enumerate(raw_scenes, start=1):
        slot = slot_by_scene.get(index)
        merged_scene = {
            "scene_number": index,
            "narration": str(scene_data.get("narration", "")).strip(),
            "visual_prompt": str(scene_data.get("visual_prompt", "")).strip(),
            "motion_prompt": str(scene_data.get("motion_prompt", "")).strip(),
            "scene_purpose": slot.scene_purpose if slot else str(scene_data.get("scene_purpose", "transition")).strip() or "transition",
            "source_anchor_id": slot.source_anchor_id if slot else extract_scene_anchor_id(scene_data),
            "story_anchor_id": slot.source_anchor_id if slot else extract_scene_anchor_id(scene_data),
            "continuity_mode": str(scene_data.get("continuity_mode", "independent")).strip() or "independent",
            "continuity_group": scene_data.get("continuity_group"),
        }
        merged_scenes.append(merged_scene)
    return {
        "title": plan_data.get("title", ""),
        "narration": plan_data.get("narration", ""),
        "scenes": merged_scenes,
    }


def extract_scene_anchor_id(scene_data: dict[str, Any]) -> str | None:
    source_anchor_raw = scene_data.get("source_anchor_id")
    if source_anchor_raw is not None:
        anchor_id = str(source_anchor_raw).strip()
        return anchor_id or None
    story_anchor_raw = scene_data.get("story_anchor_id")
    if story_anchor_raw is not None:
        anchor_id = str(story_anchor_raw).strip()
        return anchor_id or None
    return None


def derive_motion_prompt_from_scene(scene_data: dict[str, Any], scene_purpose: str) -> str | None:
    text = " ".join(
        str(scene_data.get(field, "")).strip().lower()
        for field in ("narration", "visual_prompt")
        if str(scene_data.get(field, "")).strip()
    )
    if not text:
        return None
    if "smoke" in text or "fire" in text or "flame" in text:
        return "Smoke drifts slowly while the flame flickers and the camera makes a restrained push in."
    if "dust" in text and "mirror" in text:
        return "Fine dust settles across the mirror as the camera slowly moves closer."
    if "womb" in text or "unborn" in text:
        return "A slow suspended drift holds the protective enclosure in stillness."
    if scene_purpose in {"modern_reflection", "conclusion"} or any(
        keyword in text for keyword in ("person", "human", "face", "phone", "breath", "room")
    ):
        return "Subtle breathing and a gentle camera drift keep the scene physically grounded."
    return None


def scene_neighbor_summaries(plan_data: dict[str, Any], scene_number: int) -> list[str]:
    summaries: list[str] = []
    raw_scenes = plan_data.get("scenes", [])
    for neighbor_number in (scene_number - 1, scene_number + 1):
        if 1 <= neighbor_number <= len(raw_scenes):
            neighbor = raw_scenes[neighbor_number - 1]
            narration = str(neighbor.get("narration", "")).strip()
            visual_prompt = str(neighbor.get("visual_prompt", "")).strip()
            scene_purpose = str(neighbor.get("scene_purpose", "")).strip()
            summaries.append(
                f"Scene {neighbor_number}: purpose={scene_purpose or 'unknown'}; narration={narration or 'missing'}; visual={visual_prompt or 'missing'}"
            )
    return summaries


def build_default_style_lock(settings: VideoSettings) -> str:
    style_templates = {
        "Quiet Cinematic Animation": "Quiet cinematic animation, soft natural light, muted earthy colors, realistic proportions, restrained movement, contemplative atmosphere.",
        "Animated Realism": "Animated realism, believable anatomy, natural textures, soft directional light, grounded color palette, understated motion.",
        "Hand-Painted Storybook": "Hand-painted storybook aesthetic, textured brushwork, gentle light, warm restrained palette, clear silhouettes, lyrical calm.",
        "Minimal Illustration": "Minimal illustration, clean shapes, limited color palette, spacious composition, subtle contrast, calm visual rhythm.",
        "Soft 3D Animation": "Soft 3D animation, tactile surfaces, diffused light, balanced colors, approachable realism, gentle camera language.",
        "3D Nursery Animation": "3D nursery animation, rounded forms, child-safe warmth, soft daylight, tidy compositions, playful but restrained motion.",
    }
    return style_templates.get(
        settings.visual_style,
        "Cinematic animation, soft natural light, coherent palette, realistic proportions, restrained movement, clear visual storytelling.",
    )


def sanitize_style_lock(
    style_lock: str,
    idea: str,
    settings: VideoSettings,
    creative_result: CreativeResult | None = None,
) -> str:
    cleaned = " ".join((style_lock or "").split()).strip()
    if not cleaned:
        return build_default_style_lock(settings)

    lowered = cleaned.lower()
    story_sources = [idea.lower()]
    if creative_result is not None:
        story_sources.extend(
            [
                creative_result.final_story.premise.lower(),
                creative_result.final_story.conflict.lower(),
                creative_result.final_story.progression.lower(),
                creative_result.final_story.emotional_turn.lower(),
                creative_result.final_story.ending.lower(),
            ]
        )
    if any(source and source in lowered for source in story_sources):
        return build_default_style_lock(settings)
    if any(fragment in lowered for fragment in CONTINUITY_STYLE_FRAGMENTS):
        return build_default_style_lock(settings)
    return cleaned


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
        continuity_mode = str(scene_data.get("continuity_mode", "independent")).strip() or "independent"
        continuity_group_raw = scene_data.get("continuity_group")
        continuity_group = str(continuity_group_raw).strip() if continuity_group_raw is not None else None
        scene_purpose = str(scene_data.get("scene_purpose", "transition")).strip() or "transition"
        source_anchor_id_raw = scene_data.get("source_anchor_id")
        source_anchor_id = str(source_anchor_id_raw).strip() if source_anchor_id_raw is not None else None
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
        if continuity_mode not in {"independent", "character", "world", "previous_scene"}:
            raise ValueError(f"Scene {index} continuity_mode is invalid: {continuity_mode}.")
        if continuity_mode == "independent":
            continuity_group = None
        elif not continuity_group:
            raise ValueError(f"Scene {index} continuity_group is required for continuity_mode {continuity_mode}.")
        if is_generic_scene_text(narration):
            raise ValueError(f"Scene {index} narration is too generic.")
        if is_generic_scene_text(visual_prompt):
            raise ValueError(f"Scene {index} visual prompt is too generic.")
        if is_generic_scene_text(motion_prompt):
            raise ValueError(f"Scene {index} motion prompt is too generic.")
        if scene_purpose not in {"source_metaphor", "transition", "modern_reflection", "conclusion"}:
            raise ValueError(f"Scene {index} scene_purpose is invalid: {scene_purpose}.")
        if scene_purpose == "source_metaphor" and not source_anchor_id:
            raise ValueError(f"Scene {index} source_anchor_id is required for source_metaphor scenes.")
        if scene_purpose != "source_metaphor":
            source_anchor_id = None

        scenes.append(
            VideoScene(
                scene_number=scene_number,
                duration_seconds=duration_seconds,
                narration=narration,
                visual_prompt=apply_style_lock_to_visual(style_lock, visual_prompt),
                motion_prompt=motion_prompt,
                continuity_mode=continuity_mode,
                continuity_group=continuity_group,
                scene_purpose=scene_purpose,
                source_anchor_id=source_anchor_id,
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


def validate_storyboard_scene_data(scene_data: dict[str, Any], index: int, settings: VideoSettings) -> VideoScene:
    if not isinstance(scene_data, dict):
        raise ValueError(f"Scene {index} must be an object.")

    scene_number = int(scene_data.get("scene_number", index))
    narration = str(scene_data.get("narration", "")).strip()
    visual_prompt = str(scene_data.get("visual_prompt", "")).strip()
    motion_prompt = str(scene_data.get("motion_prompt", "")).strip()
    continuity_mode = str(scene_data.get("continuity_mode", "independent")).strip() or "independent"
    continuity_group_raw = scene_data.get("continuity_group")
    continuity_group = str(continuity_group_raw).strip() if continuity_group_raw is not None else None
    scene_purpose = str(scene_data.get("scene_purpose", "transition")).strip() or "transition"
    source_anchor_id = extract_scene_anchor_id(scene_data)
    expected_duration = settings.scene_durations[index - 1]

    if scene_number != index:
        raise ValueError(f"Scene {index} must use scene_number {index}.")
    if not narration:
        raise StoryboardSceneFieldError(f"Scene {index} narration is required.", scene_number=index, missing_field="narration")
    if not visual_prompt:
        raise StoryboardSceneFieldError(
            f"Scene {index} visual prompt is required.",
            scene_number=index,
            missing_field="visual_prompt",
        )
    if not motion_prompt:
        raise StoryboardSceneFieldError(
            f"Scene {index} motion prompt is required.",
            scene_number=index,
            missing_field="motion_prompt",
        )
    if continuity_mode not in {"independent", "character", "world", "previous_scene"}:
        raise ValueError(f"Scene {index} continuity_mode is invalid: {continuity_mode}.")
    if continuity_mode == "independent":
        continuity_group = None
    elif not continuity_group:
        raise ValueError(f"Scene {index} continuity_group is required for continuity_mode {continuity_mode}.")
    if is_generic_scene_text(narration):
        raise ValueError(f"Scene {index} narration is too generic.")
    if is_generic_scene_text(visual_prompt):
        raise ValueError(f"Scene {index} visual prompt is too generic.")
    if is_generic_scene_text(motion_prompt):
        raise ValueError(f"Scene {index} motion prompt is too generic.")
    if scene_purpose not in {"source_metaphor", "transition", "modern_reflection", "conclusion"}:
        raise ValueError(f"Scene {index} scene_purpose is invalid: {scene_purpose}.")
    if scene_purpose == "source_metaphor" and not source_anchor_id:
        raise ValueError(f"Scene {index} source_anchor_id is required for source_metaphor scenes.")
    if scene_purpose != "source_metaphor":
        source_anchor_id = None

    return VideoScene(
        scene_number=scene_number,
        duration_seconds=expected_duration,
        narration=narration,
        visual_prompt=visual_prompt,
        motion_prompt=motion_prompt,
        continuity_mode=continuity_mode,
        continuity_group=continuity_group,
        scene_purpose=scene_purpose,
        source_anchor_id=source_anchor_id,
    )


def validate_storyboard_payload(plan_data: dict[str, Any], settings: VideoSettings) -> list[VideoScene]:
    if not isinstance(plan_data, dict):
        raise ValueError("Storyboard payload must be a JSON object.")

    raw_scenes = plan_data.get("scenes")
    if raw_scenes is None:
        raise ValueError("Storyboard payload must include scenes.")
    if not isinstance(raw_scenes, list):
        raise ValueError("Storyboard payload scenes must be a list.")
    if len(raw_scenes) != settings.scene_count:
        raise ValueError(f"Storyboard payload must contain exactly {settings.scene_count} scenes.")

    scenes: list[VideoScene] = []
    for index, scene_data in enumerate(raw_scenes, start=1):
        scenes.append(validate_storyboard_scene_data(scene_data, index=index, settings=settings))
    return scenes


def request_scene_field_repair_from_ollama(
    *,
    idea: str,
    settings: VideoSettings,
    creative_authority: dict[str, Any],
    plan_data: dict[str, Any],
    scene_slot: SourceSceneAssignment,
    anchor: RequiredSourceAnchor | None,
    missing_field: str,
    style_lock: str,
) -> str:
    scene_number = scene_slot.scene_number
    current_scene = next(
        (scene for scene in plan_data.get("scenes", []) if int(scene.get("scene_number", 0)) == scene_number),
        None,
    )
    if current_scene is None:
        raise ValueError(f"Cannot repair Scene {scene_number}; scene payload is missing.")

    prompt = (
        "You repair exactly one incomplete storyboard scene.\n"
        "Return only the corrected Scene JSON object with this schema:\n"
        "{\n"
        '  "scene_number": 1,\n'
        '  "narration": "Scene narration",\n'
        '  "visual_prompt": "Image prompt",\n'
        '  "motion_prompt": "Actual physical or camera motion",\n'
        '  "scene_purpose": "source_metaphor | transition | modern_reflection | conclusion",\n'
        '  "story_anchor_id": null,\n'
        '  "continuity_mode": "independent | character | world | previous_scene",\n'
        '  "continuity_group": null\n'
        "}\n"
        f"Return only the corrected Scene {scene_number} JSON.\n"
        f"The missing field is: {missing_field}.\n"
        "- Do not change the assigned story anchor or scene purpose.\n"
        "- Do not introduce unrelated symbolism.\n"
        "- visual_prompt must describe the concrete visible scene.\n"
        "- narration must sound like narration, not instructions.\n"
        "- motion_prompt must describe actual physical or camera motion.\n"
        f"Original user request: {idea}\n"
        f"Global style lock: {style_lock}\n"
        f"CreativeResult final_story: {json.dumps(creative_authority.get('final_story', {}), ensure_ascii=True)}\n"
        f"Scene number: {scene_number}\n"
        f"Scene purpose: {scene_slot.scene_purpose}\n"
        f"Assigned story anchor: {anchor.description if anchor else 'none'}\n"
        f"Assigned story_anchor_id: {scene_slot.source_anchor_id or 'null'}\n"
        f"Existing narration: {str(current_scene.get('narration', '')).strip() or 'missing'}\n"
        f"Existing visual_prompt: {str(current_scene.get('visual_prompt', '')).strip() or 'missing'}\n"
        f"Existing motion_prompt: {str(current_scene.get('motion_prompt', '')).strip() or 'missing'}\n"
        f"Neighboring scene summaries: {json.dumps(scene_neighbor_summaries(plan_data, scene_number), ensure_ascii=True)}\n"
        f"Current full storyboard: {json.dumps(plan_data, ensure_ascii=True)}\n"
    )
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": idea},
        ],
    )
    return response["message"]["content"]


def repair_or_complete_storyboard_scene(
    *,
    idea: str,
    settings: VideoSettings,
    creative_authority: dict[str, Any],
    plan_data: dict[str, Any],
    scene_slot: SourceSceneAssignment,
    anchor_by_id: dict[str, RequiredSourceAnchor],
    missing_field: str,
    style_lock: str,
) -> dict[str, Any]:
    scene_number = scene_slot.scene_number
    updated_plan_data = dict(plan_data)
    updated_scenes = [dict(scene) for scene in plan_data.get("scenes", [])]
    target_scene = dict(updated_scenes[scene_number - 1])
    scene_hints = creative_authority.get("continuity_requirements", {}).get("scene_hints", [])
    scene_hint = next((item for item in scene_hints if item.get("scene_number") == scene_number), None)
    anchor = anchor_by_id.get(scene_slot.source_anchor_id or "")

    if missing_field == "motion_prompt":
        derived_motion = derive_motion_prompt_from_scene(target_scene, str(target_scene.get("scene_purpose", "")))
        if derived_motion:
            logger.info(
                "video scene field fallback scene=%s missing_field=%s method=deterministic_motion",
                scene_number,
                missing_field,
            )
            target_scene["motion_prompt"] = derived_motion
    elif missing_field == "narration" and scene_hint and scene_hint.get("scene_beat"):
        reconstructed = str(scene_hint["scene_beat"]).strip()
        if reconstructed:
            logger.info(
                "video scene field fallback scene=%s missing_field=%s method=scene_beat",
                scene_number,
                missing_field,
            )
            target_scene["narration"] = reconstructed

    if not str(target_scene.get(missing_field, "")).strip():
        repair_attempts = 0
        last_error: Exception | None = None
        while repair_attempts < 2:
            repair_attempts += 1
            logger.warning(
                "video scene field repair attempt scene=%s missing_field=%s attempt=%s method=model",
                scene_number,
                missing_field,
                repair_attempts,
            )
            try:
                repaired_scene_raw = request_scene_field_repair_from_ollama(
                    idea=idea,
                    settings=settings,
                    creative_authority=creative_authority,
                    plan_data=updated_plan_data,
                    scene_slot=scene_slot,
                    anchor=anchor,
                    missing_field=missing_field,
                    style_lock=style_lock,
                )
                repaired_scene_data = parse_plan_json(repaired_scene_raw)
                if not isinstance(repaired_scene_data, dict):
                    raise ValueError(f"Scene {scene_number} repair did not return a JSON object.")
                repaired_scene_data["scene_number"] = scene_number
                repaired_scene_data["scene_purpose"] = scene_slot.scene_purpose
                repaired_scene_data["source_anchor_id"] = scene_slot.source_anchor_id
                repaired_scene_data["story_anchor_id"] = scene_slot.source_anchor_id
                repaired_scene_data["continuity_mode"] = repaired_scene_data.get("continuity_mode", target_scene.get("continuity_mode"))
                repaired_scene_data["continuity_group"] = repaired_scene_data.get("continuity_group", target_scene.get("continuity_group"))
                validate_storyboard_scene_data(repaired_scene_data, index=scene_number, settings=settings)
                target_scene = repaired_scene_data
                logger.info(
                    "video scene field repair success scene=%s missing_field=%s attempt=%s method=model",
                    scene_number,
                    missing_field,
                    repair_attempts,
                )
                break
            except Exception as error:
                last_error = error
                logger.warning(
                    "video scene field repair failure scene=%s missing_field=%s attempt=%s error=%s",
                    scene_number,
                    missing_field,
                    repair_attempts,
                    error,
                )
        else:
            raise ValueError(
                f"Scene-specific field repair failed after 2 attempts for Scene {scene_number} "
                f"missing {missing_field}. Last failure: {last_error}"
            ) from last_error

    updated_scenes[scene_number - 1] = target_scene
    updated_plan_data["scenes"] = updated_scenes
    return updated_plan_data


def repair_incomplete_storyboard_payload(
    *,
    idea: str,
    settings: VideoSettings,
    creative_authority: dict[str, Any],
    plan_data: dict[str, Any],
    scene_slots: list[SourceSceneAssignment],
    anchor_by_id: dict[str, RequiredSourceAnchor],
    style_lock: str,
) -> dict[str, Any]:
    slot_by_scene = {slot.scene_number: slot for slot in scene_slots}
    while True:
        try:
            validate_storyboard_payload(plan_data, settings=settings)
            return plan_data
        except StoryboardSceneFieldError as error:
            logger.warning(
                "video storyboard missing scene field scene=%s missing_field=%s",
                error.scene_number,
                error.missing_field,
            )
            plan_data = repair_or_complete_storyboard_scene(
                idea=idea,
                settings=settings,
                creative_authority=creative_authority,
                plan_data=plan_data,
                scene_slot=slot_by_scene[error.scene_number],
                anchor_by_id=anchor_by_id,
                missing_field=error.missing_field,
                style_lock=style_lock,
            )


def derive_video_plan_title(plan_data: dict[str, Any], idea: str, creative_result: CreativeResult | None = None) -> str:
    candidate = " ".join(str(plan_data.get("title", "")).split()).strip()
    if candidate:
        return candidate[:80]
    if creative_result is not None:
        sources = [
            creative_result.final_story.premise,
            creative_result.final_story.ending,
            creative_result.request.idea,
            idea,
        ]
    else:
        sources = [idea]
    for source in sources:
        cleaned = " ".join((source or "").split()).strip(" .")
        if cleaned:
            title = cleaned[:60].strip().rstrip(".")
            if title:
                return title.title()
    return "Storyboard"


def assemble_plan_narration(plan_data: dict[str, Any], scenes: list[VideoScene]) -> str:
    top_level = " ".join(str(plan_data.get("narration", "")).split()).strip()
    if top_level:
        return top_level
    return " ".join(scene.narration for scene in scenes).strip()


def construct_video_plan(
    *,
    plan_data: dict[str, Any],
    scenes: list[VideoScene],
    style_lock: str,
    settings: VideoSettings,
    idea: str,
    creative_result: CreativeResult | None = None,
) -> VideoPlan:
    if not style_lock:
        raise ValueError("Video plan style_lock is required.")

    normalized_scenes = [
        VideoScene(
            scene_number=scene.scene_number,
            duration_seconds=settings.scene_durations[scene.scene_number - 1],
            narration=scene.narration,
            visual_prompt=apply_style_lock_to_visual(style_lock, scene.visual_prompt),
            motion_prompt=scene.motion_prompt,
            continuity_mode=scene.continuity_mode,
            continuity_group=scene.continuity_group,
            scene_purpose=scene.scene_purpose,
            source_anchor_id=scene.source_anchor_id,
        )
        for scene in scenes
    ]
    plan = VideoPlan(
        title=derive_video_plan_title(plan_data, idea=idea, creative_result=creative_result),
        content_type=settings.content_type,
        duration_seconds=settings.total_duration_seconds,
        aspect_ratio=settings.aspect_ratio,
        narration=assemble_plan_narration(plan_data, normalized_scenes),
        style_lock=style_lock,
        scenes=normalized_scenes,
        settings=settings,
    )
    if not plan.narration:
        raise ValueError("Video plan narration is required after scene assembly.")
    if total_duration_seconds(plan.scenes) != settings.total_duration_seconds:
        raise ValueError("Scene durations must sum exactly to the selected total duration.")
    return plan


def build_fallback_style_lock(idea: str, settings: VideoSettings) -> str:
    del idea
    return build_default_style_lock(settings)


def build_fallback_plan(idea: str, settings: VideoSettings) -> VideoPlan:
    cleaned_idea = " ".join(idea.split()).strip() or "A simple visual story"
    title_base = cleaned_idea[:60].strip().rstrip(".")
    title = title_base.title() if title_base else "Storyboard MVP"
    style_lock = build_fallback_style_lock(cleaned_idea, settings)

    scenes: list[VideoScene] = []
    for index, duration_seconds in enumerate(settings.scene_durations, start=1):
        if index == 1:
            narration = f"Scene {index} opens the central idea: {cleaned_idea}."
            visual_prompt = "A concrete opening image that shows the main subject, place, and immediate situation."
        elif index == settings.scene_count:
            narration = f"Scene {index} lands the final turn and leaves a clear closing image."
            visual_prompt = "A concrete closing image that resolves the final turn without adding new symbols."
        else:
            narration = f"Scene {index} shows the next concrete change in the situation."
            visual_prompt = "A specific middle image showing one visible action, object, or environmental change."

        motion_prompt = "Subtle physical or camera motion that matches the visible action in this exact scene."
        scenes.append(
            VideoScene(
                scene_number=index,
                duration_seconds=duration_seconds,
                narration=narration,
                visual_prompt=apply_style_lock_to_visual(style_lock, visual_prompt),
                motion_prompt=motion_prompt,
                continuity_mode="independent",
                continuity_group=None,
                scene_purpose="conclusion" if index == settings.scene_count else "transition",
                source_anchor_id=None,
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
        used_fallback=True,
        fallback_reason="Storyboard generation failed before a valid plan was produced.",
    )


def build_creative_authority_payload(
    result: CreativeResult,
    settings: VideoSettings,
    required_source_anchors: list[RequiredSourceAnchor] | None = None,
    scene_slots: list[SourceSceneAssignment] | None = None,
) -> dict[str, Any]:
    source_metaphors = extract_explicit_source_metaphors(result.request.idea)
    required_source_anchors = required_source_anchors or build_required_source_anchors(source_metaphors)
    continuity_hints = infer_scene_continuity(result.final_story.scene_beats)
    scene_purposes = scene_slots or build_scene_slot_plan(required_source_anchors=required_source_anchors, scene_count=settings.scene_count)
    log_anchor_trace("creative_payload", required_source_anchors, scene_purposes)
    scene_purpose_by_number = {item.scene_number: item for item in scene_purposes}
    anchor_by_id = build_anchor_lookup(required_source_anchors)
    scene_authority = []
    for index in range(1, settings.scene_count + 1):
        beat = result.final_story.scene_beats[index - 1] if index - 1 < len(result.final_story.scene_beats) else ""
        continuity_mode, continuity_group = continuity_hints[index - 1] if index - 1 < len(continuity_hints) else ("independent", None)
        purpose = scene_purpose_by_number.get(
            index,
            SourceSceneAssignment(scene_number=index, source_anchor_id=None, source_anchor="", scene_purpose="transition"),
        )
        anchor = anchor_by_id.get(purpose.source_anchor_id or "")
        scene_authority.append(
            {
                "scene_number": index,
                "scene_beat": beat,
                "scene_purpose": purpose.scene_purpose,
                "source_anchor_id": purpose.source_anchor_id,
                "assigned_source_anchor": purpose.source_anchor or None,
                "story_anchor_id": purpose.source_anchor_id,
                "required_objects": list(anchor.required_objects) if anchor else [],
                "continuity_mode": continuity_mode,
                "continuity_group": continuity_group,
            }
        )

    specialist_conclusions: dict[str, Any] = {}
    if result.psychology:
        specialist_conclusions["psychology"] = {
            "visible_behavior": result.psychology.visible_behavior,
            "hidden_motive": result.psychology.hidden_motive,
            "emotional_trigger": result.psychology.emotional_trigger,
            "contradiction": result.psychology.contradiction,
        }
    if result.philosophy:
        specialist_conclusions["philosophy"] = {
            "central_question": result.philosophy.central_question,
            "source_meaning": result.philosophy.source_meaning,
            "modern_reflection": result.philosophy.modern_reflection,
            "deeper_meaning": result.philosophy.deeper_meaning,
            "preserve_source_metaphors": True,
        }
    if result.ambiguity:
        specialist_conclusions["ambiguity"] = {
            "unresolved_question": result.ambiguity.unresolved_question,
            "what_not_to_explain": result.ambiguity.what_not_to_explain,
            "competing_interpretations": result.ambiguity.competing_interpretations,
        }
    if result.humor:
        specialist_conclusions["humor"] = {
            "humor_style": result.humor.humor_style,
            "setup": result.humor.setup,
            "callback_candidate": result.humor.callback_candidate,
        }

    return {
        "global_style": {
            "visual_style": settings.visual_style,
            "style_lock_reference": build_default_style_lock(settings),
            "tone": result.request.tone,
            "content_type": result.request.content_type,
        },
        "final_story": {
            "premise": result.final_story.premise,
            "conflict": result.final_story.conflict,
            "progression": result.final_story.progression,
            "emotional_turn": result.final_story.emotional_turn,
            "ending": result.final_story.ending,
            "scene_beats": result.final_story.scene_beats,
        },
        "specialist_conclusions": specialist_conclusions,
        "continuity_requirements": {
            "scene_hints": scene_authority,
            "rules": [
                "Treat symbolic metaphor scenes as independent unless the scene beat explicitly says to continue a prior subject.",
                "Only assign a continuity_group when the same character or same world must stay identical across scenes.",
                "Do not force a recurring protagonist into symbolic scenes that work better as standalone images.",
            ],
        },
        "source_fidelity": {
            "is_source_based_reflection": is_source_based_reflection(result.request.idea, result.request.content_type),
            "explicit_source_metaphors": source_metaphors,
            "required_source_anchors": [
                {
                    "id": anchor.id,
                    "meaning": anchor.meaning,
                    "required_objects": list(anchor.required_objects),
                    "source_order": anchor.source_order,
                    "allowed_depictions": list(anchor.allowed_depictions),
                    "forbidden_replacements": list(anchor.forbidden_replacements),
                }
                for anchor in required_source_anchors
            ],
            "scene_assignments": [
                {
                    "scene_number": assignment.scene_number,
                    "scene_purpose": assignment.scene_purpose,
                    "source_anchor_id": assignment.source_anchor_id,
                    "story_anchor_id": assignment.source_anchor_id,
                    "source_anchor": assignment.source_anchor or None,
                }
                for assignment in scene_purposes
            ],
            "preserve_source_metaphors_first": True if source_metaphors else False,
            "modern_reflection_after_source_metaphors": True if source_metaphors else False,
            "third_metaphor_guidance": (
                "If the source metaphor is unborn life enclosed within the womb, preserve it respectfully and symbolically. "
                "Do not replace it with jars, seeds, cracks, or unrelated release imagery."
                if any("womb" in metaphor and "unborn" in metaphor for metaphor in source_metaphors)
                else ""
            ),
        },
        "critic_guidance": {
            "notes": result.critic.notes,
            "edit_instructions": result.critic.edit_instructions,
        },
    }


def build_planning_prompt(
    idea: str,
    settings: VideoSettings,
    creative_authority: dict[str, Any] | None = None,
) -> str:
    if creative_authority is not None:
        scene_examples = ",\n".join(
            (
                "    {\n"
                f'      "scene_number": {index},\n'
                '      "narration": "Scene narration",\n'
                '      "visual_prompt": "Image prompt",\n'
                '      "motion_prompt": "Actual physical or camera motion",\n'
                '      "scene_purpose": "transition",\n'
                '      "story_anchor_id": null,\n'
                '      "continuity_mode": "independent",\n'
                '      "continuity_group": null\n'
                "    }"
            )
            for index in range(1, settings.scene_count + 1)
        )
        schema = (
            "{\n"
            '  "title": "Short title",\n'
            '  "narration": "Optional full-video narration",\n'
            '  "scenes": [\n'
            f"{scene_examples}\n"
            "  ]\n"
            "}"
        )
    else:
        scene_examples = ",\n".join(
            (
                "    {\n"
                f'      "scene_number": {index},\n'
                f'      "duration_seconds": {duration_seconds},\n'
                '      "narration": "Scene narration",\n'
                '      "visual_prompt": "Image prompt",\n'
                '      "motion_prompt": "Actual physical or camera motion",\n'
                '      "scene_purpose": "transition",\n'
                '      "story_anchor_id": null,\n'
                '      "continuity_mode": "independent",\n'
                '      "continuity_group": null\n'
                "    }"
            )
            for index, duration_seconds in enumerate(settings.scene_durations, start=1)
        )
        schema = (
            "{\n"
            '  "title": "Short title",\n'
            f'  "content_type": "{settings.content_type}",\n'
            f'  "duration_seconds": {settings.total_duration_seconds},\n'
            f'  "aspect_ratio": "{settings.aspect_ratio}",\n'
            '  "narration": "Complete narration text for the full video",\n'
            '  "style_lock": "A shared visual style description that every scene prompt must reuse",\n'
            '  "scenes": [\n'
            f"{scene_examples}\n"
            "  ]\n"
            "}"
        )
    narration_target = narration_word_target(settings)
    if narration_target is not None:
        narration_guidance = f"- Keep the full narration around {narration_target[0]} to {narration_target[1]} words.\n"
    else:
        narration_guidance = (
            f"- Keep the narration natural for {settings.language} and short enough to finish about 0.5 seconds before the video ends.\n"
        )

    creative_section = ""
    if creative_authority is not None:
        creative_section = (
            "\nAuthoritative creative result:\n"
            f"{json.dumps(creative_authority, indent=2, ensure_ascii=True)}\n"
        )

    return f"""
You are a video storyboard planner for a Streamlit app.

Return only valid JSON with this exact schema:
{schema}

Rules:
- Create exactly {settings.scene_count} scenes.
- Every scene_number must be sequential starting at 1.
- Visual style should align with: {settings.visual_style}.
- Reuse this style lock base in every scene prompt: {build_default_style_lock(settings)}
- Separate three concepts clearly:
  1. global style = reusable aesthetics across the whole video
  2. scene content = what physically happens in this exact scene
  3. continuity = what must stay identical across selected scenes only
- If an authoritative creative result is provided, treat final_story.scene_beats as authoritative and preserve their order.
- If authoritative scene metadata marks a scene as source_metaphor, preserve its assigned source anchor in that scene before moving into interpretation.
- If required_source_anchors are present, do not invent the scene architecture from scratch. Fill the provided scene assignments in order.
- For every source_metaphor scene, copy the exact story_anchor_id from the assignment metadata.
- For every source_metaphor scene, explicitly include the required objects and preserve their relationship.
- Never replace a required source anchor with a different symbolic object, container, orb, cocoon, egg, flower, seed, jar, or abstract substitute.
- If story_anchor_id is anchor_3 for unborn life enclosed within the womb, the visual must show a respectful symbolic depiction of unborn life enclosed within a womb or maternal form. Do not use a seed, clay jar, cocoon, glowing orb, egg, flower, or generic container.
- Create exactly one concrete visual event per scene.
- Do not repeat the full user prompt.
- Do not invent recurring visual symbols unless the user requested them or the authoritative story explicitly chose them for a clear reason.
- Do not introduce a glowing orb, magical light, aura, floating symbol, or spiritual particle effect unless explicitly authorized.
- For philosophy, scripture, or supplied text, preserve the source metaphors before adding interpretation.
- Prefer literal visual metaphors from the source before generic spiritual imagery.
- For source-based reflective content, present the explicit source metaphors first and only then move into modern reflection.
- Do not substitute explicit source metaphors with poetic replacements such as seeds, jars, cracks, or invented release symbols unless the user explicitly asked for reinterpretation.
- If the source metaphor includes unborn life enclosed within the womb, present it respectfully and symbolically, not medically.
- Do not force a protagonist into symbolic scenes.
- continuity_mode must be one of: independent, character, world, previous_scene.
- continuity_group must be null for independent scenes and a short stable string for shared continuity scenes.
- Narration enabled: {"yes" if settings.narration_enabled else "no"}.
- Language: {settings.language}.
- Speaking style: {settings.speaking_style}.
{narration_guidance}- Use visual storytelling where possible.
- narration must sound like narration, not production instructions.
- visual_prompt must describe what the viewer sees.
- motion_prompt must describe actual physical motion or camera motion, not generic workflow language.
- Do not add markdown fences.
- Do not add explanatory text.
{creative_section}
Video idea: {idea}
"""


def request_video_plan_from_ollama(
    idea: str,
    settings: VideoSettings,
    creative_authority: dict[str, Any] | None = None,
) -> str:
    prompt = build_planning_prompt(idea=idea, settings=settings, creative_authority=creative_authority)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": idea},
        ],
    )
    return response["message"]["content"]


def request_scene_repair_from_ollama(
    *,
    idea: str,
    settings: VideoSettings,
    creative_authority: dict[str, Any],
    plan_data: dict[str, Any],
    scene_slot: SourceSceneAssignment,
    anchor: RequiredSourceAnchor,
    failure_reason: str,
) -> str:
    scene_number = scene_slot.scene_number
    current_scene = next(
        (scene for scene in plan_data.get("scenes", []) if int(scene.get("scene_number", 0)) == scene_number),
        None,
    )
    if current_scene is None:
        raise ValueError(f"Cannot repair Scene {scene_number}; source anchor metadata is incomplete.")

    logger.info(
        "video repair trace scene=%s purpose=%s anchor_id=%s anchor=%s order=%s",
        scene_slot.scene_number,
        scene_slot.scene_purpose,
        scene_slot.source_anchor_id,
        anchor.description,
        anchor.source_order,
    )

    prompt = (
        "You repair exactly one storyboard scene.\n"
        "Return only one valid JSON object for the rewritten scene with this schema:\n"
        "{\n"
        '  "scene_number": 3,\n'
        '  "narration": "Scene narration",\n'
        '  "visual_prompt": "Image prompt",\n'
        '  "motion_prompt": "Actual physical or camera motion",\n'
        '  "scene_purpose": "source_metaphor",\n'
        '  "story_anchor_id": "anchor_id",\n'
        '  "continuity_mode": "independent",\n'
        '  "continuity_group": null\n'
        "}\n"
        "Rules:\n"
        f"- Rewrite Scene {scene_number} only.\n"
        "- Keep every other scene unchanged.\n"
        f"- Keep scene_number exactly {scene_number}.\n"
        f"- Keep scene_purpose exactly {scene_slot.scene_purpose}.\n"
        f"- Keep story_anchor_id exactly {scene_slot.source_anchor_id}.\n"
        f"- Required source anchor meaning: {anchor.meaning}.\n"
        f"- Required objects: {', '.join(anchor.required_objects)}.\n"
        "- Preserve the source relationship literally and symbolically.\n"
        "- Do not substitute a different metaphor.\n"
        f"- Forbidden replacements: {', '.join(anchor.forbidden_replacements)}.\n"
        f"- Allowed depictions: {', '.join(anchor.allowed_depictions) or 'none specified'}.\n"
        "- narration must sound like narration, not instructions.\n"
        "- visual_prompt must describe what the viewer sees.\n"
        "- motion_prompt must describe actual physical or camera motion.\n"
        f"- Failure to repair: {failure_reason}\n"
        f"Current full storyboard:\n{json.dumps(plan_data, indent=2, ensure_ascii=True)}\n"
        f"Rewrite Scene {scene_number} only."
    )
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


def validate_plan_against_creative_result(
    plan: VideoPlan,
    idea: str,
    creative_result: CreativeResult,
    required_source_anchors: list[RequiredSourceAnchor],
    scene_slots: list[SourceSceneAssignment],
) -> VideoPlan:
    for beat in creative_result.final_story.scene_beats:
        if is_generic_scene_text(beat):
            raise ValueError(f"CreativeResult scene beat is too generic: {beat}")

    if required_source_anchors:
        anchor_by_id = build_anchor_lookup(required_source_anchors)
        assigned_source_scenes = [item for item in scene_slots if item.scene_purpose == "source_metaphor"]

        represented_anchors: set[str] = set()
        for assignment in assigned_source_scenes:
            scene = plan.scenes[assignment.scene_number - 1]
            expected_anchor_id = assignment.source_anchor_id
            anchor = anchor_by_id[expected_anchor_id or ""]
            logger.info(
                "video validating scene=%s purpose=%s anchor_id=%s anchor=%s order=%s",
                scene.scene_number,
                assignment.scene_purpose,
                expected_anchor_id,
                anchor.description,
                anchor.source_order,
            )
            if scene.scene_purpose != "source_metaphor" or scene.source_anchor_id != expected_anchor_id:
                raise SourceFidelityValidationError(
                    f"Scene {scene.scene_number} must remain a source_metaphor scene for '{anchor.description}'.",
                    scene_number=scene.scene_number,
                    source_anchor_id=expected_anchor_id,
                    source_anchor=anchor.description,
                    issue_type="metadata",
                )
            scene_text = f"{scene.narration} {scene.visual_prompt}"
            source_anchor = build_source_anchor(anchor.description)
            if not text_matches_source_anchor(scene_text, source_anchor):
                raise SourceFidelityValidationError(
                    f"Scene {scene.scene_number} was assigned to '{anchor.description}' but substituted unrelated imagery: "
                    f"{compact_scene_imagery(scene)}.",
                    scene_number=scene.scene_number,
                    source_anchor_id=expected_anchor_id,
                    source_anchor=anchor.description,
                    issue_type="drift",
                )
            represented_anchors.add(anchor.description)
            disallowed = [item for item in anchor.forbidden_replacements if item in scene_text.lower()]
            if disallowed:
                raise SourceFidelityValidationError(
                    f"Scene {scene.scene_number} was assigned to '{anchor.description}' but used forbidden replacements: {', '.join(disallowed)}.",
                    scene_number=scene.scene_number,
                    source_anchor_id=expected_anchor_id,
                    source_anchor=anchor.description,
                    issue_type="drift",
                )

        for required_anchor in required_source_anchors:
            if required_anchor.description not in represented_anchors:
                raise SourceFidelityValidationError(
                    f"Required source metaphor '{required_anchor.description}' was not faithfully represented in any source-metaphor scene.",
                    source_anchor_id=required_anchor.id,
                    source_anchor=required_anchor.description,
                    issue_type="missing",
                )

    for scene in plan.scenes:
        for symbol in FORBIDDEN_RECURRING_SYMBOLS:
            if symbol in scene.visual_prompt.lower() and not has_authorized_symbol(symbol, idea, creative_result):
                raise ValueError(f"Scene {scene.scene_number} introduced unauthorized recurring symbol: {symbol}")
        if " aura " in f" {scene.visual_prompt.lower()} " and not has_authorized_symbol("aura", idea, creative_result):
            raise ValueError(f"Scene {scene.scene_number} introduced unauthorized recurring symbol: aura")

    return plan


def build_video_plan(idea: str, settings: VideoSettings) -> tuple[VideoPlan, str | None]:
    try:
        raw_response = request_video_plan_from_ollama(idea=idea, settings=settings)
        plan_data = parse_plan_json(raw_response)
        plan_data["style_lock"] = sanitize_style_lock(
            str(plan_data.get("style_lock", "")),
            idea=idea,
            settings=settings,
        )
        plan = validate_video_plan_data(plan_data, settings=settings)
        return plan, None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        fallback_plan = build_fallback_plan(idea=idea, settings=settings)
        fallback_plan.fallback_reason = str(error)
        return fallback_plan, (
            "Storyboard generation failed and AumState used the generic fallback storyboard. "
            f"Reason: {error}"
        )


def repair_source_anchor_scene_in_plan(
    *,
    idea: str,
    settings: VideoSettings,
    creative_authority: dict[str, Any],
    plan_data: dict[str, Any],
    validation_error: SourceFidelityValidationError,
    scene_slot: SourceSceneAssignment,
    anchor: RequiredSourceAnchor,
) -> dict[str, Any]:
    if validation_error.scene_number is None or not validation_error.source_anchor_id:
        raise validation_error

    repaired_scene_raw = request_scene_repair_from_ollama(
        idea=idea,
        settings=settings,
        creative_authority=creative_authority,
        plan_data=plan_data,
        scene_slot=scene_slot,
        anchor=anchor,
        failure_reason=str(validation_error),
    )
    repaired_scene_data = parse_plan_json(repaired_scene_raw)
    if not isinstance(repaired_scene_data, dict):
        raise ValueError(f"Scene {validation_error.scene_number} repair did not return a JSON object.")

    expected_scene_number = validation_error.scene_number
    repaired_scene_data["scene_number"] = expected_scene_number
    repaired_scene_data["scene_purpose"] = scene_slot.scene_purpose
    repaired_scene_data["source_anchor_id"] = scene_slot.source_anchor_id
    repaired_scene_data["story_anchor_id"] = scene_slot.source_anchor_id

    updated_plan_data = dict(plan_data)
    updated_scenes = [dict(scene) for scene in plan_data.get("scenes", [])]
    if expected_scene_number - 1 >= len(updated_scenes):
        raise ValueError(f"Scene {expected_scene_number} repair target does not exist in the storyboard.")
    updated_scenes[expected_scene_number - 1] = repaired_scene_data
    updated_plan_data["scenes"] = updated_scenes
    return updated_plan_data


def build_video_plan_from_creative_result(
    idea: str,
    creative_result: CreativeResult,
    settings: VideoSettings,
) -> tuple[VideoPlan, str | None]:
    source_metaphors = extract_explicit_source_metaphors(creative_result.request.idea)
    required_source_anchors = build_required_source_anchors(source_metaphors)
    scene_slots = build_scene_slot_plan(required_source_anchors=required_source_anchors, scene_count=settings.scene_count)
    assert not required_source_anchors or scene_slots[0].source_anchor_id == "anchor_1"
    anchor_by_id = build_anchor_lookup(required_source_anchors)
    if "anchor_1" in anchor_by_id:
        assert anchor_by_id["anchor_1"].description == "smoke covering fire"
    logger.info("video anchor trace stage=request_extraction metaphors=%s", source_metaphors)
    log_anchor_trace("request_contract", required_source_anchors, scene_slots)
    creative_authority = build_creative_authority_payload(
        creative_result,
        settings,
        required_source_anchors=required_source_anchors,
        scene_slots=scene_slots,
    )
    try:
        style_lock = sanitize_style_lock(
            build_default_style_lock(settings),
            idea=idea,
            settings=settings,
            creative_result=creative_result,
        )
        plan_data: dict[str, Any] | None = None
        last_storyboard_error: Exception | None = None
        for attempt in range(2):
            try:
                raw_response = request_video_plan_from_ollama(
                    idea=idea,
                    settings=settings,
                    creative_authority=creative_authority,
                )
                plan_data = parse_plan_json(raw_response)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                last_storyboard_error = error
                logger.warning("video storyboard parse failure attempt=%s error=%s", attempt + 1, error)
                if attempt >= 1:
                    raise
                continue
            try:
                plan_data = merge_model_storyboard_with_scene_slots(plan_data, scene_slots)
                logger.info("video anchor trace stage=parsed_storyboard scenes=%s", json.dumps(plan_data.get("scenes", []), ensure_ascii=True))
                plan_data = repair_incomplete_storyboard_payload(
                    idea=idea,
                    settings=settings,
                    creative_authority=creative_authority,
                    plan_data=plan_data,
                    style_lock=style_lock,
                    scene_slots=scene_slots,
                    anchor_by_id=anchor_by_id,
                )
                validate_storyboard_payload(plan_data, settings=settings)
                break
            except ValueError as error:
                last_storyboard_error = error
                logger.warning("video storyboard creative payload failure attempt=%s error=%s", attempt + 1, error)
                if attempt >= 1:
                    raise
        if plan_data is None:
            if last_storyboard_error is not None:
                raise last_storyboard_error
            raise ValueError("Storyboard generation did not return a payload.")
        repair_attempts = 0
        for _attempt in range(3):
            try:
                plan_data = repair_incomplete_storyboard_payload(
                    idea=idea,
                    settings=settings,
                    creative_authority=creative_authority,
                    plan_data=plan_data,
                    style_lock=style_lock,
                    scene_slots=scene_slots,
                    anchor_by_id=anchor_by_id,
                )
                scenes = validate_storyboard_payload(plan_data, settings=settings)
            except ValueError as error:
                logger.warning("video storyboard creative payload failure during repair_loop error=%s", error)
                raise
            try:
                plan = construct_video_plan(
                    plan_data=plan_data,
                    scenes=scenes,
                    style_lock=style_lock,
                    settings=settings,
                    idea=idea,
                    creative_result=creative_result,
                )
            except ValueError as error:
                logger.warning("video final plan construction failure error=%s", error)
                raise
            try:
                plan = validate_plan_against_creative_result(
                    plan,
                    idea=idea,
                    creative_result=creative_result,
                    required_source_anchors=required_source_anchors,
                    scene_slots=scene_slots,
                )
                return plan, None
            except SourceFidelityValidationError as error:
                logger.warning(
                    "video source fidelity failure scene=%s anchor_id=%s error=%s",
                    error.scene_number,
                    error.source_anchor_id,
                    error,
                )
                if error.scene_number is None or not error.source_anchor_id:
                    raise
                if repair_attempts >= 2:
                    raise ValueError(
                        f"Scene-specific repair failed after 2 attempts for Scene {error.scene_number} "
                        f"and source anchor '{error.source_anchor}'. Last failure: {error}"
                    ) from error
                scene_slot = scene_slots[error.scene_number - 1]
                plan_data = repair_source_anchor_scene_in_plan(
                    idea=idea,
                    settings=settings,
                    creative_authority=creative_authority,
                    plan_data=plan_data,
                    validation_error=error,
                    scene_slot=scene_slot,
                    anchor=anchor_by_id[scene_slot.source_anchor_id or ""],
                )
                plan_data = merge_model_storyboard_with_scene_slots(plan_data, scene_slots)
                repair_attempts += 1
        return plan, None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Storyboard generation failed after Multi-Mind creative synthesis. "
            f"Retry storyboard generation instead of using a generic fallback. Failure reason: {error}"
        ) from error
