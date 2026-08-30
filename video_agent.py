from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import ollama

from creative_models import CreativeResult
from error_utils import format_user_error


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
    "opening frame establishing",
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
    "that",
    "this",
    "with",
    "from",
    "into",
    "over",
    "under",
}
CONTRADICTION_SPLIT_PATTERNS = (r"\bbut\b", r"\bwhile\b", r"\byet\b", r"\balthough\b")
VALID_CONTINUITY_MODES = {"independent", "character", "world", "previous_scene"}


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
class NarrativeConstraint:
    id: str
    constraint_type: str
    description: str
    importance: str = "required"
    source_order: int | None = None


@dataclass(frozen=True)
class SourceAnchor:
    canonical_text: str
    concept_groups: tuple[tuple[str, ...], ...]


class StoryboardGenerationError(ValueError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def normalize_content_type(content_type: str) -> str:
    cleaned = (content_type or "").strip()
    if not cleaned:
        return "story"
    lowered = cleaned.lower().replace("-", "_").replace(" ", "_")
    if lowered in CONTENT_TYPE_OPTIONS:
        return lowered
    if cleaned in CONTENT_TYPE_LABEL_TO_VALUE:
        return CONTENT_TYPE_LABEL_TO_VALUE[cleaned]
    for value, label in CONTENT_TYPE_LABELS.items():
        if lowered == label.lower().replace(" ", "_"):
            return value
    return "story"


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
    content_type: str = "story",
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
    scene_durations = build_scene_durations(normalized_total_duration, normalized_scene_duration)

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
    normalized_content_type = normalize_content_type(content_type)
    return (
        any(marker in lowered for marker in SOURCE_TEXT_MARKERS) or bool(extract_explicit_source_metaphors(idea))
    ) and normalized_content_type in {"philosophy", "spiritual_reflection"}


def metaphor_keywords(metaphor: str) -> list[str]:
    words = re.findall(r"[a-z]+", metaphor.lower())
    keywords = [word for word in words if word not in SOURCE_STOPWORDS and len(word) > 2]
    return list(dict.fromkeys(keywords))


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


def build_default_style_lock(settings: VideoSettings) -> str:
    style_templates = {
        "Quiet Cinematic Animation": (
            "Quiet cinematic animation, realistic anatomy, tactile natural textures, soft directional light, "
            "muted earthy palette, composed framing, gentle lens movement, no neon fantasy styling."
        ),
        "Animated Realism": (
            "Animated realism, believable proportions, skin and fabric detail, natural material response, "
            "soft contrast, grounded palette, cinematic framing, restrained camera language, no cartoon exaggeration."
        ),
        "Hand-Painted Storybook": (
            "Hand-painted storybook aesthetic, expressive brush texture, graceful anatomy, warm paper-like surfaces, "
            "diffused light, restrained palette, lyrical composition, still camera, no glossy digital sheen."
        ),
        "Minimal Illustration": (
            "Minimal illustration, clean anatomy, flat textured color fields, selective detail, controlled contrast, "
            "disciplined palette, spacious composition, graphic camera framing, no cluttered ornament."
        ),
        "Soft 3D Animation": (
            "Soft 3D animation, rounded but believable forms, tactile materials, diffused lighting, balanced palette, "
            "clear depth staging, calm camera language, no plastic over-shine or hyperactive motion."
        ),
        "3D Nursery Animation": (
            "3D nursery animation, rounded child-safe forms, soft plush textures, bright but gentle daylight, "
            "friendly palette, tidy composition, simple camera language, no harsh shadows or eerie surrealism."
        ),
    }
    return style_templates.get(
        settings.visual_style,
        "Cinematic animation, believable anatomy, coherent materials, soft natural light, controlled palette, clear composition, restrained camera language.",
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
        if creative_result.humor:
            haystacks.extend(
                [
                    creative_result.humor.setup.lower(),
                    creative_result.humor.callback_candidate.lower(),
                    " ".join(creative_result.humor.punchline_candidates).lower(),
                ]
            )
    return any(symbol in haystack for haystack in haystacks)


def extract_narrative_constraints(
    idea: str,
    creative_result: CreativeResult | None = None,
) -> list[NarrativeConstraint]:
    constraints: list[NarrativeConstraint] = []

    for index, metaphor in enumerate(extract_explicit_source_metaphors(idea), start=1):
        constraints.append(
            NarrativeConstraint(
                id=f"constraint_{index}",
                constraint_type="source_metaphor",
                description=build_source_anchor(metaphor).canonical_text,
                importance="required",
                source_order=index,
            )
        )

    if creative_result is not None:
        if creative_result.psychology and creative_result.psychology.contradiction.strip():
            constraints.append(
                NarrativeConstraint(
                    id=f"constraint_{len(constraints) + 1}",
                    constraint_type="contradiction",
                    description=creative_result.psychology.contradiction.strip(),
                    importance="required",
                )
            )

        specialist_people = " ".join(creative_result.final_story.scene_beats).lower()
        if "couple" in specialist_people:
            constraints.append(
                NarrativeConstraint(
                    id=f"constraint_{len(constraints) + 1}",
                    constraint_type="required_character",
                    description="A couple must appear as recurring human subjects where the story calls for them.",
                    importance="required",
                )
            )

    return constraints


def build_creative_authority_payload(
    result: CreativeResult,
    settings: VideoSettings,
    narrative_constraints: list[NarrativeConstraint] | None = None,
) -> dict[str, Any]:
    narrative_constraints = narrative_constraints or extract_narrative_constraints(result.request.idea, result)
    continuity_hints = infer_scene_continuity(result.final_story.scene_beats)

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
            "avoid_preaching": result.philosophy.avoid_preaching,
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
            "awkward_truth": result.humor.awkward_truth,
        }

    scene_hints = []
    for index in range(1, settings.scene_count + 1):
        beat = result.final_story.scene_beats[index - 1] if index - 1 < len(result.final_story.scene_beats) else ""
        continuity_mode, continuity_group = continuity_hints[index - 1] if index - 1 < len(continuity_hints) else ("independent", None)
        scene_hints.append(
            {
                "scene_number": index,
                "scene_beat": beat,
                "suggested_continuity_mode": continuity_mode,
                "suggested_continuity_group": continuity_group,
            }
        )

    return {
        "request": {
            "idea": result.request.idea,
            "content_type": settings.content_type,
            "tone": result.request.tone,
            "target_audience": result.request.target_audience,
            "language": result.request.language,
            "duration_seconds": settings.total_duration_seconds,
            "scene_count": settings.scene_count,
            "scene_durations": settings.scene_durations,
            "visual_style": settings.visual_style,
            "style_lock": build_default_style_lock(settings),
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
        "narrative_constraints": [
            {
                "id": constraint.id,
                "constraint_type": constraint.constraint_type,
                "description": constraint.description,
                "importance": constraint.importance,
                "source_order": constraint.source_order,
            }
            for constraint in narrative_constraints
        ],
        "continuity_hints": scene_hints,
        "source_fidelity": {
            "is_source_based_reflection": is_source_based_reflection(result.request.idea, result.request.content_type),
            "explicit_source_metaphors": extract_explicit_source_metaphors(result.request.idea),
            "preserve_source_meaning_before_modern_interpretation": True,
            "do_not_fake_quotes": True,
            "do_not_replace_concrete_source_imagery_with_generic_spiritual_symbolism": True,
        },
        "critic_guidance": {
            "notes": result.critic.notes,
            "edit_instructions": result.critic.edit_instructions,
        },
    }


def build_storyboard_schema(settings: VideoSettings) -> str:
    scene_examples = ",\n".join(
        (
            "    {\n"
            f'      "scene_number": {index},\n'
            '      "narration": "Scene narration",\n'
            '      "visual_prompt": "Image prompt",\n'
            '      "motion_prompt": "Actual physical or camera motion",\n'
            '      "scene_purpose": "What this scene does in the story",\n'
            '      "continuity_mode": "independent",\n'
            '      "continuity_group": null\n'
            "    }"
        )
        for index in range(1, settings.scene_count + 1)
    )
    return (
        "{\n"
        '  "title": "Short title",\n'
        '  "narration": "Optional full-video narration",\n'
        '  "scenes": [\n'
        f"{scene_examples}\n"
        "  ]\n"
        "}"
    )


def build_planning_prompt(
    idea: str,
    settings: VideoSettings,
    creative_authority: dict[str, Any] | None = None,
) -> str:
    narration_target = narration_word_target(settings)
    narration_guidance = (
        f"- Keep the full narration around {narration_target[0]} to {narration_target[1]} words.\n"
        if narration_target is not None
        else f"- Keep the narration natural for {settings.language} and short enough to finish before the video ends.\n"
    )
    style_lock = build_default_style_lock(settings)
    creative_section = ""
    if creative_authority is not None:
        creative_section = (
            "\nAuthoritative creative result:\n"
            f"{json.dumps(creative_authority, indent=2, ensure_ascii=True)}\n"
        )

    return f"""
You are a storyboard writer for a short-form video system.

Return only valid JSON with this exact schema:
{build_storyboard_schema(settings)}

Rules:
- Create exactly {settings.scene_count} scenes.
- Keep scene_number sequential from 1 to {settings.scene_count}.
- Use the authoritative final_story as the story source of truth when provided.
- Narrative constraints are storyboard-level requirements unless the user explicitly numbered scenes.
- Preserve the parts of the story that already work instead of rebuilding from scratch.
- style_lock is aesthetic only. Do not hide story content inside style_lock.
- Use this shared aesthetic direction across all scene visuals: {style_lock}
- Each scene needs one concrete narration line, one concrete visual prompt, and one concrete motion prompt.
- scene_purpose must state the role of the scene in plain language.
- continuity_mode must be one of: independent, character, world, previous_scene.
- continuity_group must be null for independent scenes and a short stable string otherwise.
- Let the storyboard choose continuity where it genuinely helps. Do not force continuity into symbolic scenes.
- Do not return content_type, duration_seconds, aspect_ratio, frame_rate, output size, settings, source_anchor_id, or story_anchor_id.
- Do not invent recurring symbolic motifs unless the user requested them or the authoritative story clearly chose them.
- Do not introduce glowing orb, magical light, floating symbol, or spiritual particle effect unless explicitly justified by the request or specialist insight.
- For source-based reflective content, present the source imagery faithfully before modern interpretation.
- Preserve source meaning and explicit metaphors without fake quotations or generic spiritual substitution.
- narration must sound like narration, not production instructions.
- visual_prompt must describe what the viewer sees.
- motion_prompt must describe actual physical motion or camera motion.
- Do not add markdown fences or commentary.
{narration_guidance}{creative_section}
Video idea: {idea}
""".strip()


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


def request_storyboard_revision_from_ollama(
    *,
    idea: str,
    settings: VideoSettings,
    creative_authority: dict[str, Any],
    original_storyboard: dict[str, Any] | None,
    original_raw_response: str,
    validation_issues: list[str],
    style_lock: str,
) -> str:
    prompt = (
        "You are revising a storyboard exactly once.\n"
        "Return only valid JSON using this schema:\n"
        f"{build_storyboard_schema(settings)}\n"
        "Preserve everything that already works. Correct only the identified problems.\n"
        "Do not return application state fields.\n"
        f"Shared style_lock:\n{style_lock}\n"
        f"Validation issues:\n{json.dumps(validation_issues, indent=2, ensure_ascii=True)}\n"
        f"Original parsed storyboard:\n{json.dumps(original_storyboard, indent=2, ensure_ascii=True) if original_storyboard is not None else 'null'}\n"
        f"Original raw storyboard response:\n{original_raw_response}\n"
        f"Authoritative creative result:\n{json.dumps(creative_authority, indent=2, ensure_ascii=True)}\n"
        f"Original user request: {idea}\n"
    )
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": idea},
        ],
    )
    return response["message"]["content"]


def collect_structural_issues(plan_data: dict[str, Any] | None, settings: VideoSettings) -> list[str]:
    if not isinstance(plan_data, dict):
        return ["Storyboard payload must be a JSON object."]

    raw_scenes = plan_data.get("scenes")
    if raw_scenes is None:
        return ["Storyboard payload must include scenes."]
    if not isinstance(raw_scenes, list):
        return ["Storyboard payload scenes must be a list."]
    if len(raw_scenes) != settings.scene_count:
        return [f"Storyboard payload must contain exactly {settings.scene_count} scenes."]

    issues: list[str] = []
    for index, scene_data in enumerate(raw_scenes, start=1):
        if not isinstance(scene_data, dict):
            issues.append(f"Scene {index} must be an object.")
            continue
        scene_number = int(scene_data.get("scene_number", index))
        if scene_number != index:
            issues.append(f"Scene {index} must use scene_number {index}.")

        narration = str(scene_data.get("narration", "")).strip()
        visual_prompt = str(scene_data.get("visual_prompt", "")).strip()
        motion_prompt = str(scene_data.get("motion_prompt", "")).strip()
        scene_purpose = str(scene_data.get("scene_purpose", "")).strip()
        continuity_mode = str(scene_data.get("continuity_mode", "independent")).strip() or "independent"
        continuity_group_raw = scene_data.get("continuity_group")
        continuity_group = str(continuity_group_raw).strip() if continuity_group_raw is not None else None

        if not narration:
            issues.append(f"Scene {index} narration is required.")
        if not visual_prompt:
            issues.append(f"Scene {index} visual_prompt is required.")
        if not motion_prompt:
            issues.append(f"Scene {index} motion_prompt is required.")
        if not scene_purpose:
            issues.append(f"Scene {index} scene_purpose is required.")
        if continuity_mode not in VALID_CONTINUITY_MODES:
            issues.append(f"Scene {index} continuity_mode is invalid: {continuity_mode}.")
        elif continuity_mode != "independent" and not continuity_group:
            issues.append(f"Scene {index} continuity_group is required for continuity_mode {continuity_mode}.")
        if narration and is_generic_scene_text(narration):
            issues.append(f"Scene {index} narration is too generic.")
        if visual_prompt and is_generic_scene_text(visual_prompt):
            issues.append(f"Scene {index} visual_prompt is too generic.")
        if motion_prompt and is_generic_scene_text(motion_prompt):
            issues.append(f"Scene {index} motion_prompt is too generic.")

    return issues


def materialize_storyboard_scenes(plan_data: dict[str, Any], settings: VideoSettings) -> list[VideoScene]:
    scenes: list[VideoScene] = []
    for index, scene_data in enumerate(plan_data["scenes"], start=1):
        continuity_mode = str(scene_data.get("continuity_mode", "independent")).strip() or "independent"
        continuity_group_raw = scene_data.get("continuity_group")
        continuity_group = str(continuity_group_raw).strip() if continuity_group_raw is not None else None
        if continuity_mode == "independent":
            continuity_group = None
        scenes.append(
            VideoScene(
                scene_number=index,
                duration_seconds=settings.scene_durations[index - 1],
                narration=str(scene_data["narration"]).strip(),
                visual_prompt=str(scene_data["visual_prompt"]).strip(),
                motion_prompt=str(scene_data["motion_prompt"]).strip(),
                continuity_mode=continuity_mode,
                continuity_group=continuity_group,
                scene_purpose=str(scene_data["scene_purpose"]).strip(),
            )
        )
    return scenes


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
        expected_duration = settings.scene_durations[index - 1]

        if scene_number != index:
            raise ValueError(f"Scene {index} must use scene_number {index}.")
        if duration_seconds != expected_duration:
            raise ValueError(f"Scene {index} duration must be {expected_duration} seconds, got {duration_seconds}.")
        if duration_seconds <= 0:
            raise ValueError(f"Scene {index} duration must be positive.")
        if not narration:
            raise ValueError(f"Scene {index} narration is required.")
        if not visual_prompt:
            raise ValueError(f"Scene {index} visual prompt is required.")
        if not motion_prompt:
            raise ValueError(f"Scene {index} motion prompt is required.")
        if continuity_mode not in VALID_CONTINUITY_MODES:
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


def contradiction_halves(description: str) -> tuple[list[str], list[str]] | None:
    cleaned = " ".join(description.lower().split())
    for pattern in CONTRADICTION_SPLIT_PATTERNS:
        parts = re.split(pattern, cleaned, maxsplit=1)
        if len(parts) == 2:
            left = metaphor_keywords(parts[0])
            right = metaphor_keywords(parts[1])
            if left and right:
                return left, right
    return None


def scenes_text(scenes: list[VideoScene]) -> str:
    return " ".join(
        f"{scene.narration} {scene.visual_prompt} {scene.motion_prompt}" for scene in scenes
    ).lower()


def keyword_signal_present(text: str, keyword: str) -> bool:
    if keyword in text:
        return True
    if len(keyword) >= 5 and keyword[:5] in text:
        return True
    if keyword.endswith("s") and keyword[:-1] in text:
        return True
    if keyword.endswith("ing") and keyword[:-3] in text:
        return True
    if keyword.endswith("ed") and keyword[:-2] in text:
        return True
    return False


def collect_constraint_issues(
    scenes: list[VideoScene],
    narrative_constraints: list[NarrativeConstraint],
    idea: str,
    creative_result: CreativeResult,
) -> list[str]:
    storyboard_text = scenes_text(scenes)
    issues: list[str] = []

    for constraint in narrative_constraints:
        if constraint.importance != "required":
            continue

        if constraint.constraint_type == "source_metaphor":
            anchor = build_source_anchor(constraint.description)
            if not text_matches_source_anchor(storyboard_text, anchor):
                issues.append(
                    f"Missing required source metaphor: {constraint.description}. Preserve the concrete source relationship somewhere in the storyboard."
                )
            continue

        if constraint.constraint_type == "contradiction":
            halves = contradiction_halves(constraint.description)
            if halves is None:
                continue
            left_keywords, right_keywords = halves
            left_present = any(keyword_signal_present(storyboard_text, keyword) for keyword in left_keywords[:4])
            right_present = any(keyword_signal_present(storyboard_text, keyword) for keyword in right_keywords[:5])
            if not (left_present and right_present):
                issues.append(
                    f"Missing required contradiction from the creative result: {constraint.description}."
                )
            continue

        if constraint.constraint_type == "required_character":
            if "couple" in constraint.description.lower() and "couple" not in storyboard_text:
                issues.append("Missing required recurring couple in the storyboard.")

    for scene in scenes:
        lowered_visual = scene.visual_prompt.lower()
        for symbol in FORBIDDEN_RECURRING_SYMBOLS:
            if symbol in lowered_visual and not has_authorized_symbol(symbol, idea, creative_result):
                issues.append(f"Scene {scene.scene_number} introduced unauthorized recurring symbol: {symbol}.")
        if " aura " in f" {lowered_visual} " and not has_authorized_symbol("aura", idea, creative_result):
            issues.append(f"Scene {scene.scene_number} introduced unauthorized recurring symbol: aura.")

    return issues


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


def validate_storyboard_candidate(
    *,
    plan_data: dict[str, Any] | None,
    settings: VideoSettings,
    style_lock: str,
    idea: str,
    creative_result: CreativeResult,
    narrative_constraints: list[NarrativeConstraint],
) -> tuple[VideoPlan | None, list[str]]:
    structural_issues = collect_structural_issues(plan_data, settings)
    if structural_issues:
        return None, structural_issues

    assert plan_data is not None
    scenes = materialize_storyboard_scenes(plan_data, settings)
    constraint_issues = collect_constraint_issues(scenes, narrative_constraints, idea, creative_result)
    if constraint_issues:
        return None, constraint_issues

    plan = construct_video_plan(
        plan_data=plan_data,
        scenes=scenes,
        style_lock=style_lock,
        settings=settings,
        idea=idea,
        creative_result=creative_result,
    )
    return plan, []


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
        logger.exception(
            "storyboard_generation_failed stage=standard_generation mode=%s content_type=%s scene_count=%s multi_mind=%s",
            settings.video_mode,
            settings.content_type,
            settings.scene_count,
            False,
        )
        fallback_plan = build_fallback_plan(idea=idea, settings=settings)
        fallback_plan.fallback_reason = format_user_error(error, "Storyboard generation failed")
        return fallback_plan, (
            "Storyboard generation failed and AumState used the generic fallback storyboard. "
            f"Reason: {format_user_error(error, 'Storyboard generation failed')}"
        )


def build_video_plan_from_creative_result(
    idea: str,
    creative_result: CreativeResult,
    settings: VideoSettings,
) -> tuple[VideoPlan, str | None]:
    narrative_constraints = extract_narrative_constraints(creative_result.request.idea, creative_result)
    creative_authority = build_creative_authority_payload(
        creative_result,
        settings,
        narrative_constraints=narrative_constraints,
    )
    style_lock = sanitize_style_lock(
        build_default_style_lock(settings),
        idea=idea,
        settings=settings,
        creative_result=creative_result,
    )

    try:
        initial_raw = request_video_plan_from_ollama(
            idea=idea,
            settings=settings,
            creative_authority=creative_authority,
        )
    except Exception as error:
        logger.exception(
            "storyboard_generation_failed stage=generation mode=%s content_type=%s scene_count=%s multi_mind=%s",
            settings.video_mode,
            settings.content_type,
            settings.scene_count,
            True,
        )
        raise StoryboardGenerationError(
            "generation",
            f"Storyboard generation failed during initial generation: {format_user_error(error, 'Generation error')}",
        ) from error

    initial_plan_data: dict[str, Any] | None = None
    initial_issues: list[str]
    try:
        initial_plan_data = parse_plan_json(initial_raw)
        plan, initial_issues = validate_storyboard_candidate(
            plan_data=initial_plan_data,
            settings=settings,
            style_lock=style_lock,
            idea=idea,
            creative_result=creative_result,
            narrative_constraints=narrative_constraints,
        )
        if plan is not None:
            return plan, None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        initial_issues = [f"Storyboard generation failed during JSON parsing: {format_user_error(error, 'Parsing error')}"]

    try:
        revised_raw = request_storyboard_revision_from_ollama(
            idea=idea,
            settings=settings,
            creative_authority=creative_authority,
            original_storyboard=initial_plan_data,
            original_raw_response=initial_raw,
            validation_issues=initial_issues,
            style_lock=style_lock,
        )
    except Exception as error:
        logger.exception(
            "storyboard_generation_failed stage=revision mode=%s content_type=%s scene_count=%s multi_mind=%s",
            settings.video_mode,
            settings.content_type,
            settings.scene_count,
            True,
        )
        raise StoryboardGenerationError(
            "revision",
            f"Storyboard revision failed: {format_user_error(error, 'Revision error')}",
        ) from error

    try:
        revised_plan_data = parse_plan_json(revised_raw)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        logger.exception(
            "storyboard_generation_failed stage=revision_parsing mode=%s content_type=%s scene_count=%s multi_mind=%s",
            settings.video_mode,
            settings.content_type,
            settings.scene_count,
            True,
        )
        raise StoryboardGenerationError(
            "revision_parsing",
            "Storyboard revision failed during JSON parsing. "
            f"Multi-Mind did not fall back to a generic storyboard. Failure reason: {format_user_error(error, 'Revision parsing error')}",
        ) from error

    try:
        plan, final_issues = validate_storyboard_candidate(
            plan_data=revised_plan_data,
            settings=settings,
            style_lock=style_lock,
            idea=idea,
            creative_result=creative_result,
            narrative_constraints=narrative_constraints,
        )
    except ValueError as error:
        logger.exception(
            "storyboard_generation_failed stage=video_plan_construction mode=%s content_type=%s scene_count=%s multi_mind=%s",
            settings.video_mode,
            settings.content_type,
            settings.scene_count,
            True,
        )
        raise StoryboardGenerationError(
            "video_plan_construction",
            "VideoPlan construction failed. "
            f"Multi-Mind did not fall back to a generic storyboard. Failure reason: {format_user_error(error, 'VideoPlan construction error')}",
        ) from error
    if plan is not None:
        return plan, None

    stage = "validation" if any("validation" in issue.lower() for issue in final_issues) else "revision_validation"
    message = " | ".join(final_issues) if final_issues else "No validation details were returned."
    logger.error(
        "storyboard_generation_failed stage=%s mode=%s content_type=%s scene_count=%s multi_mind=%s",
        stage,
        settings.video_mode,
        settings.content_type,
        settings.scene_count,
        True,
    )
    raise StoryboardGenerationError(
        stage,
        "Storyboard revision failed validation. "
        f"Multi-Mind did not fall back to a generic storyboard. Failure reason: {message}",
    )
