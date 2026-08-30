from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ROLE_DIRECTOR = "director"
ROLE_PSYCHOLOGY = "psychology"
ROLE_PHILOSOPHY = "philosophy"
ROLE_AMBIGUITY = "ambiguity"
ROLE_HUMOR = "humor"
ROLE_STORY = "story"
ROLE_CRITIC = "critic"

CREATIVE_ROLE_ORDER = [
    ROLE_DIRECTOR,
    ROLE_PSYCHOLOGY,
    ROLE_PHILOSOPHY,
    ROLE_AMBIGUITY,
    ROLE_HUMOR,
    ROLE_STORY,
    ROLE_CRITIC,
]


@dataclass
class CreativeRequest:
    idea: str
    content_type: str
    tone: str
    target_audience: str
    language: str
    duration_seconds: int
    visual_style: str
    humor_level: str = "off"
    depth_level: str = "medium"
    ambiguity_level: str = "balanced"


@dataclass
class DirectorDecision:
    content_intent: str
    emotional_tone: str
    narrative_shape: str
    use_psychology: bool
    use_philosophy: bool
    use_humor: bool
    use_ambiguity: bool
    humor_level: str
    philosophy_level: str
    psychology_level: str
    ambiguity_level: str
    story_focus: str
    rationale: str

    def selected_specialists(self) -> list[str]:
        selected: list[str] = []
        if self.use_psychology:
            selected.append(ROLE_PSYCHOLOGY)
        if self.use_philosophy:
            selected.append(ROLE_PHILOSOPHY)
        if self.use_ambiguity:
            selected.append(ROLE_AMBIGUITY)
        if self.use_humor:
            selected.append(ROLE_HUMOR)
        selected.extend([ROLE_STORY, ROLE_CRITIC])
        return selected


@dataclass
class PsychologyInsight:
    visible_behavior: str
    hidden_motive: str
    emotional_trigger: str
    contradiction: str
    audience_rel_path: str


@dataclass
class PhilosophyInsight:
    central_question: str
    deeper_meaning: str
    tension: str
    possible_closing_thought: str
    avoid_preaching: str
    source_meaning: str = ""
    modern_reflection: str = ""


@dataclass
class HumorInsight:
    humor_style: str
    setup: str
    punchline_candidates: list[str]
    callback_candidate: str
    awkward_truth: str
    avoid_cruelty: str


@dataclass
class AmbiguityInsight:
    competing_interpretations: list[str]
    unresolved_question: str
    contradiction: str
    what_not_to_explain: str
    ambiguity_strength: str


@dataclass
class NarrativeConstraint:
    id: str
    constraint_type: str
    description: str
    importance: str = "required"
    source_order: int | None = None


@dataclass
class StoryDraft:
    premise: str
    conflict: str
    progression: str
    emotional_turn: str
    ending: str
    scene_beats: list[str]


@dataclass
class CreativeEvaluation:
    relatability_score: int
    humor_score: int
    psychological_truth_score: int
    philosophical_depth_score: int
    ambiguity_score: int
    preachiness_score: int
    originality_score: int
    clarity_score: int
    forced_humor_score: int
    unnecessary_explanation_score: int
    notes: str
    edit_instructions: list[str]


@dataclass
class RoleExecution:
    role: str
    model_name: str
    duration_seconds: float
    status: str
    warning: str = ""


@dataclass
class CreativeResult:
    request: CreativeRequest
    director: DirectorDecision
    psychology: PsychologyInsight | None
    philosophy: PhilosophyInsight | None
    humor: HumorInsight | None
    ambiguity: AmbiguityInsight | None
    story: StoryDraft
    critic: CreativeEvaluation
    final_story: StoryDraft
    selected_specialists: list[str]
    narrative_constraints: list[NarrativeConstraint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    execution_log: list[RoleExecution] = field(default_factory=list)
    creative_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class CreativeModelConfig:
    director_model: str = "qwen3:8b"
    psychology_model: str = "qwen3:8b"
    philosophy_model: str = "qwen3:8b"
    ambiguity_model: str = "qwen3:8b"
    humor_model: str = "qwen3:8b"
    story_model: str = "qwen3:8b"
    critic_model: str = "qwen3:8b"

    def model_for_role(self, role: str) -> str:
        mapping = {
            ROLE_DIRECTOR: self.director_model,
            ROLE_PSYCHOLOGY: self.psychology_model,
            ROLE_PHILOSOPHY: self.philosophy_model,
            ROLE_AMBIGUITY: self.ambiguity_model,
            ROLE_HUMOR: self.humor_model,
            ROLE_STORY: self.story_model,
            ROLE_CRITIC: self.critic_model,
        }
        return mapping[role]
