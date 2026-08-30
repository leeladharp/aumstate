from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Callable, TypedDict, get_args, get_origin, get_type_hints

import ollama
from langgraph.graph import END, START, StateGraph

from creative_memory import (
    build_role_memory,
    create_creative_project,
    load_creative_preferences,
    load_recent_creative_feedback,
)
from creative_models import (
    CREATIVE_ROLE_ORDER,
    ROLE_AMBIGUITY,
    ROLE_CRITIC,
    ROLE_DIRECTOR,
    ROLE_HUMOR,
    ROLE_PHILOSOPHY,
    ROLE_PSYCHOLOGY,
    ROLE_STORY,
    AmbiguityInsight,
    CreativeEvaluation,
    CreativeModelConfig,
    CreativeRequest,
    CreativeResult,
    DirectorDecision,
    HumorInsight,
    PhilosophyInsight,
    PsychologyInsight,
    RoleExecution,
    StoryDraft,
)


logger = logging.getLogger(__name__)
OPTIONAL_SPECIALIST_ROLES = {ROLE_PSYCHOLOGY, ROLE_PHILOSOPHY, ROLE_AMBIGUITY, ROLE_HUMOR}
ROLE_PROGRESS_LABELS = {
    ROLE_DIRECTOR: "Understanding idea",
    ROLE_PSYCHOLOGY: "Exploring psychology",
    ROLE_PHILOSOPHY: "Exploring philosophy",
    ROLE_AMBIGUITY: "Exploring ambiguity",
    ROLE_HUMOR: "Generating humor",
    ROLE_STORY: "Writing story",
    ROLE_CRITIC: "Reviewing story",
}


class CreativeState(TypedDict):
    request: CreativeRequest
    model_config: CreativeModelConfig
    preferences: dict[str, str]
    feedback_items: list[dict[str, str | int | None]]
    director: DirectorDecision | None
    psychology: PsychologyInsight | None
    philosophy: PhilosophyInsight | None
    ambiguity: AmbiguityInsight | None
    humor: HumorInsight | None
    story: StoryDraft | None
    critic: CreativeEvaluation | None
    final_story: StoryDraft | None
    warnings: list[str]
    execution_log: list[RoleExecution]
    completed_roles: list[str]


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
REFLECTIVE_CONTENT_TYPES = {"philosophy", "spiritual_reflection"}


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


def is_source_based_reflective_request(request: CreativeRequest) -> bool:
    idea_lower = (request.idea or "").lower()
    content_type = (request.content_type or "").lower()
    has_source_marker = any(marker in idea_lower for marker in SOURCE_TEXT_MARKERS)
    has_explicit_metaphors = bool(extract_explicit_source_metaphors(request.idea))
    is_reflective_type = content_type in REFLECTIVE_CONTENT_TYPES or "reflect" in (request.tone or "").lower()
    return (has_source_marker or has_explicit_metaphors) and is_reflective_type


def build_source_fidelity_guidance(request: CreativeRequest) -> str:
    if not is_source_based_reflective_request(request):
        return ""

    metaphors = extract_explicit_source_metaphors(request.idea)
    lines = [
        "This is a source-based reflective request.",
        "Prioritize source fidelity before poetic invention.",
        "Preserve explicit source metaphors unless the user explicitly requests reinterpretation.",
        "Present the source imagery first, then move into modern reflection.",
        "Keep the tone contemplative, but do not replace source meaning with generic spiritual symbolism.",
    ]
    if metaphors:
        lines.append("Explicit source metaphors to preserve in order:")
        lines.extend(f"- {metaphor}" for metaphor in metaphors)
        if any("womb" in metaphor and "unborn" in metaphor for metaphor in metaphors):
            lines.append(
                "- Present unborn life enclosed within the womb respectfully and symbolically, not medically, "
                "and do not transform it into jars, seeds, cracks, or other substitute objects."
            )
    return "\n".join(lines)


def enforce_director_defaults(request: CreativeRequest, decision: DirectorDecision) -> DirectorDecision:
    if not is_source_based_reflective_request(request):
        return decision

    story_focus = decision.story_focus.strip()
    if "preserve source metaphors" not in story_focus.lower():
        story_focus = (
            f"{story_focus}. Preserve source metaphors first, then move into modern reflection."
            if story_focus
            else "Preserve source metaphors first, then move into modern reflection."
        )

    return DirectorDecision(
        content_intent=decision.content_intent,
        emotional_tone=decision.emotional_tone,
        narrative_shape=decision.narrative_shape or "source_metaphor_to_modern_reflection",
        use_psychology=True,
        use_philosophy=True,
        use_humor=False,
        use_ambiguity=True,
        humor_level="off",
        philosophy_level=decision.philosophy_level or "deep",
        psychology_level=decision.psychology_level or "medium",
        ambiguity_level=decision.ambiguity_level or "balanced",
        story_focus=story_focus,
        rationale=decision.rationale,
    )


def dataclass_to_json(instance: Any) -> str:
    if instance is None:
        return "None"
    if is_dataclass(instance):
        return json.dumps(asdict(instance), indent=2, ensure_ascii=True)
    return json.dumps(instance, indent=2, ensure_ascii=True)


def dataclass_from_dict(data_class: type[Any], payload: Any) -> Any:
    if not is_dataclass(data_class):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object for {data_class.__name__}, got {type(payload).__name__}.")

    hints = get_type_hints(data_class)
    values: dict[str, Any] = {}
    for field_info in fields(data_class):
        field_type = hints.get(field_info.name, field_info.type)
        if field_info.name not in payload:
            raise ValueError(f"Missing field '{field_info.name}' for {data_class.__name__}.")
        values[field_info.name] = convert_value(payload[field_info.name], field_type)
    return data_class(**values)


def convert_value(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is list:
        item_type = get_args(annotation)[0]
        if not isinstance(value, list):
            raise ValueError("Expected a list value.")
        return [convert_value(item, item_type) for item in value]
    if origin in {dict, tuple}:
        return value
    if origin is None and is_dataclass(annotation):
        return dataclass_from_dict(annotation, value)
    if origin is None and annotation in {str, int, float, bool}:
        return annotation(value)
    return value


def build_dataclass_schema_instructions(data_class: type[Any]) -> str:
    schema_fields: list[str] = []
    hints = get_type_hints(data_class)
    for field_info in fields(data_class):
        field_type = hints.get(field_info.name, field_info.type)
        type_name = getattr(field_type, "__name__", str(field_type).replace("typing.", ""))
        schema_fields.append(f'- "{field_info.name}": {type_name}')
    return "\n".join(schema_fields)


def call_local_model(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    output_schema: type[Any],
    ollama_chat: Callable[..., dict[str, Any]] = ollama.chat,
) -> Any:
    schema_instructions = build_dataclass_schema_instructions(output_schema)
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt.strip()}\n\n"
                "Return only valid JSON matching this schema exactly:\n"
                "{\n"
                f"{schema_instructions}\n"
                "}\n"
                "Do not include markdown fences or commentary."
            ),
        },
        {"role": "user", "content": user_prompt.strip()},
    ]

    last_error: Exception | None = None
    for attempt in range(2):
        response = ollama_chat(model=model_name, messages=messages)
        raw_text = response["message"]["content"].strip()
        try:
            parsed = json.loads(extract_json_object(raw_text))
            return dataclass_from_dict(output_schema, parsed)
        except Exception as error:
            last_error = error
            if attempt == 0:
                messages.append(
                    {
                        "role": "assistant",
                        "content": raw_text,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "The previous reply was not valid JSON for the requested schema. Return only corrected JSON.",
                    }
                )
    raise ValueError(
        f"{output_schema.__name__} generation failed for model '{model_name}' after one retry: {last_error}"
    )


def extract_json_object(raw_text: str) -> str:
    text = raw_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def load_model_config() -> CreativeModelConfig:
    return CreativeModelConfig(
        director_model=os.getenv("AUMSTATE_DIRECTOR_MODEL", "qwen3:8b").strip() or "qwen3:8b",
        psychology_model=os.getenv("AUMSTATE_PSYCHOLOGY_MODEL", "qwen3:8b").strip() or "qwen3:8b",
        philosophy_model=os.getenv("AUMSTATE_PHILOSOPHY_MODEL", "qwen3:8b").strip() or "qwen3:8b",
        ambiguity_model=os.getenv("AUMSTATE_AMBIGUITY_MODEL", "qwen3:8b").strip() or "qwen3:8b",
        humor_model=os.getenv("AUMSTATE_HUMOR_MODEL", "qwen3:8b").strip() or "qwen3:8b",
        story_model=os.getenv("AUMSTATE_STORY_MODEL", "qwen3:8b").strip() or "qwen3:8b",
        critic_model=os.getenv("AUMSTATE_CRITIC_MODEL", "qwen3:8b").strip() or "qwen3:8b",
    )


def build_director_prompt(request: CreativeRequest, role_memory: str) -> str:
    source_guidance = build_source_fidelity_guidance(request)
    source_section = f"\n\nSource fidelity guidance:\n{source_guidance}" if source_guidance else ""
    return (
        "You are the Creative Director for a short-form video system.\n"
        "Decide which specialist lenses are necessary before storyboard generation.\n"
        "Do not select every specialist unless the idea truly needs them.\n"
        "Humor is optional. Philosophy is optional. Educational content often needs story plus critic only.\n"
        "For source-based reflective prompts, select philosophy, psychology, ambiguity, story, and critic.\n"
        "For verse-based or scripture-based reflection, the first pass must prioritize source fidelity over poetic invention.\n"
        f"Relevant memory:\n{role_memory or '- none'}\n\n"
        f"Request:\n{dataclass_to_json(request)}"
        f"{source_section}"
    )


def build_psychology_prompt(state: CreativeState, role_memory: str) -> str:
    source_guidance = build_source_fidelity_guidance(state["request"])
    source_section = f"\n\nSource fidelity guidance:\n{source_guidance}" if source_guidance else ""
    return (
        "You analyze ordinary human behavior without diagnosing mental illness.\n"
        "Focus on insecurity, status, self-deception, fear, desire, family dynamics, and contradiction.\n"
        "Do not use clinical labels unless explicitly required.\n"
        f"Relevant memory:\n{role_memory or '- none'}\n\n"
        f"Request:\n{dataclass_to_json(state['request'])}\n\n"
        f"Director:\n{dataclass_to_json(state['director'])}"
        f"{source_section}"
    )


def build_philosophy_prompt(state: CreativeState, role_memory: str) -> str:
    source_guidance = build_source_fidelity_guidance(state["request"])
    source_section = f"\n\nSource fidelity guidance:\n{source_guidance}" if source_guidance else ""
    return (
        "You extract philosophical tension without becoming preachy.\n"
        "If the request references a source text, separate source meaning from modern reflection.\n"
        "Avoid fake quotations and avoid claiming interpretation is scripture.\n"
        f"Relevant memory:\n{role_memory or '- none'}\n\n"
        f"Request:\n{dataclass_to_json(state['request'])}\n\n"
        f"Director:\n{dataclass_to_json(state['director'])}"
        f"{source_section}"
    )


def build_ambiguity_prompt(state: CreativeState, role_memory: str) -> str:
    return (
        "You preserve ambiguity where it makes the story more human.\n"
        "Return at least two plausible interpretations and say what should remain unresolved.\n"
        "Do not collapse a contradiction into one absolute truth.\n"
        f"Relevant memory:\n{role_memory or '- none'}\n\n"
        f"Request:\n{dataclass_to_json(state['request'])}\n\n"
        f"Director:\n{dataclass_to_json(state['director'])}\n\n"
        f"Psychology:\n{dataclass_to_json(state['psychology'])}"
    )


def build_humor_prompt(state: CreativeState, role_memory: str) -> str:
    return (
        "You generate humor from recognizable human truth.\n"
        "Supported styles: observational, dry, playful, absurd, satirical.\n"
        "Default to observational unless the request strongly implies otherwise.\n"
        "Return 3 to 5 punchline candidates. Avoid cruelty, random dad jokes, and forced punchlines.\n"
        f"Relevant memory:\n{role_memory or '- none'}\n\n"
        f"Request:\n{dataclass_to_json(state['request'])}\n\n"
        f"Director:\n{dataclass_to_json(state['director'])}\n\n"
        f"Psychology:\n{dataclass_to_json(state['psychology'])}\n\n"
        f"Ambiguity:\n{dataclass_to_json(state['ambiguity'])}"
    )


def build_story_prompt(state: CreativeState, role_memory: str) -> str:
    source_guidance = build_source_fidelity_guidance(state["request"])
    source_section = f"\n\nSource fidelity guidance:\n{source_guidance}" if source_guidance else ""
    return (
        "You are the story writer. Convert insight into one coherent short-form story.\n"
        "Do not concatenate specialist text. Use action, visual situations, and concise narration.\n"
        "For 15 to 30 seconds, create a strong first beat, one contradiction, one turn, and a concise ending.\n"
        "Scene beats must be concrete visual events, objects, or actions that can each become a specific shot.\n"
        "Reject vague beats such as 'introduces the idea', 'develops the story', 'next visual beat', "
        "'establishes the protagonist', or 'resolves the concept'.\n"
        "If the request references a verse, source text, or supplied metaphor, preserve its concrete imagery before adding interpretation.\n"
        "Prefer literal source metaphors over generic spiritual symbols.\n"
        f"Relevant memory:\n{role_memory or '- none'}\n\n"
        f"Request:\n{dataclass_to_json(state['request'])}\n\n"
        f"Director:\n{dataclass_to_json(state['director'])}\n\n"
        f"Psychology:\n{dataclass_to_json(state['psychology'])}\n\n"
        f"Philosophy:\n{dataclass_to_json(state['philosophy'])}\n\n"
        f"Ambiguity:\n{dataclass_to_json(state['ambiguity'])}\n\n"
        f"Humor:\n{dataclass_to_json(state['humor'])}"
        f"{source_section}"
    )


def build_critic_prompt(state: CreativeState, role_memory: str) -> str:
    return (
        "You are the final editor and critic.\n"
        "Score clarity, relatability, humor, psychological truth, philosophical depth, ambiguity, originality, preachiness, forced humor, and unnecessary explanation.\n"
        "Provide concrete edit instructions for one final revision pass.\n"
        f"Relevant memory:\n{role_memory or '- none'}\n\n"
        f"Request:\n{dataclass_to_json(state['request'])}\n\n"
        f"Director:\n{dataclass_to_json(state['director'])}\n\n"
        f"Story draft:\n{dataclass_to_json(state['story'])}"
    )


def build_revision_prompt(state: CreativeState) -> str:
    source_guidance = build_source_fidelity_guidance(state["request"])
    source_section = f"\n\nSource fidelity guidance:\n{source_guidance}" if source_guidance else ""
    return (
        "Revise the story exactly once using the critic instructions.\n"
        "Preserve the strongest central idea. Remove preachiness and unnecessary explanation.\n"
        "Keep the story concise and visual.\n"
        "Every scene beat must remain a concrete visible event, object, or action.\n"
        "Do not leave generic placeholders like 'next beat' or 'develops the contradiction'.\n"
        "If the source included explicit metaphors or images, keep them in the revised scene beats.\n\n"
        f"Request:\n{dataclass_to_json(state['request'])}\n\n"
        f"Original story:\n{dataclass_to_json(state['story'])}\n\n"
        f"Critic:\n{dataclass_to_json(state['critic'])}"
        f"{source_section}"
    )


def make_state(
    request: CreativeRequest,
    model_config: CreativeModelConfig,
) -> CreativeState:
    return {
        "request": request,
        "model_config": model_config,
        "preferences": load_creative_preferences(),
        "feedback_items": load_recent_creative_feedback(),
        "director": None,
        "psychology": None,
        "philosophy": None,
        "ambiguity": None,
        "humor": None,
        "story": None,
        "critic": None,
        "final_story": None,
        "warnings": [],
        "execution_log": [],
        "completed_roles": [],
    }


def role_memory(state: CreativeState, role: str) -> str:
    return build_role_memory(role, state["preferences"], state["feedback_items"])


def run_role(
    state: CreativeState,
    role: str,
    output_schema: type[Any],
    prompt_builder: Callable[[CreativeState, str], str] | Callable[[CreativeRequest, str], str],
    target_key: str,
    ollama_chat: Callable[..., dict[str, Any]] = ollama.chat,
    progress_callback: Callable[[str], None] | None = None,
) -> CreativeState:
    if progress_callback:
        progress_callback(ROLE_PROGRESS_LABELS[role])

    model_name = state["model_config"].model_for_role(role)
    started_at = time.perf_counter()
    try:
        if role == ROLE_DIRECTOR:
            prompt = prompt_builder(state["request"], role_memory(state, role))
        else:
            prompt = prompt_builder(state, role_memory(state, role))
        result = call_local_model(
            model_name=model_name,
            system_prompt=prompt,
            user_prompt="Return the structured analysis now.",
            output_schema=output_schema,
            ollama_chat=ollama_chat,
        )
        if role == ROLE_DIRECTOR:
            result = enforce_director_defaults(state["request"], result)
        duration_seconds = time.perf_counter() - started_at
        logger.info("creative role=%s model=%s duration_seconds=%.3f", role, model_name, duration_seconds)
        state[target_key] = result
        state["completed_roles"] = [*state["completed_roles"], role]
        state["execution_log"] = [
            *state["execution_log"],
            RoleExecution(role=role, model_name=model_name, duration_seconds=duration_seconds, status="ok"),
        ]
        return state
    except Exception as error:
        duration_seconds = time.perf_counter() - started_at
        logger.warning("creative role=%s model=%s failed error=%s", role, model_name, error)
        execution = RoleExecution(
            role=role,
            model_name=model_name,
            duration_seconds=duration_seconds,
            status="failed",
            warning=str(error),
        )
        state["execution_log"] = [*state["execution_log"], execution]
        if role in OPTIONAL_SPECIALIST_ROLES:
            state["warnings"] = [*state["warnings"], f"{role.title()} specialist failed: {error}"]
            return state
        raise


def director_node_factory(
    ollama_chat: Callable[..., dict[str, Any]],
    progress_callback: Callable[[str], None] | None,
) -> Callable[[CreativeState], CreativeState]:
    def node(state: CreativeState) -> CreativeState:
        return run_role(
            state=state,
            role=ROLE_DIRECTOR,
            output_schema=DirectorDecision,
            prompt_builder=build_director_prompt,
            target_key="director",
            ollama_chat=ollama_chat,
            progress_callback=progress_callback,
        )

    return node


def specialist_node_factory(
    role: str,
    output_schema: type[Any],
    prompt_builder: Callable[[CreativeState, str], str],
    target_key: str,
    ollama_chat: Callable[..., dict[str, Any]],
    progress_callback: Callable[[str], None] | None,
) -> Callable[[CreativeState], CreativeState]:
    def node(state: CreativeState) -> CreativeState:
        return run_role(
            state=state,
            role=role,
            output_schema=output_schema,
            prompt_builder=prompt_builder,
            target_key=target_key,
            ollama_chat=ollama_chat,
            progress_callback=progress_callback,
        )

    return node


def revision_node_factory(
    ollama_chat: Callable[..., dict[str, Any]],
    progress_callback: Callable[[str], None] | None,
) -> Callable[[CreativeState], CreativeState]:
    def node(state: CreativeState) -> CreativeState:
        if progress_callback:
            progress_callback("Revising story")
        model_name = state["model_config"].model_for_role(ROLE_CRITIC)
        started_at = time.perf_counter()
        final_story = call_local_model(
            model_name=model_name,
            system_prompt=build_revision_prompt(state),
            user_prompt="Return the final revised story JSON.",
            output_schema=StoryDraft,
            ollama_chat=ollama_chat,
        )
        duration_seconds = time.perf_counter() - started_at
        logger.info("creative role=revision model=%s duration_seconds=%.3f", model_name, duration_seconds)
        state["final_story"] = final_story
        state["execution_log"] = [
            *state["execution_log"],
            RoleExecution(role="revision", model_name=model_name, duration_seconds=duration_seconds, status="ok"),
        ]
        return state

    return node


def should_run_psychology(state: CreativeState) -> str:
    return ROLE_PSYCHOLOGY if state["director"] and state["director"].use_psychology else next_after(ROLE_PSYCHOLOGY, state)


def should_run_philosophy(state: CreativeState) -> str:
    return ROLE_PHILOSOPHY if state["director"] and state["director"].use_philosophy else next_after(ROLE_PHILOSOPHY, state)


def should_run_ambiguity(state: CreativeState) -> str:
    return ROLE_AMBIGUITY if state["director"] and state["director"].use_ambiguity else next_after(ROLE_AMBIGUITY, state)


def should_run_humor(state: CreativeState) -> str:
    return ROLE_HUMOR if state["director"] and state["director"].use_humor else next_after(ROLE_HUMOR, state)


def next_after(role: str, state: CreativeState) -> str:
    if role == ROLE_PSYCHOLOGY:
        if state["director"] and state["director"].use_philosophy:
            return ROLE_PHILOSOPHY
        return next_after(ROLE_PHILOSOPHY, state)
    if role == ROLE_PHILOSOPHY:
        if state["director"] and state["director"].use_ambiguity:
            return ROLE_AMBIGUITY
        return next_after(ROLE_AMBIGUITY, state)
    if role == ROLE_AMBIGUITY:
        if state["director"] and state["director"].use_humor:
            return ROLE_HUMOR
        return ROLE_STORY
    if role == ROLE_HUMOR:
        return ROLE_STORY
    return ROLE_STORY


def create_creative_graph(
    ollama_chat: Callable[..., dict[str, Any]] = ollama.chat,
    progress_callback: Callable[[str], None] | None = None,
):
    builder = StateGraph(CreativeState)
    builder.add_node(ROLE_DIRECTOR, director_node_factory(ollama_chat, progress_callback))
    builder.add_node(
        ROLE_PSYCHOLOGY,
        specialist_node_factory(ROLE_PSYCHOLOGY, PsychologyInsight, build_psychology_prompt, "psychology", ollama_chat, progress_callback),
    )
    builder.add_node(
        ROLE_PHILOSOPHY,
        specialist_node_factory(ROLE_PHILOSOPHY, PhilosophyInsight, build_philosophy_prompt, "philosophy", ollama_chat, progress_callback),
    )
    builder.add_node(
        ROLE_AMBIGUITY,
        specialist_node_factory(ROLE_AMBIGUITY, AmbiguityInsight, build_ambiguity_prompt, "ambiguity", ollama_chat, progress_callback),
    )
    builder.add_node(
        ROLE_HUMOR,
        specialist_node_factory(ROLE_HUMOR, HumorInsight, build_humor_prompt, "humor", ollama_chat, progress_callback),
    )
    builder.add_node(
        ROLE_STORY,
        specialist_node_factory(ROLE_STORY, StoryDraft, build_story_prompt, "story", ollama_chat, progress_callback),
    )
    builder.add_node(
        ROLE_CRITIC,
        specialist_node_factory(ROLE_CRITIC, CreativeEvaluation, build_critic_prompt, "critic", ollama_chat, progress_callback),
    )
    builder.add_node("revision", revision_node_factory(ollama_chat, progress_callback))

    builder.add_edge(START, ROLE_DIRECTOR)
    builder.add_conditional_edges(
        ROLE_DIRECTOR,
        should_run_psychology,
        {
            ROLE_PSYCHOLOGY: ROLE_PSYCHOLOGY,
            ROLE_PHILOSOPHY: ROLE_PHILOSOPHY,
            ROLE_AMBIGUITY: ROLE_AMBIGUITY,
            ROLE_HUMOR: ROLE_HUMOR,
            ROLE_STORY: ROLE_STORY,
        },
    )
    builder.add_conditional_edges(
        ROLE_PSYCHOLOGY,
        should_run_philosophy,
        {
            ROLE_PHILOSOPHY: ROLE_PHILOSOPHY,
            ROLE_AMBIGUITY: ROLE_AMBIGUITY,
            ROLE_HUMOR: ROLE_HUMOR,
            ROLE_STORY: ROLE_STORY,
        },
    )
    builder.add_conditional_edges(
        ROLE_PHILOSOPHY,
        should_run_ambiguity,
        {
            ROLE_AMBIGUITY: ROLE_AMBIGUITY,
            ROLE_HUMOR: ROLE_HUMOR,
            ROLE_STORY: ROLE_STORY,
        },
    )
    builder.add_conditional_edges(
        ROLE_AMBIGUITY,
        should_run_humor,
        {
            ROLE_HUMOR: ROLE_HUMOR,
            ROLE_STORY: ROLE_STORY,
        },
    )
    builder.add_edge(ROLE_HUMOR, ROLE_STORY)
    builder.add_edge(ROLE_STORY, ROLE_CRITIC)
    builder.add_edge(ROLE_CRITIC, "revision")
    builder.add_edge("revision", END)
    return builder.compile()


def build_creative_summary(result: CreativeResult) -> dict[str, Any]:
    return {
        "selected_specialists": result.selected_specialists,
        "psychological_contradiction": result.psychology.contradiction if result.psychology else "",
        "philosophical_question": result.philosophy.central_question if result.philosophy else "",
        "humor_direction": result.humor.humor_style if result.humor else "",
        "ambiguity_note": result.ambiguity.unresolved_question if result.ambiguity else "",
        "critic_scores": {
            "relatability": result.critic.relatability_score,
            "clarity": result.critic.clarity_score,
            "humor": result.critic.humor_score,
            "psychological_truth": result.critic.psychological_truth_score,
            "philosophical_depth": result.critic.philosophical_depth_score,
            "ambiguity": result.critic.ambiguity_score,
            "originality": result.critic.originality_score,
            "preachiness": result.critic.preachiness_score,
        },
    }


def run_creative_pipeline(
    request: CreativeRequest,
    progress_callback: Callable[[str], None] | None = None,
    ollama_chat: Callable[..., dict[str, Any]] = ollama.chat,
    model_config: CreativeModelConfig | None = None,
) -> CreativeResult:
    selected_model_config = model_config or load_model_config()
    graph = create_creative_graph(ollama_chat=ollama_chat, progress_callback=progress_callback)
    final_state = graph.invoke(make_state(request=request, model_config=selected_model_config))

    director = final_state["director"]
    story = final_state["story"]
    critic = final_state["critic"]
    final_story = final_state["final_story"]
    if director is None or story is None or critic is None or final_story is None:
        raise RuntimeError("Creative pipeline did not produce the required final outputs.")

    result = CreativeResult(
        request=request,
        director=director,
        psychology=final_state["psychology"],
        philosophy=final_state["philosophy"],
        humor=final_state["humor"],
        ambiguity=final_state["ambiguity"],
        story=story,
        critic=critic,
        final_story=final_story,
        selected_specialists=director.selected_specialists(),
        warnings=final_state["warnings"],
        execution_log=final_state["execution_log"],
        creative_summary={},
    )
    result.creative_summary = build_creative_summary(result)
    create_creative_project(
        project_id=f"creative_{int(time.time())}",
        title=result.final_story.premise[:120] or request.idea[:120],
        content_type=request.content_type,
    )
    return result
