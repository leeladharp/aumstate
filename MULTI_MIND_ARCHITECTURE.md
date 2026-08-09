# AumState Multi-Mind Architecture

## Overview

AumState now supports two storyboard modes inside Video Studio:

- `Standard`: the existing single-model storyboard planner using `qwen3:8b`
- `Multi-Mind`: one orchestrated creative pipeline that runs specialist reasoning before final storyboard generation

The user still interacts with one AumState agent. The downstream video pipeline remains unchanged:

`CreativeResult -> build_video_plan_from_creative_result(...) -> VideoPlan -> image providers -> narration -> FFmpeg/Kling assembly`

## Specialist Roles

- `Director`: decides intent, tone, narrative shape, and which specialists are needed
- `Psychology`: extracts recognizable human motives and contradictions without clinical diagnosis
- `Philosophy`: identifies deeper tension, source meaning, and modern reflection without preaching
- `Ambiguity`: preserves multiple plausible interpretations and prevents over-explaining
- `Humor`: generates observational or other configured humor candidates from human truth
- `Story`: synthesizes one coherent short-form story
- `Critic`: scores the draft and gives edit instructions for one revision pass

## Orchestration Flow

The creative pipeline lives in [creative_agent.py](/home/leelu/projects/aumstate/creative_agent.py).

It uses a dedicated LangGraph subgraph:

`director -> optional specialists -> story -> critic -> revision`

Conditional routing is driven by the Director:

- philosophical reflection: psychology, philosophy, ambiguity, story, critic
- human contradiction comedy: psychology, ambiguity, humor, story, critic
- straightforward education: story, critic

Specialists run sequentially to avoid loading many local models at once.

## Model Configuration

All roles use local Ollama models only in this phase.

Environment variables:

- `AUMSTATE_DIRECTOR_MODEL`
- `AUMSTATE_PSYCHOLOGY_MODEL`
- `AUMSTATE_PHILOSOPHY_MODEL`
- `AUMSTATE_AMBIGUITY_MODEL`
- `AUMSTATE_HUMOR_MODEL`
- `AUMSTATE_STORY_MODEL`
- `AUMSTATE_CRITIC_MODEL`

Default mapping for every role is `qwen3:8b`.

This keeps the current model in place while allowing one specialist at a time to be swapped later.

## Structured Outputs

Structured dataclasses live in [creative_models.py](/home/leelu/projects/aumstate/creative_models.py).

Key objects:

- `CreativeRequest`
- `DirectorDecision`
- `PsychologyInsight`
- `PhilosophyInsight`
- `HumorInsight`
- `AmbiguityInsight`
- `StoryDraft`
- `CreativeEvaluation`
- `CreativeResult`

The shared Ollama helper `call_local_model(...)`:

- uses role-specific prompts
- requests JSON only
- parses structured output
- retries once on malformed JSON
- raises an actionable error if the second response is still invalid

## VideoPlan Integration

The existing `VideoPlan` contract in [video_agent.py](/home/leelu/projects/aumstate/video_agent.py) is unchanged:

- `title`
- `content_type`
- `duration_seconds`
- `aspect_ratio`
- `narration`
- `style_lock`
- `scenes`

Each scene still contains:

- `scene_number`
- `duration_seconds`
- `narration`
- `visual_prompt`
- `motion_prompt`

`build_video_plan_from_creative_result(...)` converts the creative synthesis into a final storyboard request and validates the same downstream schema.

## Memory Model

Creative memory lives in [creative_memory.py](/home/leelu/projects/aumstate/creative_memory.py).

Tables:

- `creative_preferences`
- `creative_projects`
- `creative_feedback`

This does not replace chat memory or user facts in `app.py`.

Role-specific memory filtering is limited and lightweight:

- humor receives humor preferences and recent humor feedback
- philosophy and critic receive ending/preachiness preferences
- story receives visual and ending preferences

No vector search is used in this phase.

## UI Changes

Video Studio now includes:

- `Creative Intelligence` mode selector
- `Standard` and `Multi-Mind`
- `Humor`, `Depth`, and `Ambiguity` controls in `Multi-Mind`
- `Creative reasoning summary` expander

The main UI does not expose raw prompts or model names.

## Provider Prompt Change

Provider-level nursery hardcoding was removed from [video_providers.py](/home/leelu/projects/aumstate/video_providers.py).

The image provider now stays responsible for:

- continuity
- identity consistency
- technical constraints
- aspect ratio
- no text/logos/watermarks

Creative style now comes from `VideoPlan.style_lock` and each scene prompt.

## Adding Another Specialist Later

To add a new specialist:

1. Add a dataclass in [creative_models.py](/home/leelu/projects/aumstate/creative_models.py)
2. Add a role model env var if needed
3. Add a prompt builder and node in [creative_agent.py](/home/leelu/projects/aumstate/creative_agent.py)
4. Extend Director routing logic
5. Add summary/UI exposure if the output should be visible
6. Add mocked tests

## Example Workflows

Humor workflow:

- Director selects psychology, ambiguity, humor
- Story turns contradiction into scenes
- Critic trims forced explanation
- Revision locks one final version

Philosophy workflow:

- Director selects philosophy, psychology, ambiguity
- Philosophy separates source meaning from modern reflection
- Story converts metaphors into visual beats
- Critic reduces preachiness before final storyboard generation
