import base64
import json
import math
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from creative_models import (
    AmbiguityInsight,
    CreativeEvaluation,
    CreativeRequest,
    CreativeResult,
    DirectorDecision,
    PhilosophyInsight,
    PsychologyInsight,
    StoryDraft,
)
from kling_assisted import KLING_ASSISTED_MODE
from video_agent import (
    DEFAULT_VIDEO_MODE,
    DEFAULT_VISUAL_STYLE,
    NarrativeConstraint,
    StoryboardGenerationError,
    VideoPlan,
    VideoScene,
    build_creative_authority_payload,
    build_fallback_plan,
    build_required_constraint_anchor,
    build_scene_durations,
    build_video_plan,
    build_video_plan_from_creative_result,
    build_video_settings,
    infer_scene_continuity,
    narration_word_target,
    normalize_content_type,
    settings_changed,
    settings_snapshot,
    total_duration_seconds,
    validate_video_plan_data,
)
from video_providers import (
    AUDIBLE_AUDIO_THRESHOLD_DB,
    NarrationAudioInfo,
    OpenAIImageProvider,
    OpenAISpeechProvider,
    PlaceholderImageProvider,
    SilentSpeechProvider,
    build_openai_image_prompt,
    create_silent_wav,
    generate_narration_audio,
    generate_scene_images,
    get_config_value,
    get_openai_tts_model,
    inspect_narration_audio_file,
    is_video_dev_mode,
    load_app_config,
    map_quality_label_to_openai,
    select_image_provider,
    select_speech_provider,
)


class MockImagesAPI:
    def __init__(self) -> None:
        self.generate_calls = []
        self.edit_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        image = Image.new("RGB", (1024, 1536), color="#F2D0A7")
        temp_buffer = tempfile.SpooledTemporaryFile()
        image.save(temp_buffer, format="PNG")
        temp_buffer.seek(0)
        encoded = base64.b64encode(temp_buffer.read()).decode("utf-8")
        temp_buffer.close()

        class Response:
            data = [type("ImageData", (), {"b64_json": encoded})()]

        return Response()

    def edit(self, **kwargs):
        self.edit_calls.append(kwargs)
        image = Image.new("RGB", (1024, 1536), color="#DDEAA7")
        temp_buffer = tempfile.SpooledTemporaryFile()
        image.save(temp_buffer, format="PNG")
        temp_buffer.seek(0)
        encoded = base64.b64encode(temp_buffer.read()).decode("utf-8")
        temp_buffer.close()

        class Response:
            data = [type("ImageData", (), {"b64_json": encoded})()]

        return Response()


class MockOpenAIClient:
    def __init__(self) -> None:
        self.images = MockImagesAPI()


class MockSpeechProvider:
    provider_name = "MockSpeech"
    model_name = "mock-model"
    requires_audible_audio = True

    def __init__(self, durations: list[float]) -> None:
        self.durations = durations
        self.calls = []

    def generate_narration_audio(self, narration: str, output_path: Path):
        self.calls.append(narration)
        duration = self.durations[min(len(self.calls) - 1, len(self.durations) - 1)]
        create_silent_wav(output_path=output_path, duration_seconds=duration)
        return type("Result", (), {"path": output_path, "used_fallback": False, "message": "Generated speech."})()


class MockSpeechResponse:
    def __init__(self, audio_bytes: bytes) -> None:
        self.audio_bytes = audio_bytes

    def write_to_file(self, output_path: Path) -> None:
        output_path.write_bytes(self.audio_bytes)


class MockSpeechAPI:
    def __init__(self, audio_bytes: bytes) -> None:
        self.audio_bytes = audio_bytes
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return MockSpeechResponse(self.audio_bytes)


class MockAudioAPI:
    def __init__(self, audio_bytes: bytes) -> None:
        self.speech = MockSpeechAPI(audio_bytes)


class MockOpenAITTSClient:
    def __init__(self, audio_bytes: bytes) -> None:
        self.audio = MockAudioAPI(audio_bytes)


def create_tone_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 22050) -> bytes:
    frame_count = int(sample_rate * duration_seconds)
    amplitude = 12000
    buffer = tempfile.SpooledTemporaryFile()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            sample = int(amplitude * math.sin(2 * math.pi * 440 * (index / sample_rate)))
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))
    buffer.seek(0)
    data = buffer.read()
    buffer.close()
    return data


class VideoAgentTests(unittest.TestCase):
    def make_creative_result(
        self,
        *,
        idea: str,
        content_type: str = "story",
        tone: str = "reflective",
        premise: str,
        conflict: str,
        progression: str,
        emotional_turn: str,
        ending: str,
        scene_beats: list[str],
        use_philosophy: bool = False,
        philosophy: PhilosophyInsight | None = None,
        psychology_contradiction: str = "A self-story and visible behavior do not match.",
        narrative_constraints: list[NarrativeConstraint] | None = None,
    ) -> CreativeResult:
        return CreativeResult(
            request=CreativeRequest(
                idea=idea,
                content_type=content_type,
                tone=tone,
                target_audience="general",
                language="English",
                duration_seconds=30,
                visual_style="Quiet Cinematic Animation",
            ),
            director=DirectorDecision(
                content_intent="human truth",
                emotional_tone=tone,
                narrative_shape="progressive_reveal",
                use_psychology=True,
                use_philosophy=use_philosophy,
                use_humor=False,
                use_ambiguity=True,
                humor_level="off",
                philosophy_level="high" if use_philosophy else "light",
                psychology_level="high",
                ambiguity_level="balanced",
                story_focus="show concrete scenes",
                rationale="This request depends on concrete story beats.",
            ),
            psychology=PsychologyInsight(
                visible_behavior="Visible behavior.",
                hidden_motive="Hidden motive.",
                emotional_trigger="Trigger.",
                contradiction=psychology_contradiction,
                audience_rel_path="Relatable path.",
            ),
            philosophy=philosophy,
            humor=None,
            ambiguity=AmbiguityInsight(
                competing_interpretations=["One reading", "Another reading"],
                unresolved_question="What remains unresolved?",
                contradiction="Contradiction.",
                what_not_to_explain="Leave some implication unstated.",
                ambiguity_strength="balanced",
            ),
            story=StoryDraft(
                premise=premise,
                conflict=conflict,
                progression=progression,
                emotional_turn=emotional_turn,
                ending=ending,
                scene_beats=scene_beats,
            ),
            critic=CreativeEvaluation(
                relatability_score=8,
                humor_score=3,
                psychological_truth_score=8,
                philosophical_depth_score=8 if use_philosophy else 3,
                ambiguity_score=7,
                preachiness_score=2,
                originality_score=7,
                clarity_score=8,
                forced_humor_score=1,
                unnecessary_explanation_score=2,
                notes="Keep visuals concrete.",
                edit_instructions=["Preserve the source image", "Keep beats concrete"],
            ),
            final_story=StoryDraft(
                premise=premise,
                conflict=conflict,
                progression=progression,
                emotional_turn=emotional_turn,
                ending=ending,
                scene_beats=scene_beats,
            ),
            selected_specialists=["psychology", "ambiguity", "story", "critic"],
            narrative_constraints=narrative_constraints or [],
        )

    def make_gita_creative_result(self) -> CreativeResult:
        return self.make_creative_result(
            idea=(
                "Create a quiet animated reflection on Bhagavad Gita 3.38 using smoke covering fire, "
                "dust covering a mirror, and unborn life enclosed within the womb. Use those metaphors "
                "to explore how desire can obscure human clarity."
            ),
            content_type="spiritual_reflection",
            premise="Desire hides clarity through layered coverings named in the verse.",
            conflict="The mind wants to see clearly but keeps getting obscured.",
            progression="The verse images move toward a modern mind clouded by craving.",
            emotional_turn="A modern person notices how comparison and wanting have clouded attention.",
            ending="The disturbance settles and ordinary perception becomes clear again.",
            scene_beats=[
                "Smoke partially hides a flame.",
                "Dust obscures a mirror.",
                "Unborn life remains enclosed within the womb.",
                "A modern person becomes mentally restless while scrolling comparison-driven images.",
                "The person's attention clouds with desire and then slowly settles.",
                "The room becomes still as clear attention returns.",
            ],
            use_philosophy=True,
            philosophy=PhilosophyInsight(
                central_question="What covers clarity?",
                deeper_meaning="Desire can cloud perception without destroying the underlying light.",
                tension="The obscuration feels intimate and self-created.",
                possible_closing_thought="Clarity returns when grasping loosens.",
                avoid_preaching="yes",
                source_meaning="The verse names smoke, dust, and the womb as coverings over what is real.",
                modern_reflection="Comparison and wanting can cloud ordinary attention today.",
            ),
            psychology_contradiction="The person wants clarity but keeps feeding desire.",
            narrative_constraints=[
                NarrativeConstraint("constraint_1", "source_metaphor", "smoke covering fire", "required", 1),
                NarrativeConstraint("constraint_2", "source_metaphor", "dust covering a mirror", "required", 2),
                NarrativeConstraint("constraint_3", "source_metaphor", "unborn life enclosed within the womb", "required", 3),
            ],
        )

    def make_human_comedy_result(self) -> CreativeResult:
        return self.make_creative_result(
            idea="A person claims not to care what others think but keeps checking the WhatsApp status viewer list.",
            content_type="human_behavior",
            tone="wry",
            premise="A person performs indifference while chasing proof of attention.",
            conflict="Words and behavior keep exposing each other.",
            progression="Each glance at the viewer list weakens the claim of detachment.",
            emotional_turn="The person realizes the contradiction mid-ritual.",
            ending="The phone goes dark, but not the need underneath it.",
            scene_beats=[
                "The person says they do not care what other people think.",
                "The same person checks the WhatsApp status viewer list again.",
                "The same person catches the contradiction in silence.",
            ],
            psychology_contradiction="The person says they do not care what people think but keeps checking who viewed the status.",
            narrative_constraints=[
                NarrativeConstraint(
                    "constraint_1",
                    "contradiction",
                    "The person says they do not care what people think but keeps checking who viewed the status.",
                )
            ],
        )

    def make_storyboard(
        self,
        scenes: list[dict[str, object]],
        *,
        title: str | None = "Covered Clarity",
        narration: str | None = "A source-faithful reflection on obscured clarity.",
    ) -> dict[str, object]:
        payload: dict[str, object] = {"scenes": scenes}
        if title is not None:
            payload["title"] = title
        if narration is not None:
            payload["narration"] = narration
        return payload

    def make_gita_storyboard_out_of_order(self) -> dict[str, object]:
        return self.make_storyboard(
            [
                {
                    "scene_number": 1,
                    "narration": "A modern person scrolls in quiet restlessness.",
                    "visual_prompt": "A modern person sits alone at night, scrolling comparison-driven images on a phone.",
                    "motion_prompt": "The thumb moves as the room stays still.",
                    "scene_purpose": "introduce modern restlessness",
                    "continuity_mode": "character",
                    "continuity_group": "human_a",
                },
                {
                    "scene_number": 2,
                    "narration": "Smoke drifts across a small flame and partly hides its light.",
                    "visual_prompt": "Smoke passes in front of a low fire until the flame is only partly visible.",
                    "motion_prompt": "Smoke curls while the flame flickers.",
                    "scene_purpose": "present source metaphor",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 3,
                    "narration": "Desire clouds the person's attention.",
                    "visual_prompt": "The same person lowers the phone as tension gathers in the face.",
                    "motion_prompt": "The shoulders tighten and the camera eases closer.",
                    "scene_purpose": "show obscuration",
                    "continuity_mode": "character",
                    "continuity_group": "human_a",
                },
                {
                    "scene_number": 4,
                    "narration": "Dust settles across a mirror until reflection turns dull.",
                    "visual_prompt": "Fine dust gathers on a standing mirror and obscures the reflected face.",
                    "motion_prompt": "Dust settles while the camera moves closer.",
                    "scene_purpose": "present source metaphor",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 5,
                    "narration": "The room becomes still as attention begins to clear.",
                    "visual_prompt": "The same person sits quietly as the room regains simple clarity.",
                    "motion_prompt": "A gentle exhale and a slight widening of the frame end the sequence.",
                    "scene_purpose": "resolve modern reflection",
                    "continuity_mode": "character",
                    "continuity_group": "human_a",
                },
                {
                    "scene_number": 6,
                    "narration": "Unborn life remains enclosed within the womb.",
                    "visual_prompt": "A respectful symbolic depiction of unborn life enclosed within the womb in shadowed warmth.",
                    "motion_prompt": "The frame holds with a slow protective drift.",
                    "scene_purpose": "present source metaphor",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
            ]
        )

    def test_basic_motion_is_default_video_mode(self) -> None:
        settings = build_video_settings()
        self.assertEqual(settings.video_mode, DEFAULT_VIDEO_MODE)

    def test_scene_durations_sum_exactly_and_never_zero(self) -> None:
        durations = build_scene_durations(60, 6)
        self.assertEqual(sum(durations), 60)
        self.assertTrue(all(duration > 0 for duration in durations))

    def test_validate_video_plan_data_enforces_exact_scene_durations(self) -> None:
        settings = build_video_settings(total_duration_seconds=10, preferred_scene_duration_seconds=3)
        invalid_plan = {
            "title": "Invalid",
            "content_type": "story",
            "duration_seconds": 10,
            "aspect_ratio": "9:16",
            "narration": "Narration",
            "style_lock": "Shared style",
            "scenes": [
                {"scene_number": 1, "duration_seconds": 3, "narration": "One", "visual_prompt": "One", "motion_prompt": "Still"},
                {"scene_number": 2, "duration_seconds": 3, "narration": "Two", "visual_prompt": "Two", "motion_prompt": "Still"},
                {"scene_number": 3, "duration_seconds": 3, "narration": "Three", "visual_prompt": "Three", "motion_prompt": "Still"},
                {"scene_number": 4, "duration_seconds": 2, "narration": "Four", "visual_prompt": "Four", "motion_prompt": "Still"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "must be 1 seconds"):
            validate_video_plan_data(invalid_plan, settings=settings)

    def test_unknown_content_type_falls_back_to_story(self) -> None:
        self.assertEqual(normalize_content_type("totally unknown"), "story")
        self.assertEqual(build_video_settings(content_type="???").content_type, "story")

    def test_creative_result_constraints_drive_source_metaphors(self) -> None:
        constraints = self.make_gita_creative_result().narrative_constraints
        self.assertEqual(
            [constraint.description for constraint in constraints[:3]],
            [
                "smoke covering fire",
                "dust covering a mirror",
                "unborn life enclosed within the womb",
            ],
        )
        self.assertTrue(all(isinstance(constraint, NarrativeConstraint) for constraint in constraints))

    def test_creative_authority_payload_uses_constraints_not_scene_assignments(self) -> None:
        result = self.make_gita_creative_result()
        settings = build_video_settings(total_duration_seconds=30)
        payload = build_creative_authority_payload(result, settings)
        self.assertIn("narrative_constraints", payload)
        self.assertNotIn("scene_assignments", payload.get("source_fidelity", {}))
        self.assertNotIn("third_metaphor_guidance", payload.get("source_fidelity", {}))

    def test_symbolic_scene_continuity_hints_stay_independent(self) -> None:
        hints = infer_scene_continuity(
            [
                "Smoke drifts across a low fire.",
                "Dust settles over a mirror.",
                "Unborn life remains enclosed within the womb.",
            ]
        )

        self.assertEqual(
            hints,
            [("independent", None), ("independent", None), ("independent", None)],
        )

    def test_recurring_human_scene_continuity_hints_share_character_group(self) -> None:
        hints = infer_scene_continuity(
            [
                "A person waits by the train window.",
                "The same person folds the letter again.",
                "The person finally steps onto the platform.",
            ]
        )

        self.assertEqual(
            hints,
            [("character", "human_a"), ("character", "human_a"), ("character", "human_a")],
        )

    def test_gita_style_reflection_passes_without_scene_number_anchor_locking(self) -> None:
        result = self.make_gita_creative_result()
        settings = build_video_settings(total_duration_seconds=30, content_type="spiritual_reflection")
        raw_plan = self.make_gita_storyboard_out_of_order()

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)) as plan_mock:
            with patch("video_agent.request_storyboard_revision_from_ollama") as revision_mock:
                plan, warning = build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertIsNone(warning)
        self.assertEqual(plan_mock.call_count, 1)
        revision_mock.assert_not_called()
        joined = " ".join(scene.narration + " " + scene.visual_prompt for scene in plan.scenes).lower()
        self.assertIn("smoke", joined)
        self.assertIn("mirror", joined)
        self.assertIn("womb", joined)
        self.assertEqual(plan.scenes[1].scene_purpose, "present source metaphor")
        self.assertEqual(plan.scenes[5].scene_purpose, "present source metaphor")

    def test_missing_womb_constraint_triggers_one_revision_then_passes(self) -> None:
        result = self.make_gita_creative_result()
        settings = build_video_settings(total_duration_seconds=30, content_type="spiritual_reflection")
        bad_plan = self.make_gita_storyboard_out_of_order()
        bad_plan["scenes"][5]["narration"] = "A seed waits inside a clay jar."
        bad_plan["scenes"][5]["visual_prompt"] = "A seed sealed inside a clay jar in warm darkness."
        revised_plan = self.make_gita_storyboard_out_of_order()

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(bad_plan)):
            with patch("video_agent.request_storyboard_revision_from_ollama", return_value=json.dumps(revised_plan)) as revision_mock:
                plan, warning = build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertIsNone(warning)
        self.assertEqual(revision_mock.call_count, 1)
        self.assertIn("womb", plan.scenes[5].visual_prompt.lower())
        self.assertTrue(
            any(term in plan.scenes[5].visual_prompt.lower() for term in ("unborn", "fetal", "fetus", "child", "baby"))
        )

    def test_concrete_required_subject_survives_storyboard_generation(self) -> None:
        result = self.make_creative_result(
            idea="A wife discovers a package by the door and realizes it was meant for her.",
            content_type="story",
            premise="A delivery becomes a small emotional reveal.",
            conflict="The unopened package carries quiet uncertainty.",
            progression="The wife notices the package, approaches it, and opens it.",
            emotional_turn="Recognition replaces suspicion.",
            ending="She smiles when she sees her name on the note.",
            scene_beats=[
                "A package waits by the front door.",
                "The wife discovers the package and kneels beside it.",
                "The wife opens the package and reads the note.",
            ],
            narrative_constraints=[
                NarrativeConstraint(
                    "constraint_1",
                    "required_event",
                    "package discovered by wife",
                )
            ],
        )
        settings = build_video_settings(total_duration_seconds=15, content_type="story")
        bad_plan = self.make_storyboard(
            [
                {
                    "scene_number": 1,
                    "narration": "A strange glow waits near the doorway.",
                    "visual_prompt": "A mysterious box-shaped glow hovers near the front door.",
                    "motion_prompt": "The glow pulses faintly in the quiet hall.",
                    "scene_purpose": "introduce mystery",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 2,
                    "narration": "She senses that the sign is meant for her.",
                    "visual_prompt": "A woman pauses before the glowing shape without touching it.",
                    "motion_prompt": "She leans in while the light trembles.",
                    "scene_purpose": "build intrigue",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
                {
                    "scene_number": 3,
                    "narration": "The hallway softens into understanding.",
                    "visual_prompt": "Warm light fills the entryway as the meaning becomes clear.",
                    "motion_prompt": "The camera drifts closer through the glow.",
                    "scene_purpose": "resolve mystery",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
            ],
            title="Doorway Signal",
            narration=None,
        )
        fixed_plan = self.make_storyboard(
            [
                {
                    "scene_number": 1,
                    "narration": "A package waits by the front door at the wife's feet.",
                    "visual_prompt": "A taped package rests by the front door in a quiet apartment entryway.",
                    "motion_prompt": "The camera eases toward the package from the hallway.",
                    "scene_purpose": "introduce discovery setup",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 2,
                    "narration": "The wife discovers the package and kneels beside it in surprise.",
                    "visual_prompt": "The wife kneels by the front door and touches the discovered package with cautious curiosity.",
                    "motion_prompt": "She slows, bends down, and reaches toward the package.",
                    "scene_purpose": "show discovery",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
                {
                    "scene_number": 3,
                    "narration": "She opens the package and sees that it was meant for her.",
                    "visual_prompt": "The same wife opens the package on the floor and finds a handwritten note inside.",
                    "motion_prompt": "The lid lifts and her expression softens.",
                    "scene_purpose": "resolve reveal",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
            ],
            title="Doorway Delivery",
            narration=None,
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(bad_plan)):
            with patch("video_agent.request_storyboard_revision_from_ollama", return_value=json.dumps(fixed_plan)) as revision_mock:
                plan, warning = build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertIsNone(warning)
        self.assertEqual(revision_mock.call_count, 1)
        storyboard_text = " ".join(scene.narration + " " + scene.visual_prompt for scene in plan.scenes).lower()
        self.assertIn("package", storyboard_text)
        self.assertIn("wife", storyboard_text)

    def test_concrete_required_subject_ignores_abstract_metaphor_explanation_tail(self) -> None:
        result = self.make_creative_result(
            idea="A reflective sequence uses the womb image to show how desire can hide potential.",
            content_type="spiritual_reflection",
            premise="Potential remains present even when covered.",
            conflict="Desire obscures what is still alive underneath.",
            progression="The image appears first, then the reflection interprets it.",
            emotional_turn="The viewer recognizes the covering without losing sight of what is covered.",
            ending="The metaphor lands without replacing the subject itself.",
            scene_beats=[
                "Unborn life remains enclosed by the womb.",
                "The image invites reflection on obscured potential.",
                "Clarity returns without erasing the original metaphor.",
            ],
            narrative_constraints=[
                NarrativeConstraint(
                    "constraint_1",
                    "required_object",
                    "Unborn life enclosed by the womb as a metaphor for desire's containment of potential.",
                )
            ],
        )
        settings = build_video_settings(total_duration_seconds=15, content_type="spiritual_reflection")
        raw_plan = self.make_storyboard(
            [
                {
                    "scene_number": 1,
                    "narration": "Unborn life rests within the womb in protected stillness.",
                    "visual_prompt": "A respectful symbolic depiction of unborn life visibly enclosed within the womb in warm shadow.",
                    "motion_prompt": "The frame holds with a slow protective drift.",
                    "scene_purpose": "present source metaphor",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 2,
                    "narration": "The image lingers before the interpretation begins.",
                    "visual_prompt": "The same warm enclosure remains still as the metaphor settles.",
                    "motion_prompt": "The camera breathes forward almost imperceptibly.",
                    "scene_purpose": "hold reflection",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 3,
                    "narration": "The meaning becomes clear without losing the original image.",
                    "visual_prompt": "The protected unborn form remains identifiable as the scene fades toward clarity.",
                    "motion_prompt": "A soft widening of the frame closes the sequence.",
                    "scene_purpose": "resolve reflection",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
            ],
            narration=None,
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_storyboard_revision_from_ollama") as revision_mock:
                plan, warning = build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertIsNone(warning)
        revision_mock.assert_not_called()
        self.assertIn("womb", plan.scenes[0].visual_prompt.lower())
        self.assertTrue(
            any(term in plan.scenes[0].visual_prompt.lower() for term in ("unborn", "fetal", "fetus", "child", "baby"))
        )

    def test_required_constraint_anchor_splits_illustrates_and_hints_at_tails(self) -> None:
        womb_anchor = build_required_constraint_anchor(
            NarrativeConstraint(
                "constraint_1",
                "required_object",
                "Unborn life enclosed by the womb illustrates desire concealing potential.",
            )
        )
        water_anchor = build_required_constraint_anchor(
            NarrativeConstraint(
                "constraint_2",
                "required_object",
                "Viewer lingers on water's surface where shadow's dissolution hints at erasure and potential.",
            )
        )

        assert womb_anchor is not None
        assert water_anchor is not None
        self.assertEqual(womb_anchor.canonical_text, "unborn life enclosed by the womb")
        self.assertEqual(
            water_anchor.canonical_text,
            "viewer lingers on water's surface where shadow's dissolution",
        )
        self.assertEqual(womb_anchor.minimum_groups, 2)
        self.assertEqual(water_anchor.minimum_groups, 2)

    def test_required_constraint_anchor_skips_abstract_only_constraint(self) -> None:
        anchor = build_required_constraint_anchor(
            NarrativeConstraint(
                "constraint_1",
                "required_object",
                "Desire functions as a mental fog filtering experience through craving.",
            )
        )

        self.assertIsNone(anchor)

    def test_revision_failure_is_explicit_and_stops_after_one_try(self) -> None:
        result = self.make_gita_creative_result()
        settings = build_video_settings(total_duration_seconds=30, content_type="spiritual_reflection")
        bad_plan = self.make_gita_storyboard_out_of_order()
        bad_plan["scenes"][5]["narration"] = "A seed waits inside a clay jar."
        bad_plan["scenes"][5]["visual_prompt"] = "A seed sealed inside a clay jar in warm darkness."

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(bad_plan)):
            with patch("video_agent.request_storyboard_revision_from_ollama", return_value=json.dumps(bad_plan)) as revision_mock:
                with self.assertRaisesRegex(ValueError, "Storyboard revision failed validation"):
                    build_video_plan_from_creative_result(result.request.idea, result, settings)
        self.assertEqual(revision_mock.call_count, 1)

    def test_revision_json_decode_error_reports_stage(self) -> None:
        result = self.make_gita_creative_result()
        settings = build_video_settings(total_duration_seconds=30, content_type="spiritual_reflection")

        with patch("video_agent.request_video_plan_from_ollama", return_value="{bad json"):
            with patch("video_agent.request_storyboard_revision_from_ollama", return_value="{still bad json"):
                with self.assertRaises(StoryboardGenerationError) as context:
                    build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertEqual(context.exception.stage, "revision_parsing")
        self.assertIn("Storyboard revision failed during JSON parsing", str(context.exception))

    def test_revision_validation_error_reports_stage(self) -> None:
        result = self.make_gita_creative_result()
        settings = build_video_settings(total_duration_seconds=30, content_type="spiritual_reflection")
        bad_plan = self.make_gita_storyboard_out_of_order()
        bad_plan["scenes"][5]["narration"] = "A seed waits inside a clay jar."
        bad_plan["scenes"][5]["visual_prompt"] = "A seed sealed inside a clay jar in warm darkness."

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(bad_plan)):
            with patch("video_agent.request_storyboard_revision_from_ollama", return_value=json.dumps(bad_plan)):
                with self.assertRaises(StoryboardGenerationError) as context:
                    build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertEqual(context.exception.stage, "revision_validation")
        self.assertIn("Storyboard revision failed validation", str(context.exception))

    def test_human_behavior_contradiction_survives_into_storyboard(self) -> None:
        result = self.make_human_comedy_result()
        settings = build_video_settings(total_duration_seconds=15, content_type="human_behavior")
        raw_plan = self.make_storyboard(
            [
                {
                    "scene_number": 1,
                    "narration": "The person says they do not care what people think.",
                    "visual_prompt": "A person shrugs in public and claims indifference.",
                    "motion_prompt": "A small shrug and dismissive hand wave.",
                    "scene_purpose": "state the claim",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
                {
                    "scene_number": 2,
                    "narration": "Seconds later the same person checks who viewed the status again.",
                    "visual_prompt": "The same person stares at the WhatsApp status viewer list on the phone.",
                    "motion_prompt": "The thumb refreshes the list and pauses.",
                    "scene_purpose": "reveal contradiction",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
                {
                    "scene_number": 3,
                    "narration": "The screen light catches the person's quiet embarrassment.",
                    "visual_prompt": "The same person locks the phone and avoids their own reflection.",
                    "motion_prompt": "The phone lowers and the eyes look away.",
                    "scene_purpose": "land the recognition",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
            ],
            title="Status Ritual",
            narration=None,
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertIsNone(warning)
        self.assertIn("do not care", plan.narration.lower())
        self.assertIn("viewed the status", plan.narration.lower())

    def test_recurring_character_story_keeps_shared_character_continuity(self) -> None:
        result = self.make_creative_result(
            idea="A couple moves through a tense but affectionate morning.",
            content_type="story",
            premise="A couple moves through tension without losing tenderness.",
            conflict="Minor friction keeps interrupting small acts of care.",
            progression="Breakfast, silence, and repair unfold across the same morning.",
            emotional_turn="They finally laugh at themselves.",
            ending="The kitchen feels like theirs again.",
            scene_beats=[
                "A couple prepares breakfast together.",
                "The same couple goes quiet after a small misunderstanding.",
                "The same couple reconnects over a laugh at the kitchen table.",
            ],
            narrative_constraints=[
                NarrativeConstraint(
                    "constraint_1",
                    "required_character",
                    "A couple must appear as recurring human subjects where the story calls for them.",
                )
            ],
        )
        settings = build_video_settings(total_duration_seconds=15, content_type="story")
        raw_plan = self.make_storyboard(
            [
                {
                    "scene_number": 1,
                    "narration": "A couple moves around each other in a familiar kitchen.",
                    "visual_prompt": "A couple prepares breakfast together in the same sunlit kitchen.",
                    "motion_prompt": "Hands pass plates while morning light stays steady.",
                    "scene_purpose": "introduce relationship",
                    "continuity_mode": "character",
                    "continuity_group": "couple_a",
                },
                {
                    "scene_number": 2,
                    "narration": "The same couple turns quiet after a small misunderstanding.",
                    "visual_prompt": "The same couple stands apart in the same kitchen, both pretending to stay busy.",
                    "motion_prompt": "One person wipes the counter while the other pauses at the sink.",
                    "scene_purpose": "build tension",
                    "continuity_mode": "character",
                    "continuity_group": "couple_a",
                },
                {
                    "scene_number": 3,
                    "narration": "The same couple laughs and lets the room soften again.",
                    "visual_prompt": "The same couple sits close at the kitchen table and smiles at the same joke.",
                    "motion_prompt": "Shoulders relax and both lean into the shared laugh.",
                    "scene_purpose": "resolve tension",
                    "continuity_mode": "character",
                    "continuity_group": "couple_a",
                },
            ]
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, _ = build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertEqual([scene.continuity_group for scene in plan.scenes], ["couple_a", "couple_a", "couple_a"])

    def test_missing_title_uses_derived_title(self) -> None:
        result = self.make_gita_creative_result()
        settings = build_video_settings(total_duration_seconds=30, content_type="spiritual_reflection")
        raw_plan = self.make_gita_storyboard_out_of_order()
        raw_plan.pop("title")

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, _ = build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertTrue(plan.title)
        self.assertNotEqual(plan.title, "Storyboard")

    def test_missing_top_level_narration_assembles_from_scene_narration(self) -> None:
        result = self.make_human_comedy_result()
        result.psychology = PsychologyInsight(
            visible_behavior="Visible behavior.",
            hidden_motive="Hidden motive.",
            emotional_trigger="Trigger.",
            contradiction="",
            audience_rel_path="Relatable path.",
        )
        result.narrative_constraints = []
        settings = build_video_settings(total_duration_seconds=15, content_type="human_behavior")
        raw_plan = self.make_storyboard(
            [
                {
                    "scene_number": 1,
                    "narration": "First line.",
                    "visual_prompt": "A concrete first image.",
                    "motion_prompt": "A gentle first motion.",
                    "scene_purpose": "open",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 2,
                    "narration": "Second line.",
                    "visual_prompt": "A concrete second image.",
                    "motion_prompt": "A gentle second motion.",
                    "scene_purpose": "middle",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 3,
                    "narration": "Third line.",
                    "visual_prompt": "A concrete third image.",
                    "motion_prompt": "A gentle third motion.",
                    "scene_purpose": "close",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
            ],
            narration=None,
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, _ = build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertEqual(plan.narration, "First line. Second line. Third line.")

    def test_missing_scene_purpose_defaults_during_creative_storyboard_validation(self) -> None:
        result = self.make_human_comedy_result()
        settings = build_video_settings(total_duration_seconds=15, content_type="human_behavior")
        raw_plan = self.make_storyboard(
            [
                {
                    "scene_number": 1,
                    "narration": "The person says they do not care what people think.",
                    "visual_prompt": "A person shrugs in public and claims indifference.",
                    "motion_prompt": "A small shrug and dismissive hand wave.",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
                {
                    "scene_number": 2,
                    "narration": "Seconds later the same person checks who viewed the status again.",
                    "visual_prompt": "The same person stares at the WhatsApp status viewer list on the phone.",
                    "motion_prompt": "The thumb refreshes the list and pauses.",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
                {
                    "scene_number": 3,
                    "narration": "The screen light catches the person's quiet embarrassment.",
                    "visual_prompt": "The same person locks the phone and avoids their own reflection.",
                    "motion_prompt": "The phone lowers and the eyes look away.",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
            ],
            narration=None,
        )
        for scene in raw_plan["scenes"]:
            scene.pop("scene_purpose", None)

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_storyboard_revision_from_ollama") as revision_mock:
                plan, warning = build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertIsNone(warning)
        revision_mock.assert_not_called()
        self.assertEqual([scene.scene_purpose for scene in plan.scenes], ["open", "middle", "close"])

    def test_generic_placeholder_scene_triggers_one_revision(self) -> None:
        result = self.make_human_comedy_result()
        settings = build_video_settings(total_duration_seconds=15, content_type="human_behavior")
        bad_plan = self.make_storyboard(
            [
                {
                    "scene_number": 1,
                    "narration": "The person says they do not care what people think.",
                    "visual_prompt": "A concrete first image.",
                    "motion_prompt": "A gentle first motion.",
                    "scene_purpose": "open",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
                {
                    "scene_number": 2,
                    "narration": "Develops the story.",
                    "visual_prompt": "Middle frame showing the next visual beat.",
                    "motion_prompt": "A gentle second motion.",
                    "scene_purpose": "middle",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
                {
                    "scene_number": 3,
                    "narration": "The phone goes dark.",
                    "visual_prompt": "A concrete final image.",
                    "motion_prompt": "A gentle final motion.",
                    "scene_purpose": "close",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
            ]
        )
        fixed_plan = self.make_storyboard(
            [
                bad_plan["scenes"][0],
                {
                    "scene_number": 2,
                    "narration": "The same person checks the viewer list again.",
                    "visual_prompt": "The same person refreshes the WhatsApp status viewer list in the glow of the phone.",
                    "motion_prompt": "The thumb taps refresh and then hesitates.",
                    "scene_purpose": "reveal contradiction",
                    "continuity_mode": "character",
                    "continuity_group": "person_a",
                },
                bad_plan["scenes"][2],
            ]
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(bad_plan)):
            with patch("video_agent.request_storyboard_revision_from_ollama", return_value=json.dumps(fixed_plan)) as revision_mock:
                plan, _ = build_video_plan_from_creative_result(result.request.idea, result, settings)

        self.assertEqual(revision_mock.call_count, 1)
        self.assertIn("viewer list", plan.scenes[1].narration.lower())

    def test_standard_mode_still_builds_valid_video_plan(self) -> None:
        settings = build_video_settings(total_duration_seconds=15, content_type="story")
        raw_plan = {
            "title": "Simple Story",
            "content_type": "story",
            "duration_seconds": 15,
            "aspect_ratio": "9:16",
            "narration": "A simple narrated story.",
            "style_lock": "Quiet cinematic animation, soft natural light, coherent palette.",
            "scenes": [
                {
                    "scene_number": 1,
                    "duration_seconds": 5,
                    "narration": "Scene one.",
                    "visual_prompt": "A child stands under a blue umbrella at a rainy bus stop.",
                    "motion_prompt": "Rain slides along the umbrella edge as the camera drifts closer.",
                    "scene_purpose": "open",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 2,
                    "duration_seconds": 5,
                    "narration": "Scene two.",
                    "visual_prompt": "The bus doors open and warm interior light reaches the child.",
                    "motion_prompt": "The doors fold open and the frame eases forward.",
                    "scene_purpose": "middle",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 3,
                    "duration_seconds": 5,
                    "narration": "Scene three.",
                    "visual_prompt": "The child takes a seat by the window and watches the town pass quietly.",
                    "motion_prompt": "Street reflections slide across the glass as the bus begins to move.",
                    "scene_purpose": "close",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
            ],
        }

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan("Tell a simple story.", settings)

        self.assertIsNone(warning)
        self.assertEqual(plan.title, "Simple Story")
        self.assertFalse(plan.used_fallback)

    def test_kling_assisted_mode_still_builds_fallback_plan(self) -> None:
        settings = build_video_settings(video_mode=KLING_ASSISTED_MODE, total_duration_seconds=10)
        plan = build_fallback_plan("A quiet tea moment.", settings)
        self.assertIsInstance(plan, VideoPlan)
        self.assertEqual(plan.settings.video_mode, KLING_ASSISTED_MODE)
        self.assertEqual(total_duration_seconds(plan.scenes), 10)

    def test_openai_image_provider_uses_continuity_group_references(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, content_type="story")
        scenes = [
            VideoScene(1, 5, "One.", "Independent image one.", "Still.", "independent", None, "open"),
            VideoScene(2, 5, "Two.", "Independent image two.", "Still.", "independent", None, "beat"),
            VideoScene(3, 5, "Three.", "Independent image three.", "Still.", "independent", None, "beat"),
            VideoScene(4, 5, "Four.", "Couple first image.", "Still.", "character", "couple_a", "introduce"),
            VideoScene(5, 5, "Five.", "Couple second image.", "Still.", "character", "couple_a", "continue"),
            VideoScene(6, 5, "Six.", "Couple third image.", "Still.", "character", "couple_a", "resolve"),
        ]
        plan = VideoPlan(
            title="Continuity",
            content_type="story",
            duration_seconds=30,
            aspect_ratio="9:16",
            narration="Narration",
            style_lock="Quiet cinematic animation.",
            scenes=scenes,
            settings=settings,
        )
        client = MockOpenAIClient()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            provider = OpenAIImageProvider(plan=plan, settings=settings, client=client)
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                for scene in plan.scenes:
                    provider.generate_scene_image(scene, output_dir / f"scene_{scene.scene_number:02d}.png")

        self.assertEqual(len(client.images.generate_calls), 4)
        self.assertEqual(len(client.images.edit_calls), 2)

    def test_independent_scenes_generate_fresh_images(self) -> None:
        settings = build_video_settings(total_duration_seconds=15, content_type="story")
        plan = build_fallback_plan("A short story.", settings=settings)
        client = MockOpenAIClient()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            provider = OpenAIImageProvider(plan=plan, settings=settings, client=client)
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                for scene in plan.scenes:
                    provider.generate_scene_image(scene, output_dir / f"scene_{scene.scene_number:02d}.png")

        self.assertEqual(len(client.images.generate_calls), len(plan.scenes))
        self.assertEqual(len(client.images.edit_calls), 0)

    def test_openai_image_prompt_mentions_continuity_group_when_present(self) -> None:
        settings = build_video_settings(total_duration_seconds=15)
        plan = build_fallback_plan("A short story.", settings)
        scene = plan.scenes[0]
        scene.continuity_mode = "character"
        scene.continuity_group = "human_a"
        prompt = build_openai_image_prompt(plan, scene)
        self.assertIn("continuity_group=human_a", prompt)

    def test_generate_scene_images_falls_back_to_placeholder_files_in_dev_mode(self) -> None:
        settings = build_video_settings()
        plan = build_fallback_plan("Create a simple story.", settings=settings)

        with patch.dict("os.environ", {"VIDEO_DEV_MODE": "true"}, clear=False):
            with tempfile.TemporaryDirectory() as temp_dir:
                image_paths, messages = generate_scene_images(plan=plan, output_dir=Path(temp_dir), settings=settings)
                self.assertEqual(len(image_paths), 3)
                self.assertTrue(all(path.exists() for path in image_paths))
                self.assertIn("PlaceholderImageProvider", messages[0])

    def test_narration_disabled_skips_speech_generation(self) -> None:
        settings = build_video_settings(narration_enabled=False)
        plan = build_fallback_plan("A short story", settings=settings)
        audio_path, message = generate_narration_audio(plan=plan, output_dir=Path(tempfile.mkdtemp()), settings=settings)
        self.assertIsNone(audio_path)
        self.assertEqual(message, "Narration disabled.")

    def test_missing_video_dev_mode_defaults_to_production_mode(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            load_app_config(force=True)
            self.assertFalse(is_video_dev_mode())

    def test_video_dev_mode_true_selects_silent_speech_provider(self) -> None:
        settings = build_video_settings()
        with patch.dict(os.environ, {"VIDEO_DEV_MODE": "true"}, clear=True):
            load_app_config(force=True)
            provider = select_speech_provider(settings=settings)
        self.assertIsInstance(provider, SilentSpeechProvider)

    def test_video_dev_mode_false_selects_openai_speech_provider(self) -> None:
        settings = build_video_settings()
        with patch.dict(os.environ, {"VIDEO_DEV_MODE": "false"}, clear=True):
            load_app_config(force=True)
            provider = select_speech_provider(settings=settings)
        self.assertIsInstance(provider, OpenAISpeechProvider)

    def test_dotenv_values_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = Path(temp_dir) / ".env"
            dotenv_path.write_text(
                "VIDEO_DEV_MODE=true\nOPENAI_API_KEY=dotenv-key\nOPENAI_TTS_MODEL=dotenv-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                load_app_config(force=True, dotenv_path=dotenv_path)
                self.assertTrue(is_video_dev_mode())
                self.assertEqual(get_config_value("OPENAI_API_KEY"), "dotenv-key")
                self.assertEqual(get_openai_tts_model(), "dotenv-model")

    def test_environment_variables_override_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = Path(temp_dir) / ".env"
            dotenv_path.write_text("OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
                load_app_config(force=True, dotenv_path=dotenv_path)
                self.assertEqual(get_config_value("OPENAI_API_KEY"), "env-key")

    def test_missing_production_api_key_raises_actionable_error(self) -> None:
        settings = build_video_settings()
        with patch.dict(os.environ, {"VIDEO_DEV_MODE": "false"}, clear=True):
            load_app_config(force=True)
            provider = OpenAISpeechProvider(settings=settings, client=object())
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY is missing"):
                provider.generate_narration_audio("Hello world", Path(tempfile.mkdtemp()) / "narration.wav")

    def test_mocked_openai_tts_receives_model_voice_text_style_and_speed(self) -> None:
        settings = build_video_settings(voice="Warm Female", speaking_style="Calm", speaking_speed="Fast", language="English")
        client = MockOpenAITTSClient(create_tone_wav_bytes())
        with patch.dict(
            os.environ,
            {"VIDEO_DEV_MODE": "false", "OPENAI_API_KEY": "test-key", "OPENAI_TTS_MODEL": "gpt-4o-mini-tts"},
            clear=True,
        ):
            load_app_config(force=True)
            provider = OpenAISpeechProvider(settings=settings, client=client)
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "narration.wav"
                result = provider.generate_narration_audio("Narration text", output_path)
                self.assertTrue(result.path.exists())
        call = client.audio.speech.calls[0]
        self.assertEqual(call["model"], "gpt-4o-mini-tts")
        self.assertEqual(call["voice"], "coral")
        self.assertEqual(call["input"], "Narration text")
        self.assertIn("Speaking style: Calm", call["instructions"])
        self.assertEqual(call["speed"], 1.1)

    def test_valid_audio_file_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "narration.wav"
            audio_path.write_bytes(create_tone_wav_bytes())
            with patch("video_providers.probe_audio_file", return_value={"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}], "format": {"duration": "1.0"}}):
                with patch("video_providers.get_audio_max_volume_db", return_value=-12.0):
                    info = inspect_narration_audio_file(audio_path, provider_name="OpenAI", model_name="gpt-4o-mini-tts")
        self.assertTrue(info.contains_audible_audio)
        self.assertEqual(info.codec_name, "pcm_s16le")

    def test_silent_audio_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "silent.wav"
            audio_path.write_bytes(create_tone_wav_bytes())
            with patch("video_providers.probe_audio_file", return_value={"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}], "format": {"duration": "1.0"}}):
                with patch("video_providers.get_audio_max_volume_db", return_value=-91.0):
                    with self.assertRaisesRegex(ValueError, "silent or invalid"):
                        inspect_narration_audio_file(audio_path)

    def test_audible_audio_succeeds_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audible.wav"
            audio_path.write_bytes(create_tone_wav_bytes())
            with patch("video_providers.probe_audio_file", return_value={"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}], "format": {"duration": "1.0"}}):
                with patch("video_providers.get_audio_max_volume_db", return_value=AUDIBLE_AUDIO_THRESHOLD_DB + 1.0):
                    info = inspect_narration_audio_file(audio_path)
        self.assertTrue(info.contains_audible_audio)

    def test_narration_retry_does_not_regenerate_images(self) -> None:
        settings = build_video_settings(total_duration_seconds=10)
        plan = build_fallback_plan("A short story", settings=settings)
        provider = MockSpeechProvider([12.0, 8.0])

        with patch(
            "video_providers.inspect_narration_audio_file",
            side_effect=[
                NarrationAudioInfo(Path("narration.wav"), True, 100, 12.0, True, True, "pcm_s16le", "MockSpeech", "mock-model"),
                NarrationAudioInfo(Path("narration_retry.wav"), True, 100, 8.0, True, True, "pcm_s16le", "MockSpeech", "mock-model"),
            ],
        ):
            with patch("video_providers.shorten_narration_once", return_value="Shortened narration"):
                with patch("video_providers.generate_scene_images") as image_generation_mock:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        audio_path, _ = generate_narration_audio(plan=plan, output_dir=Path(temp_dir), settings=settings, speech_provider=provider)
        self.assertIsNotNone(audio_path)
        image_generation_mock.assert_not_called()

    def test_narration_target_changes_with_duration(self) -> None:
        self.assertEqual(narration_word_target(build_video_settings(total_duration_seconds=10)), (15, 22))
        self.assertEqual(narration_word_target(build_video_settings(total_duration_seconds=60)), (105, 135))

    def test_overlong_narration_retries_only_once(self) -> None:
        settings = build_video_settings(total_duration_seconds=10)
        plan = build_fallback_plan("A short story", settings=settings)
        provider = MockSpeechProvider([12.0, 8.0])

        with patch(
            "video_providers.inspect_narration_audio_file",
            side_effect=[
                NarrationAudioInfo(Path("narration.wav"), True, 100, 12.0, True, True, "pcm_s16le", "MockSpeech", "mock-model"),
                NarrationAudioInfo(Path("narration_retry.wav"), True, 100, 8.0, True, True, "pcm_s16le", "MockSpeech", "mock-model"),
            ],
        ):
            with patch("video_providers.shorten_narration_once", return_value="Shortened narration"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    audio_path, message = generate_narration_audio(plan=plan, output_dir=Path(temp_dir), settings=settings, speech_provider=provider)
                    self.assertTrue(audio_path.exists())
                    self.assertEqual(len(provider.calls), 2)
                    self.assertIn("shortened once", message.lower())

    def test_state_change_detection_for_duration(self) -> None:
        settings = build_video_settings(total_duration_seconds=15)
        snapshot = settings_snapshot(settings)
        updated = build_video_settings(total_duration_seconds=30)
        self.assertTrue(settings_changed(snapshot, updated))

    def test_state_change_detection_for_video_mode(self) -> None:
        settings = build_video_settings(video_mode="basic_motion")
        snapshot = settings_snapshot(settings)
        updated = build_video_settings(video_mode="kling_assisted")
        self.assertTrue(settings_changed(snapshot, updated))

    def test_quality_mapping_draft_maps_to_low(self) -> None:
        self.assertEqual(map_quality_label_to_openai("Draft"), "low")

    def test_select_image_provider_prefers_explicit_provider(self) -> None:
        settings = build_video_settings()
        plan = build_fallback_plan("A short story.", settings=settings)
        provider = PlaceholderImageProvider(plan=plan, settings=settings)
        selected = select_image_provider(plan=plan, settings=settings, image_provider=provider)
        self.assertIs(selected, provider)


if __name__ == "__main__":
    unittest.main()
