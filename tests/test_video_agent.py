import base64
import json
import math
import os
import shutil
import subprocess
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
from video_agent import (
    DEFAULT_VISUAL_STYLE,
    DEFAULT_IMAGE_QUALITY,
    DEFAULT_VIDEO_MODE,
    VideoScene,
    build_required_source_anchors,
    build_scene_slot_plan,
    merge_model_storyboard_with_scene_slots,
    build_creative_authority_payload,
    build_video_plan,
    build_fallback_plan,
    build_scene_durations,
    build_video_plan_from_creative_result,
    build_video_settings,
    normalize_content_type,
    narration_word_target,
    settings_changed,
    settings_snapshot,
    total_duration_seconds,
    validate_video_plan_data,
)
from video_providers import (
    AUDIBLE_AUDIO_THRESHOLD_DB,
    NarrationAudioInfo,
    OpenAISpeechProvider,
    OpenAIImageProvider,
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
    normalize_quality_label,
    select_image_provider,
    select_speech_provider,
)
from video_renderer import (
    DEFAULT_OUTPUT_FPS,
    build_generation_output_dir,
    build_scene_filter,
    build_scene_render_command,
    calculate_frame_count,
    render_scene_clip,
)
from kling_assisted import build_kling_prompt


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
        visual_style: str = "Quiet Cinematic Animation",
        premise: str,
        conflict: str,
        progression: str,
        emotional_turn: str,
        ending: str,
        scene_beats: list[str],
        use_philosophy: bool = False,
        philosophy: PhilosophyInsight | None = None,
    ) -> CreativeResult:
        return CreativeResult(
            request=CreativeRequest(
                idea=idea,
                content_type=content_type,
                tone="reflective",
                target_audience="general",
                language="English",
                duration_seconds=15,
                visual_style=visual_style,
            ),
            director=DirectorDecision(
                content_intent="human truth",
                emotional_tone="reflective",
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
                rationale="This request depends on scene-specific story beats.",
            ),
            psychology=PsychologyInsight(
                visible_behavior="Visible behavior.",
                hidden_motive="Hidden motive.",
                emotional_trigger="Trigger.",
                contradiction="Contradiction.",
                audience_rel_path="Relatable path.",
            ),
            philosophy=philosophy,
            humor=None,
            ambiguity=AmbiguityInsight(
                competing_interpretations=["One reading", "Another reading"],
                unresolved_question="What remains unresolved?",
                contradiction="Contradiction",
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
                humor_score=2,
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
            progression="The verse images move from smoke to dust to enclosure before turning toward a restless modern mind.",
            emotional_turn="A modern person notices how comparison and wanting have clouded attention.",
            ending="The disturbance settles and ordinary perception becomes clear again.",
            scene_beats=[
                "Smoke partially hides a flame.",
                "Dust obscures a mirror.",
                "Unborn child remains enclosed within the womb.",
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
        )

    def make_gita_storyboard_plan(
        self,
        scene_1_visual: str = "A low fire flickers while pale smoke passes in front of it.",
        scene_1_narration: str = "Smoke drifts over a small flame.",
        scene_2_visual: str = "A mirror slowly clouds with dust.",
        scene_2_narration: str = "Dust gathers on a mirror.",
        scene_3_visual: str = "An unborn child rests safely within the womb, enclosed in shadowed warmth.",
        scene_3_narration: str = "Unborn life remains enclosed within the womb.",
        include_title: bool = True,
        include_narration: bool = True,
        include_content_type: bool = True,
        include_aspect_ratio: bool = True,
        include_duration_seconds: bool = True,
        include_style_lock: bool = True,
    ) -> dict[str, object]:
        plan: dict[str, object] = {
            "scenes": [
                {
                    "scene_number": 1,
                    "duration_seconds": 5,
                    "narration": scene_1_narration,
                    "visual_prompt": scene_1_visual,
                    "motion_prompt": "Smoke curls slowly across the frame.",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 2,
                    "duration_seconds": 5,
                    "narration": scene_2_narration,
                    "visual_prompt": scene_2_visual,
                    "motion_prompt": "Fine dust settles while the camera moves closer.",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 3,
                    "duration_seconds": 5,
                    "narration": scene_3_narration,
                    "visual_prompt": scene_3_visual,
                    "motion_prompt": "The frame holds with a slow, protective drift.",
                    "continuity_mode": "independent",
                    "continuity_group": None,
                },
                {
                    "scene_number": 4,
                    "duration_seconds": 5,
                    "narration": "A modern person begins to scroll in quiet restlessness.",
                    "visual_prompt": "A modern person sits alone at night, scrolling comparison-driven images on a phone.",
                    "motion_prompt": "The thumb moves as the room stays still.",
                    "continuity_mode": "character",
                    "continuity_group": "human_a",
                },
                {
                    "scene_number": 5,
                    "duration_seconds": 5,
                    "narration": "Desire clouds the person's attention and then starts to loosen.",
                    "visual_prompt": "The same person lowers the phone as mental clutter softens from the face.",
                    "motion_prompt": "The shoulders release and the camera eases back.",
                    "continuity_mode": "character",
                    "continuity_group": "human_a",
                },
                {
                    "scene_number": 6,
                    "duration_seconds": 5,
                    "narration": "Ordinary sight becomes enough again.",
                    "visual_prompt": "The same person sits quietly as the room regains simple clarity.",
                    "motion_prompt": "A gentle exhale and a slight widening of the frame end the sequence.",
                    "continuity_mode": "character",
                    "continuity_group": "human_a",
                },
            ],
        }
        if include_title:
            plan["title"] = "Covered Clarity"
        if include_narration:
            plan["narration"] = "A source-faithful reflection on obscured clarity."
        if include_content_type:
            plan["content_type"] = "spiritual_reflection"
        if include_aspect_ratio:
            plan["aspect_ratio"] = "9:16"
        if include_duration_seconds:
            plan["duration_seconds"] = 30
        if include_style_lock:
            plan["style_lock"] = "Quiet cinematic animation, soft natural light, muted earthy colors, realistic proportions, restrained movement, contemplative atmosphere."
        return plan

    def make_gita_repaired_scene(
        self,
        *,
        narration: str = "Unborn life remains enclosed within the womb.",
        visual_prompt: str = "An unborn child rests safely within the womb, enclosed in shadowed warmth.",
        motion_prompt: str = "The frame holds with a slow, protective drift.",
        scene_number: int = 3,
        scene_purpose: str = "source_metaphor",
        story_anchor_id: str | None = "anchor_3",
        continuity_mode: str = "independent",
        continuity_group: str | None = None,
    ) -> dict[str, object]:
        return {
            "scene_number": scene_number,
            "narration": narration,
            "visual_prompt": visual_prompt,
            "motion_prompt": motion_prompt,
            "scene_purpose": scene_purpose,
            "story_anchor_id": story_anchor_id,
            "continuity_mode": continuity_mode,
            "continuity_group": continuity_group,
        }

    def test_basic_motion_is_default_video_mode(self) -> None:
        settings = build_video_settings()
        self.assertEqual(settings.video_mode, DEFAULT_VIDEO_MODE)

    def test_scene_durations_15_over_5(self) -> None:
        self.assertEqual(build_scene_durations(15, 5), [5, 5, 5])

    def test_scene_durations_10_over_3(self) -> None:
        self.assertEqual(build_scene_durations(10, 3), [3, 3, 3, 1])

    def test_scene_durations_30_over_4(self) -> None:
        self.assertEqual(build_scene_durations(30, 4), [4, 4, 4, 4, 4, 4, 4, 2])

    def test_scene_durations_45_over_6(self) -> None:
        self.assertEqual(build_scene_durations(45, 6), [6, 6, 6, 6, 6, 6, 6, 3])

    def test_scene_durations_sum_exactly_and_never_zero(self) -> None:
        durations = build_scene_durations(60, 6)
        self.assertEqual(sum(durations), 60)
        self.assertTrue(all(duration > 0 for duration in durations))

    def test_validate_video_plan_data_enforces_exact_scene_durations(self) -> None:
        settings = build_video_settings(total_duration_seconds=10, preferred_scene_duration_seconds=3)
        invalid_plan = {
            "title": "Invalid",
            "content_type": "nursery",
            "duration_seconds": 10,
            "aspect_ratio": "9:16",
            "narration": "Narration",
            "style_lock": "Shared style",
            "scenes": [
                {
                    "scene_number": 1,
                    "duration_seconds": 3,
                    "narration": "Scene 1",
                    "visual_prompt": "One",
                    "motion_prompt": "Still",
                },
                {
                    "scene_number": 2,
                    "duration_seconds": 3,
                    "narration": "Scene 2",
                    "visual_prompt": "Two",
                    "motion_prompt": "Still",
                },
                {
                    "scene_number": 3,
                    "duration_seconds": 3,
                    "narration": "Scene 3",
                    "visual_prompt": "Three",
                    "motion_prompt": "Still",
                },
                {
                    "scene_number": 4,
                    "duration_seconds": 2,
                    "narration": "Scene 4",
                    "visual_prompt": "Four",
                    "motion_prompt": "Still",
                },
            ],
        }

        with self.assertRaisesRegex(ValueError, "must be 1 seconds"):
            validate_video_plan_data(invalid_plan, settings=settings)

    def test_fallback_plan_respects_dynamic_timing(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=4)
        plan = build_fallback_plan("A playful nursery story", settings=settings)

        self.assertEqual(len(plan.scenes), settings.scene_count)
        self.assertEqual([scene.duration_seconds for scene in plan.scenes], settings.scene_durations)
        self.assertEqual(total_duration_seconds(plan.scenes), settings.total_duration_seconds)

    def test_build_video_settings_normalizes_new_content_type_labels(self) -> None:
        settings = build_video_settings(content_type="Human Behavior", visual_style="")
        self.assertEqual(settings.content_type, "human_behavior")
        self.assertEqual(settings.visual_style, DEFAULT_VISUAL_STYLE)

    def test_normalize_content_type_accepts_human_readable_label(self) -> None:
        self.assertEqual(normalize_content_type("Spiritual Reflection"), "spiritual_reflection")

    def test_generate_scene_images_falls_back_to_placeholder_files_in_dev_mode(self) -> None:
        settings = build_video_settings()
        plan = build_fallback_plan(
            idea="Create a nursery video about a baby elephant learning colors.",
            settings=settings,
        )

        with patch.dict("os.environ", {"VIDEO_DEV_MODE": "true"}, clear=False):
            with tempfile.TemporaryDirectory() as temp_dir:
                image_paths, messages = generate_scene_images(
                    plan=plan,
                    output_dir=Path(temp_dir),
                    settings=settings,
                )
                self.assertEqual(len(image_paths), 3)
                self.assertTrue(all(path.exists() for path in image_paths))
                self.assertIn("PlaceholderImageProvider", messages[0])
                self.assertIn("Image quality: Draft", messages[1])

    def test_narration_disabled_skips_speech_generation(self) -> None:
        settings = build_video_settings(narration_enabled=False)
        plan = build_fallback_plan("A short story", settings=settings)
        audio_path, message = generate_narration_audio(
            plan=plan,
            output_dir=Path(tempfile.mkdtemp()),
            settings=settings,
        )
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

    def test_streamlit_secrets_fallback_is_used(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("video_providers.get_streamlit_secret", return_value="secret-key"):
                self.assertEqual(get_config_value("OPENAI_API_KEY"), "secret-key")

    def test_missing_production_api_key_raises_actionable_error(self) -> None:
        settings = build_video_settings()
        with patch.dict(os.environ, {"VIDEO_DEV_MODE": "false"}, clear=True):
            load_app_config(force=True)
            provider = OpenAISpeechProvider(settings=settings, client=object())
            with self.assertRaisesRegex(
                ValueError,
                "OPENAI_API_KEY is missing. Configure it before generating real narration.",
            ):
                provider.generate_narration_audio("Hello world", Path(tempfile.mkdtemp()) / "narration.wav")

    def test_mocked_openai_tts_receives_model_voice_text_style_and_speed(self) -> None:
        settings = build_video_settings(
            voice="Warm Female",
            speaking_style="Calm",
            speaking_speed="Fast",
            language="English",
        )
        client = MockOpenAITTSClient(create_tone_wav_bytes())
        with patch.dict(
            os.environ,
            {
                "VIDEO_DEV_MODE": "false",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_TTS_MODEL": "gpt-4o-mini-tts",
            },
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
        self.assertEqual(call["response_format"], "wav")

    def test_valid_audio_file_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "narration.wav"
            audio_path.write_bytes(create_tone_wav_bytes())
            with patch("video_providers.probe_audio_file", return_value={"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}], "format": {"duration": "1.0"}}):
                with patch("video_providers.get_audio_max_volume_db", return_value=-12.0):
                    info = inspect_narration_audio_file(audio_path, provider_name="OpenAI", model_name="gpt-4o-mini-tts")
            self.assertTrue(info.contains_audible_audio)
            self.assertEqual(info.codec_name, "pcm_s16le")

    def test_empty_audio_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "empty.wav"
            audio_path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "Narration file is empty"):
                inspect_narration_audio_file(audio_path)

    def test_silent_audio_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "silent.wav"
            audio_path.write_bytes(create_tone_wav_bytes())
            with patch("video_providers.probe_audio_file", return_value={"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}], "format": {"duration": "1.0"}}):
                with patch("video_providers.get_audio_max_volume_db", return_value=-91.0):
                    with self.assertRaisesRegex(
                        ValueError,
                        "Narration was generated, but the audio is silent or invalid. Please retry narration generation.",
                    ):
                        inspect_narration_audio_file(audio_path)

    def test_negative_ninety_one_db_audio_fails_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "quiet.wav"
            audio_path.write_bytes(create_tone_wav_bytes())
            with patch("video_providers.probe_audio_file", return_value={"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}], "format": {"duration": "1.0"}}):
                with patch("video_providers.get_audio_max_volume_db", return_value=-91.0):
                    with self.assertRaisesRegex(
                        ValueError,
                        "Narration was generated, but the audio is silent or invalid. Please retry narration generation.",
                    ):
                        inspect_narration_audio_file(audio_path)

    def test_audible_audio_succeeds_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audible.wav"
            audio_path.write_bytes(create_tone_wav_bytes())
            with patch("video_providers.probe_audio_file", return_value={"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}], "format": {"duration": "1.0"}}):
                with patch("video_providers.get_audio_max_volume_db", return_value=AUDIBLE_AUDIO_THRESHOLD_DB + 1.0):
                    info = inspect_narration_audio_file(audio_path)
            self.assertTrue(info.contains_audible_audio)

    def test_non_audio_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "not_audio.wav"
            audio_path.write_bytes(b"not audio")
            with patch("video_providers.probe_audio_file", return_value={"streams": [], "format": {"duration": "1.0"}}):
                with self.assertRaisesRegex(ValueError, "does not contain a readable audio stream"):
                    inspect_narration_audio_file(audio_path)

    def test_narration_retry_does_not_regenerate_images(self) -> None:
        settings = build_video_settings(total_duration_seconds=10)
        plan = build_fallback_plan("A short story", settings=settings)
        provider = MockSpeechProvider([12.0, 8.0])

        with patch("video_providers.inspect_narration_audio_file", side_effect=[
            NarrationAudioInfo(Path("narration.wav"), True, 100, 12.0, True, True, "pcm_s16le", "MockSpeech", "mock-model"),
            NarrationAudioInfo(Path("narration_retry.wav"), True, 100, 8.0, True, True, "pcm_s16le", "MockSpeech", "mock-model"),
        ]):
            with patch("video_providers.shorten_narration_once", return_value="Shortened narration"):
                with patch("video_providers.generate_scene_images") as image_generation_mock:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        audio_path, _ = generate_narration_audio(
                            plan=plan,
                            output_dir=Path(temp_dir),
                            settings=settings,
                            speech_provider=provider,
                        )
        self.assertIsNotNone(audio_path)
        image_generation_mock.assert_not_called()

    def test_narration_target_changes_with_duration(self) -> None:
        self.assertEqual(
            narration_word_target(build_video_settings(total_duration_seconds=10)),
            (15, 22),
        )
        self.assertEqual(
            narration_word_target(build_video_settings(total_duration_seconds=60)),
            (105, 135),
        )

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
                    audio_path, message = generate_narration_audio(
                        plan=plan,
                        output_dir=Path(temp_dir),
                        settings=settings,
                        speech_provider=provider,
                    )
                    self.assertTrue(audio_path.exists())
                    self.assertEqual(len(provider.calls), 2)
                    self.assertIn("shortened once", message.lower())

    def test_state_change_detection_for_duration(self) -> None:
        settings = build_video_settings(total_duration_seconds=15)
        snapshot = settings_snapshot(settings)
        updated = build_video_settings(total_duration_seconds=30)
        self.assertTrue(settings_changed(snapshot, updated))

    def test_state_change_detection_for_scene_duration(self) -> None:
        settings = build_video_settings(preferred_scene_duration_seconds=5)
        snapshot = settings_snapshot(settings)
        updated = build_video_settings(preferred_scene_duration_seconds=4)
        self.assertTrue(settings_changed(snapshot, updated))

    def test_state_change_detection_for_fps(self) -> None:
        settings = build_video_settings(frame_rate=30)
        snapshot = settings_snapshot(settings)
        updated = build_video_settings(frame_rate=24)
        self.assertTrue(settings_changed(snapshot, updated))

    def test_state_change_detection_for_voice_settings(self) -> None:
        settings = build_video_settings(voice="Warm Female")
        snapshot = settings_snapshot(settings)
        updated = build_video_settings(voice="Warm Male")
        self.assertTrue(settings_changed(snapshot, updated))

    def test_state_change_detection_for_video_mode(self) -> None:
        settings = build_video_settings(video_mode="basic_motion")
        snapshot = settings_snapshot(settings)
        updated = build_video_settings(video_mode="kling_assisted")
        self.assertTrue(settings_changed(snapshot, updated))

    def test_quality_mapping_draft_maps_to_low(self) -> None:
        self.assertEqual(map_quality_label_to_openai("Draft"), "low")

    def test_quality_mapping_standard_maps_to_medium(self) -> None:
        self.assertEqual(map_quality_label_to_openai("Standard"), "medium")

    def test_quality_mapping_final_maps_to_high(self) -> None:
        self.assertEqual(map_quality_label_to_openai("Final"), "high")

    def test_quality_mapping_invalid_defaults_to_low(self) -> None:
        self.assertEqual(normalize_quality_label("unexpected"), "Draft")
        self.assertEqual(map_quality_label_to_openai("unexpected"), "low")

    def test_select_image_provider_uses_placeholder_in_dev_mode(self) -> None:
        settings = build_video_settings()
        plan = build_fallback_plan("A nursery story", settings=settings)
        with patch.dict("os.environ", {"VIDEO_DEV_MODE": "true"}, clear=False):
            provider = select_image_provider(plan=plan, settings=settings)
        self.assertIsInstance(provider, PlaceholderImageProvider)

    def test_select_image_provider_uses_openai_in_production_mode(self) -> None:
        settings = build_video_settings(image_quality="Draft")
        plan = build_fallback_plan("A nursery story", settings=settings)
        with patch.dict("os.environ", {"VIDEO_DEV_MODE": "false", "OPENAI_API_KEY": "test-key"}, clear=False):
            provider = select_image_provider(plan=plan, settings=settings)
        self.assertIsInstance(provider, OpenAIImageProvider)
        self.assertEqual(provider.quality_label, DEFAULT_IMAGE_QUALITY)

    def test_openai_image_provider_receives_selected_quality(self) -> None:
        settings = build_video_settings(image_quality="Final")
        plan = build_fallback_plan("A nursery story", settings=settings)
        scene = plan.scenes[0]
        client = MockOpenAIClient()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            provider = OpenAIImageProvider(plan=plan, settings=settings, client=client)
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "scene_01.png"
                result = provider.generate_scene_image(scene=scene, output_path=output_path)
                self.assertTrue(result.path.exists())
                self.assertEqual(client.images.generate_calls[0]["quality"], "high")

    def test_openai_image_provider_uses_group_reference_for_later_scenes_only(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5)
        plan = build_fallback_plan("A nursery story", settings=settings)
        plan.scenes[0].continuity_mode = "character"
        plan.scenes[0].continuity_group = "human_a"
        plan.scenes[1].continuity_mode = "character"
        plan.scenes[1].continuity_group = "human_a"
        client = MockOpenAIClient()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            provider = OpenAIImageProvider(plan=plan, settings=settings, client=client)
            with tempfile.TemporaryDirectory() as temp_dir:
                first_path = Path(temp_dir) / "scene_01.png"
                second_path = Path(temp_dir) / "scene_02.png"
                provider.generate_scene_image(scene=plan.scenes[0], output_path=first_path)
                provider.generate_scene_image(scene=plan.scenes[1], output_path=second_path)

        self.assertEqual(len(client.images.generate_calls), 1)
        self.assertEqual(len(client.images.edit_calls), 1)
        self.assertEqual(client.images.edit_calls[0]["quality"], "low")
        self.assertEqual(client.images.edit_calls[0]["input_fidelity"], "high")

    def test_openai_image_provider_keeps_independent_scenes_fresh(self) -> None:
        settings = build_video_settings(total_duration_seconds=15, preferred_scene_duration_seconds=5)
        plan = build_fallback_plan("A symbolic reflection", settings=settings)
        for scene in plan.scenes:
            scene.continuity_mode = "independent"
            scene.continuity_group = None
        client = MockOpenAIClient()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            provider = OpenAIImageProvider(plan=plan, settings=settings, client=client)
            with tempfile.TemporaryDirectory() as temp_dir:
                for scene in plan.scenes:
                    provider.generate_scene_image(
                        scene=scene,
                        output_path=Path(temp_dir) / f"scene_{scene.scene_number:02d}.png",
                    )

        self.assertEqual(len(client.images.generate_calls), 3)
        self.assertEqual(len(client.images.edit_calls), 0)

    def test_build_openai_image_prompt_includes_style_lock_and_scene_details(self) -> None:
        settings = build_video_settings()
        plan = build_fallback_plan("A nursery story", settings=settings)
        scene = plan.scenes[0]
        scene.continuity_mode = "character"
        scene.continuity_group = "human_a"
        prompt = build_openai_image_prompt(plan=plan, scene=scene)
        self.assertIn(plan.style_lock, prompt)
        self.assertIn(scene.visual_prompt, prompt)
        self.assertIn(scene.motion_prompt, prompt)
        self.assertIn("hairstyle", prompt.lower())
        self.assertIn("hair length", prompt.lower())
        self.assertIn("continuity_group=human_a", prompt)
        self.assertNotIn("large expressive eyes", prompt.lower())
        self.assertNotIn("bright pastel colors", prompt.lower())

    def test_build_openai_image_prompt_keeps_provider_neutral_for_philosophy(self) -> None:
        settings = build_video_settings(content_type="philosophy", visual_style="Quiet Cinematic Animation")
        plan = build_fallback_plan("A quiet reflection on impermanence", settings=settings)
        prompt = build_openai_image_prompt(plan=plan, scene=plan.scenes[0])
        self.assertNotIn("nursery", prompt.lower())
        self.assertIn("style lock", prompt.lower())

    def test_build_openai_image_prompt_allows_nursery_via_style_lock(self) -> None:
        settings = build_video_settings(content_type="nursery", visual_style="3D Nursery Animation")
        plan = build_fallback_plan("A nursery story", settings=settings)
        prompt = build_openai_image_prompt(plan=plan, scene=plan.scenes[0])
        self.assertIn("3d nursery animation", prompt.lower())

    def test_build_video_plan_from_creative_result_returns_valid_plan(self) -> None:
        settings = build_video_settings(total_duration_seconds=15, preferred_scene_duration_seconds=5, content_type="story")
        creative_result = CreativeResult(
            request=CreativeRequest(
                idea="A reflective short story about status anxiety",
                content_type="story",
                tone="reflective",
                target_audience="general",
                language="English",
                duration_seconds=15,
                visual_style="Quiet Cinematic Animation",
            ),
            director=DirectorDecision(
                content_intent="human truth",
                emotional_tone="reflective",
                narrative_shape="contradiction_to_reveal",
                use_psychology=True,
                use_philosophy=False,
                use_humor=False,
                use_ambiguity=True,
                humor_level="off",
                philosophy_level="light",
                psychology_level="high",
                ambiguity_level="balanced",
                story_focus="show contradiction through behavior",
                rationale="The request is behavior-driven.",
            ),
            psychology=PsychologyInsight(
                visible_behavior="He boasts casually.",
                hidden_motive="He wants reassurance.",
                emotional_trigger="Comparison.",
                contradiction="He dismisses approval while seeking it.",
                audience_rel_path="People recognize this instantly.",
            ),
            philosophy=PhilosophyInsight(
                central_question="",
                deeper_meaning="",
                tension="",
                possible_closing_thought="",
                avoid_preaching="yes",
            ),
            humor=None,
            ambiguity=AmbiguityInsight(
                competing_interpretations=["Status anxiety", "Nervous habit"],
                unresolved_question="Does he know he is doing it?",
                contradiction="Confidence mixed with insecurity",
                what_not_to_explain="Do not declare one motive as final truth.",
                ambiguity_strength="balanced",
            ),
            story=StoryDraft(
                premise="A man performs confidence while tracking approval.",
                conflict="His actions contradict his words.",
                progression="He checks reactions more and more often.",
                emotional_turn="He catches himself in the act.",
                ending="The silence says more than the speech.",
                scene_beats=["Claim indifference", "Check reactions", "Reveal contradiction"],
            ),
            critic=CreativeEvaluation(
                relatability_score=8,
                humor_score=4,
                psychological_truth_score=9,
                philosophical_depth_score=3,
                ambiguity_score=7,
                preachiness_score=2,
                originality_score=7,
                clarity_score=8,
                forced_humor_score=1,
                unnecessary_explanation_score=2,
                notes="Keep it concise.",
                edit_instructions=["Trim obvious lines", "Let the final image carry the ending"],
            ),
            final_story=StoryDraft(
                premise="A man says he does not care what people think, then checks who watched his status.",
                conflict="His self-image and behavior clash.",
                progression="Each refresh exposes more neediness.",
                emotional_turn="He notices his own contradiction.",
                ending="The phone glow becomes the punchline.",
                scene_beats=["Claim indifference", "Repeated checking", "Quiet self-reveal"],
            ),
            selected_specialists=["psychology", "ambiguity", "story", "critic"],
        )
        raw_plan = {
            "title": "Status Check",
            "content_type": "story",
            "duration_seconds": 15,
            "aspect_ratio": "9:16",
            "narration": "He says he does not care what people think. Then he checks who watched his status again.",
            "style_lock": "Quiet Cinematic Animation. Keep the same man, phone, outfit, and apartment lighting in every scene.",
            "scenes": [
                {
                    "scene_number": 1,
                    "duration_seconds": 5,
                    "narration": "He shrugs and says other opinions mean nothing to him.",
                    "visual_prompt": "A man acts casual in his apartment, phone face down on the table.",
                    "motion_prompt": "Gentle push in.",
                },
                {
                    "scene_number": 2,
                    "duration_seconds": 5,
                    "narration": "Seconds later, his hand flips the phone to check who viewed his status.",
                    "visual_prompt": "The same man refreshes his WhatsApp status viewers with anxious focus.",
                    "motion_prompt": "Subtle zoom on the glowing phone.",
                },
                {
                    "scene_number": 3,
                    "duration_seconds": 5,
                    "narration": "He catches himself doing it and says nothing.",
                    "visual_prompt": "The room goes quiet as he stares at the screen and then at himself.",
                    "motion_prompt": "Hold on his expression.",
                },
            ],
        }

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(len(plan.scenes), settings.scene_count)
        self.assertEqual([scene.duration_seconds for scene in plan.scenes], settings.scene_durations)
        self.assertEqual(plan.content_type, "story")

    def test_gita_style_symbolic_sequence_preserves_source_metaphors_and_independence(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(
            scene_1_visual="A low fire flickers while pale smoke passes in front of it and softens its light.",
            scene_1_narration="Smoke drifts over a small flame and dims what was visible a moment ago.",
            scene_2_visual="A mirror on a wall slowly clouds with dust until the reflection becomes smeared and hard to read.",
            scene_2_narration="Dust gathers on a mirror until the reflected face loses definition.",
            scene_3_visual="An unborn child rests inside a dark protective womb, enclosed and unseen from the outside world.",
            scene_3_narration="Life remains enclosed, present but hidden from open sight.",
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        prompts = " ".join(scene.visual_prompt.lower() for scene in plan.scenes)
        self.assertIn("fire", prompts)
        self.assertIn("mirror", prompts)
        self.assertIn("womb", prompts)
        self.assertEqual(plan.scenes[0].continuity_mode, "independent")
        self.assertEqual(plan.scenes[1].continuity_mode, "independent")
        self.assertEqual(plan.scenes[2].continuity_mode, "independent")
        self.assertNotIn("glowing orb", prompts)
        self.assertFalse(any("next visual beat" in scene.visual_prompt.lower() for scene in plan.scenes))
        self.assertEqual(plan.scenes[0].scene_purpose, "source_metaphor")
        self.assertEqual(plan.scenes[0].source_anchor_id, "anchor_1")
        self.assertEqual(plan.scenes[1].source_anchor_id, "anchor_2")
        self.assertEqual(plan.scenes[2].source_anchor_id, "anchor_3")
        self.assertEqual(plan.scenes[3].scene_purpose, "modern_reflection")
        self.assertEqual(plan.scenes[5].scene_purpose, "conclusion")

    def test_multimind_missing_title_uses_fallback_title(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_title=False, include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertTrue(plan.title)
        self.assertNotEqual(plan.title.strip(), "")

    def test_multimind_missing_top_level_narration_assembles_from_scenes(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_narration=False, include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(plan.narration, " ".join(scene.narration for scene in plan.scenes))

    def test_multimind_missing_content_type_uses_settings_content_type(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(plan.content_type, settings.content_type)

    def test_multimind_missing_aspect_ratio_uses_settings_aspect_ratio(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(plan.aspect_ratio, settings.aspect_ratio)

    def test_multimind_valid_scenes_plus_deterministic_metadata_build_video_plan(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(plan.duration_seconds, settings.total_duration_seconds)
        self.assertEqual([scene.duration_seconds for scene in plan.scenes], settings.scene_durations)
        self.assertTrue(plan.style_lock)

    def test_ordered_required_story_anchors_preserve_source_order(self) -> None:
        source_metaphors = [
            "smoke covering fire",
            "dust covering a mirror",
            "unborn life enclosed within the womb",
        ]
        anchors = build_required_source_anchors(source_metaphors)
        scene_slots = build_scene_slot_plan(anchors, scene_count=6)

        self.assertEqual(
            [(anchor.id, anchor.description, anchor.source_order) for anchor in anchors],
            [
                ("anchor_1", "smoke covering fire", 1),
                ("anchor_2", "dust covering a mirror", 2),
                ("anchor_3", "unborn life enclosed within the womb", 3),
            ],
        )
        self.assertEqual(
            [(slot.scene_number, slot.source_anchor_id) for slot in scene_slots[:3]],
            [(1, "anchor_1"), (2, "anchor_2"), (3, "anchor_3")],
        )

    def test_reordered_dict_insertion_does_not_change_scene_assignments(self) -> None:
        anchors = build_required_source_anchors(
            [
                "smoke covering fire",
                "dust covering a mirror",
                "unborn life enclosed within the womb",
            ]
        )
        scene_slots = build_scene_slot_plan(anchors, scene_count=6)
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)
        raw_plan["scenes"][0]["story_anchor_id"] = "anchor_3"
        raw_plan["scenes"][1]["story_anchor_id"] = "anchor_1"
        merged = merge_model_storyboard_with_scene_slots(raw_plan, scene_slots)

        self.assertEqual(
            [scene["story_anchor_id"] for scene in merged["scenes"][:3]],
            ["anchor_1", "anchor_2", "anchor_3"],
        )

    def test_multimind_malformed_scene_payload_retries_storyboard_generation(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        malformed = {"title": "Broken", "scenes": [{"scene_number": 1, "narration": "Only one scene"}]}
        corrected = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)

        with patch("video_agent.request_video_plan_from_ollama", side_effect=[json.dumps(malformed), json.dumps(corrected)]) as plan_mock:
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(plan_mock.call_count, 2)
        self.assertEqual(len(plan.scenes), 6)

    def test_multimind_missing_visual_prompt_repairs_only_scene_one(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)
        raw_plan["scenes"][0]["visual_prompt"] = ""
        repaired_scene = self.make_gita_repaired_scene(
            scene_number=1,
            narration=raw_plan["scenes"][0]["narration"],
            visual_prompt="A small flame burns in a dark room while dense smoke drifts across and partially obscures it.",
            motion_prompt=raw_plan["scenes"][0]["motion_prompt"],
            scene_purpose="source_metaphor",
            story_anchor_id="anchor_1",
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)) as plan_mock:
            with patch("video_agent.request_scene_field_repair_from_ollama", return_value=json.dumps(repaired_scene)) as repair_mock:
                with patch("video_agent.request_scene_repair_from_ollama") as anchor_repair_mock:
                    plan, warning = build_video_plan_from_creative_result(
                        idea=creative_result.request.idea,
                        creative_result=creative_result,
                        settings=settings,
                    )

        self.assertIsNone(warning)
        self.assertEqual(plan_mock.call_count, 1)
        self.assertEqual(repair_mock.call_count, 1)
        self.assertEqual(anchor_repair_mock.call_count, 0)
        self.assertIn("smoke", plan.scenes[0].visual_prompt.lower())
        self.assertEqual(plan.scenes[0].source_anchor_id, "anchor_1")
        self.assertNotEqual(plan.scenes[0].source_anchor_id, "anchor_3")

    def test_multimind_missing_motion_prompt_uses_scene_aware_fallback(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)
        raw_plan["scenes"][2]["motion_prompt"] = ""

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_scene_field_repair_from_ollama") as field_repair_mock:
                plan, warning = build_video_plan_from_creative_result(
                    idea=creative_result.request.idea,
                    creative_result=creative_result,
                    settings=settings,
                )

        self.assertIsNone(warning)
        self.assertEqual(field_repair_mock.call_count, 0)
        self.assertIn("protective", plan.scenes[2].motion_prompt.lower())

    def test_multimind_other_scenes_remain_unchanged_during_scene_one_field_repair(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)
        raw_plan["scenes"][0]["visual_prompt"] = ""
        original_scene_2 = dict(raw_plan["scenes"][1])
        original_scene_3 = dict(raw_plan["scenes"][2])
        repaired_scene = self.make_gita_repaired_scene(
            scene_number=1,
            narration=raw_plan["scenes"][0]["narration"],
            visual_prompt="A small flame burns in a dark room while dense smoke drifts across and partially obscures it.",
            motion_prompt=raw_plan["scenes"][0]["motion_prompt"],
            scene_purpose="source_metaphor",
            story_anchor_id="anchor_1",
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_scene_field_repair_from_ollama", return_value=json.dumps(repaired_scene)):
                plan, warning = build_video_plan_from_creative_result(
                    idea=creative_result.request.idea,
                    creative_result=creative_result,
                    settings=settings,
                )

        self.assertIsNone(warning)
        self.assertEqual(plan.scenes[1].narration, original_scene_2["narration"])
        self.assertEqual(plan.scenes[1].motion_prompt, original_scene_2["motion_prompt"])
        self.assertIn(original_scene_2["visual_prompt"].lower(), plan.scenes[1].visual_prompt.lower())
        self.assertEqual(plan.scenes[2].narration, original_scene_3["narration"])

    def test_multimind_source_anchor_survives_scene_field_repair(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)
        raw_plan["scenes"][0]["visual_prompt"] = ""
        repaired_scene = self.make_gita_repaired_scene(
            scene_number=1,
            narration="Smoke drifts over a small flame.",
            visual_prompt="A small flame burns in a dark room while dense smoke drifts across and partially obscures it.",
            motion_prompt="Smoke curls slowly across the frame.",
            scene_purpose="source_metaphor",
            story_anchor_id="anchor_1",
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_scene_field_repair_from_ollama", return_value=json.dumps(repaired_scene)):
                plan, warning = build_video_plan_from_creative_result(
                    idea=creative_result.request.idea,
                    creative_result=creative_result,
                    settings=settings,
                )

        self.assertIsNone(warning)
        self.assertEqual(plan.scenes[0].source_anchor_id, "anchor_1")
        self.assertEqual(plan.scenes[0].scene_purpose, "source_metaphor")

    def test_scene_one_is_never_validated_against_womb_anchor(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(
            scene_1_visual="A small flame burns in a dark room while dense smoke drifts across and partially obscures it.",
            scene_1_narration="Smoke curls upward from a dormant fire, its tendrils thickening until the flame is swallowed.",
            include_content_type=False,
            include_aspect_ratio=False,
            include_duration_seconds=False,
            include_style_lock=False,
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(plan.scenes[0].source_anchor_id, "anchor_1")
        self.assertNotIn("womb", plan.scenes[0].visual_prompt.lower())

    def test_scene_three_womb_validates_against_anchor_three(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(plan.scenes[2].source_anchor_id, "anchor_3")
        self.assertIn("womb", plan.scenes[2].visual_prompt.lower())

    def test_multimind_generic_placeholder_visual_prompt_is_rejected(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)
        raw_plan["scenes"][0]["visual_prompt"] = "A specific middle image showing one visible action, object, or environmental change."

        with patch("video_agent.request_video_plan_from_ollama", side_effect=[json.dumps(raw_plan), json.dumps(raw_plan)]) as plan_mock:
            with self.assertRaisesRegex(ValueError, "Scene 1 visual prompt is too generic"):
                build_video_plan_from_creative_result(
                    idea=creative_result.request.idea,
                    creative_result=creative_result,
                    settings=settings,
                )

        self.assertEqual(plan_mock.call_count, 2)

    def test_multimind_scene_field_repairs_fail_after_two_attempts(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)
        raw_plan["scenes"][0]["visual_prompt"] = ""
        failed_scene = self.make_gita_repaired_scene(
            scene_number=1,
            narration=raw_plan["scenes"][0]["narration"],
            visual_prompt="",
            motion_prompt=raw_plan["scenes"][0]["motion_prompt"],
            scene_purpose="source_metaphor",
            story_anchor_id="anchor_1",
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_scene_field_repair_from_ollama", side_effect=[json.dumps(failed_scene), json.dumps(failed_scene)]) as repair_mock:
                with self.assertRaisesRegex(
                    ValueError,
                    "Scene-specific field repair failed after 2 attempts for Scene 1 missing visual_prompt",
                ):
                    build_video_plan_from_creative_result(
                        idea=creative_result.request.idea,
                        creative_result=creative_result,
                        settings=settings,
                    )

        self.assertEqual(repair_mock.call_count, 2)

    def test_multimind_complete_storyboard_skips_scene_field_repair(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_scene_field_repair_from_ollama") as repair_mock:
                plan, warning = build_video_plan_from_creative_result(
                    idea=creative_result.request.idea,
                    creative_result=creative_result,
                    settings=settings,
                )

        self.assertIsNone(warning)
        self.assertEqual(repair_mock.call_count, 0)
        self.assertEqual(len(plan.scenes), 6)

    def test_gita_style_seed_in_jar_repairs_scene_three_only_and_passes(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(
            scene_3_visual="A seed glows inside a clay jar with cracks spreading through it.",
            scene_3_narration="A seed warms inside a clay jar until the jar cracks open.",
        )
        repaired_scene = self.make_gita_repaired_scene()

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_scene_repair_from_ollama", return_value=json.dumps(repaired_scene)) as repair_mock:
                plan, warning = build_video_plan_from_creative_result(
                    idea=creative_result.request.idea,
                    creative_result=creative_result,
                    settings=settings,
                )

        self.assertIsNone(warning)
        self.assertEqual(repair_mock.call_count, 1)
        self.assertIn("womb", plan.scenes[2].visual_prompt.lower())
        self.assertIn("unborn", plan.scenes[2].narration.lower())

    def test_scene_one_repair_receives_anchor_one(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)
        raw_plan["scenes"][0]["visual_prompt"] = ""

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_scene_field_repair_from_ollama", return_value=json.dumps(self.make_gita_repaired_scene(
                scene_number=1,
                narration=raw_plan["scenes"][0]["narration"],
                visual_prompt="A small flame burns in a dark room while dense smoke drifts across and partially obscures it.",
                motion_prompt=raw_plan["scenes"][0]["motion_prompt"],
                scene_purpose="source_metaphor",
                story_anchor_id="anchor_1",
            ))) as repair_mock:
                plan, warning = build_video_plan_from_creative_result(
                    idea=creative_result.request.idea,
                    creative_result=creative_result,
                    settings=settings,
                )

        self.assertIsNone(warning)
        self.assertEqual(repair_mock.call_args.kwargs["scene_slot"].source_anchor_id, "anchor_1")
        self.assertEqual(repair_mock.call_args.kwargs["anchor"].description, "smoke covering fire")
        self.assertEqual(plan.scenes[0].source_anchor_id, "anchor_1")

    def test_scene_three_repair_receives_anchor_three(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(
            scene_3_visual="A seed glows inside a clay jar with cracks spreading through it.",
            scene_3_narration="A seed warms inside a clay jar until the jar cracks open.",
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_scene_repair_from_ollama", return_value=json.dumps(self.make_gita_repaired_scene())) as repair_mock:
                plan, warning = build_video_plan_from_creative_result(
                    idea=creative_result.request.idea,
                    creative_result=creative_result,
                    settings=settings,
                )

        self.assertIsNone(warning)
        self.assertEqual(repair_mock.call_args.kwargs["scene_slot"].source_anchor_id, "anchor_3")
        self.assertEqual(repair_mock.call_args.kwargs["anchor"].description, "unborn life enclosed within the womb")
        self.assertEqual(plan.scenes[2].source_anchor_id, "anchor_3")

    def test_model_output_attempting_to_change_story_anchor_id_is_ignored(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)
        raw_plan["scenes"][0]["story_anchor_id"] = "anchor_3"
        raw_plan["scenes"][2]["story_anchor_id"] = "anchor_1"

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(
            [scene.source_anchor_id for scene in plan.scenes[:3]],
            ["anchor_1", "anchor_2", "anchor_3"],
        )

    def test_serialization_deserialization_preserves_anchor_ids(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(include_content_type=False, include_aspect_ratio=False, include_duration_seconds=False, include_style_lock=False)
        serialized = json.dumps(raw_plan)
        parsed = json.loads(serialized)

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(parsed)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(
            [scene.source_anchor_id for scene in plan.scenes[:3]],
            ["anchor_1", "anchor_2", "anchor_3"],
        )

    def test_gita_style_source_metaphors_pass_when_split_across_independent_scenes(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan()

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertIn("smoke", plan.scenes[0].visual_prompt.lower())
        self.assertIn("mirror", plan.scenes[1].visual_prompt.lower())
        self.assertIn("womb", plan.scenes[2].visual_prompt.lower())

    def test_gita_style_scene_one_does_not_need_to_contain_all_source_metaphors(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(
            scene_1_visual="A flame is briefly obscured by smoke in a dark still frame.",
            scene_1_narration="Smoke veils a flame for a moment.",
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertNotIn("womb", plan.scenes[0].visual_prompt.lower())

    def test_gita_style_glowing_orb_repairs_scene_three_only_and_passes(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(
            scene_3_visual="A glowing orb hangs inside a dark cocoon-like chamber.",
            scene_3_narration="A glowing orb waits inside a sealed chamber.",
        )
        repaired_scene = self.make_gita_repaired_scene(
            narration="An unborn child stays enclosed within the womb.",
            visual_prompt="An unborn child remains enclosed within the womb in protective stillness.",
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_scene_repair_from_ollama", return_value=json.dumps(repaired_scene)) as repair_mock:
                plan, warning = build_video_plan_from_creative_result(
                    idea=creative_result.request.idea,
                    creative_result=creative_result,
                    settings=settings,
                )

        self.assertIsNone(warning)
        self.assertEqual(repair_mock.call_count, 1)
        self.assertNotIn("glowing orb", plan.scenes[2].visual_prompt.lower())

    def test_gita_style_other_valid_scenes_remain_unchanged_during_repair(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(
            scene_3_visual="A seed glows inside a clay jar with cracks spreading through it.",
            scene_3_narration="A seed warms inside a clay jar until the jar cracks open.",
        )
        original_scene_1 = dict(raw_plan["scenes"][0])
        original_scene_2 = dict(raw_plan["scenes"][1])
        original_scene_4 = dict(raw_plan["scenes"][3])

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_scene_repair_from_ollama", return_value=json.dumps(self.make_gita_repaired_scene())):
                plan, warning = build_video_plan_from_creative_result(
                    idea=creative_result.request.idea,
                    creative_result=creative_result,
                    settings=settings,
                )

        self.assertIsNone(warning)
        self.assertEqual(plan.scenes[0].narration, original_scene_1["narration"])
        self.assertIn(original_scene_2["visual_prompt"].lower(), plan.scenes[1].visual_prompt.lower())
        self.assertIn(original_scene_4["visual_prompt"].lower(), plan.scenes[3].visual_prompt.lower())

    def test_gita_style_failed_repairs_after_two_attempts_raise_explicit_error(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(
            scene_3_visual="A seed glows inside a clay jar with cracks spreading through it.",
            scene_3_narration="A seed warms inside a clay jar until the jar cracks open.",
        )
        failed_repair = self.make_gita_repaired_scene(
            narration="A seed keeps warming in a hidden jar.",
            visual_prompt="A seed remains inside a sealed clay jar.",
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            with patch("video_agent.request_scene_repair_from_ollama", side_effect=[json.dumps(failed_repair), json.dumps(failed_repair)]) as repair_mock:
                with self.assertRaisesRegex(
                    ValueError,
                    "Scene-specific repair failed after 2 attempts for Scene 3",
                ):
                    build_video_plan_from_creative_result(
                        idea=creative_result.request.idea,
                        creative_result=creative_result,
                        settings=settings,
                    )

        self.assertEqual(repair_mock.call_count, 2)

    def test_gita_style_required_source_anchor_survives_even_if_story_beats_omit_it(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        creative_result = CreativeResult(
            **{
                **creative_result.__dict__,
                "final_story": StoryDraft(
                    premise=creative_result.final_story.premise,
                    conflict=creative_result.final_story.conflict,
                    progression=creative_result.final_story.progression,
                    emotional_turn=creative_result.final_story.emotional_turn,
                    ending=creative_result.final_story.ending,
                    scene_beats=[
                        "Smoke partially hides a flame.",
                        "Dust obscures a mirror.",
                        "A modern person becomes mentally restless while scrolling comparison-driven images.",
                        "The person's attention clouds with desire and then slowly settles.",
                        "The room becomes still as clear attention returns.",
                        "Ordinary sight becomes enough again.",
                    ],
                ),
            }
        )
        raw_plan = self.make_gita_storyboard_plan()

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(plan.scenes[2].scene_purpose, "source_metaphor")
        self.assertEqual(plan.scenes[2].source_anchor_id, "anchor_3")
        self.assertIn("womb", plan.scenes[2].visual_prompt.lower())

    def test_gita_style_semantic_paraphrases_pass_source_validation(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(
            scene_1_visual="A flame is obscured by smoke in a quiet dark space.",
            scene_1_narration="A flame is briefly hidden behind smoke.",
            scene_2_visual="A dust-coated mirror loses its clear reflection.",
            scene_2_narration="Dust settles across a mirror until it dulls.",
            scene_3_visual="An unborn child remains enclosed within the womb in protective stillness.",
            scene_3_narration="An unborn child stays enclosed within the womb.",
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(len(plan.scenes), 6)

    def test_gita_style_required_source_anchor_metadata_survives_into_video_plan(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5, content_type="spiritual_reflection")
        creative_result = self.make_gita_creative_result()
        payload = build_creative_authority_payload(creative_result, settings)
        raw_plan = self.make_gita_storyboard_plan()

        self.assertEqual(
            payload["source_fidelity"]["required_source_anchors"],
            [
                {
                    "id": "anchor_1",
                    "meaning": "smoke covering fire",
                    "required_objects": ["smoke", "fire"],
                    "source_order": 1,
                    "allowed_depictions": [],
                    "forbidden_replacements": ["glowing orb"],
                },
                {
                    "id": "anchor_2",
                    "meaning": "dust covering a mirror",
                    "required_objects": ["dust", "mirror"],
                    "source_order": 2,
                    "allowed_depictions": [],
                    "forbidden_replacements": ["pool of water"],
                },
                {
                    "id": "anchor_3",
                    "meaning": "unborn life enclosed within the womb",
                    "required_objects": ["unborn life", "womb"],
                    "source_order": 3,
                    "allowed_depictions": [
                        "warm abstract womb environment",
                        "fetal silhouette",
                        "unborn life enclosed in a maternal form",
                        "non-medical symbolic depiction",
                    ],
                    "forbidden_replacements": ["seed", "clay jar", "cocoon", "glowing orb", "egg", "closed flower", "container"],
                },
            ],
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        self.assertEqual(
            [(scene.scene_purpose, scene.source_anchor_id) for scene in plan.scenes],
            [
                ("source_metaphor", "anchor_1"),
                ("source_metaphor", "anchor_2"),
                ("source_metaphor", "anchor_3"),
                ("modern_reflection", None),
                ("modern_reflection", None),
                ("conclusion", None),
            ],
        )

    def test_multimind_video_plan_remains_compatible_with_kling_prompt_generation(self) -> None:
        settings = build_video_settings(
            total_duration_seconds=30,
            preferred_scene_duration_seconds=5,
            content_type="spiritual_reflection",
            video_mode="kling_assisted",
        )
        creative_result = self.make_gita_creative_result()
        raw_plan = self.make_gita_storyboard_plan(
            include_content_type=False,
            include_aspect_ratio=False,
            include_duration_seconds=False,
            include_style_lock=False,
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value=json.dumps(raw_plan)):
            plan, warning = build_video_plan_from_creative_result(
                idea=creative_result.request.idea,
                creative_result=creative_result,
                settings=settings,
            )

        self.assertIsNone(warning)
        prompt = build_kling_prompt(plan.scenes[0], settings)
        self.assertIn(plan.scenes[0].motion_prompt, prompt)
        self.assertIn(plan.scenes[0].visual_prompt, prompt)

    def test_character_comedy_scenes_share_character_continuity_group(self) -> None:
        settings = build_video_settings(total_duration_seconds=15, preferred_scene_duration_seconds=5, content_type="humor")
        creative_result = self.make_creative_result(
            idea="A dry comedy about a couple pretending not to argue during dinner.",
            content_type="humor",
            premise="A couple insist dinner is calm while every gesture says otherwise.",
            conflict="They deny the argument while escalating it.",
            progression="Small domestic actions reveal growing tension.",
            emotional_turn="They realize they are fighting through politeness.",
            ending="A shared laugh breaks the pose.",
            scene_beats=[
                "A couple set plates down too carefully.",
                "The same couple pass dishes without making eye contact.",
                "The same couple finally laugh at the absurd politeness.",
            ],
        )
        payload = build_creative_authority_payload(creative_result, settings)
        scene_hints = payload["continuity_requirements"]["scene_hints"]

        self.assertEqual(
            [hint["continuity_group"] for hint in scene_hints],
            ["human_a", "human_a", "human_a"],
        )
        self.assertTrue(all(hint["continuity_mode"] == "character" for hint in scene_hints))

    def test_mixed_story_uses_independent_then_character_continuity(self) -> None:
        settings = build_video_settings(total_duration_seconds=15, preferred_scene_duration_seconds=5, content_type="story")
        creative_result = self.make_creative_result(
            idea="A city sunrise opens a story before one commuter carries the rest of it.",
            premise="The city wakes before one commuter enters the frame.",
            conflict="The commuter feels swallowed by the day ahead.",
            progression="The city scene gives way to a specific person and then stays with them.",
            emotional_turn="The commuter pauses and regains composure.",
            ending="The walk continues with steadier focus.",
            scene_beats=[
                "Sunrise spreads over empty city buildings.",
                "A commuter hurries through the station with coffee in hand.",
                "The same commuter stops, breathes, and walks on.",
            ],
        )
        payload = build_creative_authority_payload(creative_result, settings)
        scene_hints = payload["continuity_requirements"]["scene_hints"]

        self.assertEqual(scene_hints[0]["continuity_mode"], "independent")
        self.assertIsNone(scene_hints[0]["continuity_group"])
        self.assertEqual(scene_hints[1]["continuity_mode"], "character")
        self.assertEqual(scene_hints[1]["continuity_group"], "human_a")
        self.assertEqual(scene_hints[2]["continuity_mode"], "character")
        self.assertEqual(scene_hints[2]["continuity_group"], "human_a")

    def test_style_lock_does_not_repeat_entire_user_prompt(self) -> None:
        settings = build_video_settings(content_type="spiritual_reflection", visual_style="Quiet Cinematic Animation")
        long_idea = (
            "Create a quiet animated reflection on Bhagavad Gita 3.38 using smoke covering fire, dust "
            "covering a mirror, and the womb enclosing unborn life. Use those metaphors to explore how "
            "desire can obscure human clarity."
        )
        plan = build_fallback_plan(long_idea, settings=settings)

        self.assertNotIn("Bhagavad Gita 3.38", plan.style_lock)
        self.assertNotIn("smoke covering fire", plan.style_lock.lower())
        self.assertNotIn(long_idea.lower(), plan.style_lock.lower())

    def test_standard_fallback_warning_is_explicit(self) -> None:
        settings = build_video_settings(total_duration_seconds=15, preferred_scene_duration_seconds=5)
        with patch("video_agent.request_video_plan_from_ollama", return_value="{not valid json"):
            plan, warning = build_video_plan(
                idea="A short story",
                settings=settings,
            )

        self.assertTrue(plan.used_fallback)
        self.assertIn("used the generic fallback storyboard", warning.lower())

    def test_multimind_invalid_storyboard_raises_instead_of_fallback(self) -> None:
        settings = build_video_settings(total_duration_seconds=15, preferred_scene_duration_seconds=5, content_type="story")
        creative_result = self.make_creative_result(
            idea="A concrete reflective story.",
            premise="A premise.",
            conflict="A conflict.",
            progression="A progression.",
            emotional_turn="A turn.",
            ending="An ending.",
            scene_beats=[
                "A person drops a letter into a mailbox.",
                "The same person waits by the window.",
                "The same person smiles at the reply.",
            ],
        )

        with patch("video_agent.request_video_plan_from_ollama", return_value="{bad json"):
            with self.assertRaisesRegex(ValueError, "Retry storyboard generation instead of using a generic fallback"):
                build_video_plan_from_creative_result(
                    idea=creative_result.request.idea,
                    creative_result=creative_result,
                    settings=settings,
                )

    def test_calculate_frame_count_respects_selected_fps(self) -> None:
        self.assertEqual(calculate_frame_count(duration_seconds=10, fps=24), 240)
        self.assertEqual(calculate_frame_count(duration_seconds=15, fps=30), 450)

    def test_build_scene_render_command_uses_24_fps(self) -> None:
        settings = build_video_settings(frame_rate=24)
        command = build_scene_render_command(
            image_path=Path("scene.png"),
            output_path=Path("clip.mp4"),
            duration_seconds=10,
            settings=settings,
        )
        self.assertIn("24", command)
        self.assertIn("240", command)

    def test_build_scene_render_command_uses_30_fps(self) -> None:
        settings = build_video_settings(frame_rate=30)
        command = build_scene_render_command(
            image_path=Path("scene.png"),
            output_path=Path("clip.mp4"),
            duration_seconds=15,
            settings=settings,
        )
        self.assertIn("30", command)
        self.assertIn("450", command)

    def test_build_scene_filter_uses_selected_dimensions_and_fps(self) -> None:
        settings = build_video_settings(frame_rate=24, aspect_ratio_label="Square 1:1")
        filter_chain = build_scene_filter(settings=settings, duration_seconds=10)
        self.assertIn("crop=1080:1080", filter_chain)
        self.assertIn("fps=24", filter_chain)

    def test_build_generation_output_dir_creates_unique_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_dir = build_generation_output_dir(base_dir=temp_dir)
            second_dir = build_generation_output_dir(base_dir=temp_dir)
            self.assertTrue(first_dir.exists())
            self.assertTrue(second_dir.exists())
            self.assertNotEqual(first_dir, second_dir)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not available")
    def test_render_scene_clip_outputs_expected_video_properties(self) -> None:
        settings = build_video_settings(total_duration_seconds=10, preferred_scene_duration_seconds=3, frame_rate=24)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "scene.png"
            clip_path = Path(temp_dir) / "scene_clip.mp4"
            Image.new("RGB", (1024, 1536), color="#CDEAC0").save(image_path, format="PNG")

            render_scene_clip(
                image_path=image_path,
                output_path=clip_path,
                duration_seconds=3,
                settings=settings,
            )

            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,r_frame_rate,nb_frames",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(clip_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(probe.stdout)
            stream = payload["streams"][0]
            duration = float(payload["format"]["duration"])

            self.assertEqual(stream["width"], settings.output_width)
            self.assertEqual(stream["height"], settings.output_height)
            self.assertEqual(stream["r_frame_rate"], "24/1")
            self.assertTrue(2.9 <= duration <= 3.1)


if __name__ == "__main__":
    unittest.main()
