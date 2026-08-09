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

    def test_openai_image_provider_uses_first_scene_as_reference_for_later_scenes(self) -> None:
        settings = build_video_settings(total_duration_seconds=30, preferred_scene_duration_seconds=5)
        plan = build_fallback_plan("A nursery story", settings=settings)
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

    def test_build_openai_image_prompt_includes_style_lock_and_scene_details(self) -> None:
        settings = build_video_settings()
        plan = build_fallback_plan("A nursery story", settings=settings)
        scene = plan.scenes[0]
        prompt = build_openai_image_prompt(plan=plan, scene=scene)
        self.assertIn(plan.style_lock, prompt)
        self.assertIn(scene.visual_prompt, prompt)
        self.assertIn(scene.motion_prompt, prompt)
        self.assertIn("hairstyle", prompt.lower())
        self.assertIn("hair length", prompt.lower())
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
        self.assertIn("3D Nursery Animation", prompt)

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
