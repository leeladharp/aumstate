import base64
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from video_agent import (
    DEFAULT_IMAGE_QUALITY,
    VideoScene,
    build_fallback_plan,
    build_scene_durations,
    build_video_settings,
    narration_word_target,
    settings_changed,
    settings_snapshot,
    total_duration_seconds,
    validate_video_plan_data,
)
from video_providers import (
    OpenAIImageProvider,
    PlaceholderImageProvider,
    build_openai_image_prompt,
    create_silent_wav,
    generate_narration_audio,
    generate_scene_images,
    map_quality_label_to_openai,
    normalize_quality_label,
    select_image_provider,
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
    def __init__(self, durations: list[float]) -> None:
        self.durations = durations
        self.calls = []

    def generate_narration_audio(self, narration: str, output_path: Path):
        self.calls.append(narration)
        duration = self.durations[min(len(self.calls) - 1, len(self.durations) - 1)]
        create_silent_wav(output_path=output_path, duration_seconds=duration)
        return type("Result", (), {"path": output_path, "used_fallback": False, "message": "Generated speech."})()


class VideoAgentTests(unittest.TestCase):
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
