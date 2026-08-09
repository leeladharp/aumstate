import json
import math
import shutil
import subprocess
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

from PIL import Image

from kling_assisted import (
    KLING_ASSISTED_MODE,
    KLING_DURATION_TOLERANCE_SECONDS,
    STATUS_INVALID,
    STATUS_READY,
    STATUS_VALID,
    assemble_kling_video,
    build_kling_package,
    build_kling_prompt,
    create_import_entry,
    extract_rotation,
    load_import_state,
    match_scene_number_from_filename,
    needs_renormalization,
    parse_rotation_value,
    normalize_clip,
    resolve_narration_path,
    save_import_state,
    store_uploaded_clip,
    update_entry_from_validation,
    validate_clip,
)
from video_agent import build_fallback_plan, build_video_settings
from video_providers import build_narration_audio_path, build_scene_image_path, create_silent_wav
from video_renderer import concat_scene_clips, render_video, write_concat_manifest
from unittest.mock import patch


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def run_command(command: list[str]) -> None:
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "command failed")


def probe_json(path: Path) -> dict:
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(process.stdout)


def create_tone_wav(output_path: Path, duration_seconds: float = 1.0, sample_rate: int = 22050) -> Path:
    frame_count = int(sample_rate * duration_seconds)
    amplitude = 12000
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            sample = int(amplitude * math.sin(2 * math.pi * 440 * (index / sample_rate)))
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))
    return output_path


def probe_rotation_json(path: Path) -> dict:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_entries",
            "stream=index,codec_type,tags,side_data_list",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(process.stdout)


def collect_rotation_sources(stream: dict) -> list[tuple[str, int]]:
    sources: list[tuple[str, int]] = []
    tags = stream.get("tags", {}) or {}
    side_data_list = stream.get("side_data_list", []) or []

    tag_rotation = parse_rotation_value(tags.get("rotate"))
    if tag_rotation is not None:
        sources.append(("tags.rotate", tag_rotation))

    for item in side_data_list:
        side_rotation = parse_rotation_value(item.get("rotation"))
        if side_rotation is not None:
            side_data_type = item.get("side_data_type", "side_data")
            sources.append((f"side_data_list.{side_data_type}", side_rotation))

    return sources


class KlingAssistedUnitTests(unittest.TestCase):
    def test_prompt_generation_includes_constraints_and_duration_guidance(self) -> None:
        settings = build_video_settings(video_mode=KLING_ASSISTED_MODE, visual_style="Simple Life Story")
        plan = build_fallback_plan("A quiet family breakfast", settings=settings)
        prompt = build_kling_prompt(plan.scenes[0], settings)
        self.assertIn(plan.scenes[0].motion_prompt, prompt)
        self.assertIn("Target visible action duration: approximately", prompt)
        self.assertIn("ordinary everyday actions", prompt)
        self.assertIn("no jitter", prompt)
        self.assertNotIn("api key", prompt.lower())
        self.assertNotIn("credentials", prompt.lower())

    def test_filename_mapping_supports_expected_patterns(self) -> None:
        self.assertEqual(match_scene_number_from_filename("scene_01_kling.mp4"), 1)
        self.assertEqual(match_scene_number_from_filename("scene-01-kling.mp4"), 1)
        self.assertEqual(match_scene_number_from_filename("scene_1_kling.mp4"), 1)
        self.assertEqual(match_scene_number_from_filename("scene01.mp4"), 1)
        self.assertIsNone(match_scene_number_from_filename("intro_clip.mp4"))

    def test_concat_manifest_uses_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            clip_one = base_dir / "scene 01.mp4"
            clip_two = base_dir / "scene 02.mp4"
            clip_one.write_bytes(b"1")
            clip_two.write_bytes(b"2")
            manifest_path = base_dir / "concat_manifest.txt"
            write_concat_manifest([clip_one, clip_two], manifest_path)
            manifest_text = manifest_path.read_text(encoding="utf-8")
            self.assertIn(str(clip_one.resolve()), manifest_text)
            self.assertIn(str(clip_two.resolve()), manifest_text)
            self.assertNotIn("file 'scene 01.mp4'", manifest_text)

    def test_rotation_from_tags_rotate(self) -> None:
        self.assertEqual(extract_rotation({"tags": {"rotate": "90"}}), 90)

    def test_rotation_from_side_data_display_matrix(self) -> None:
        stream = {
            "tags": {"rotate": "0"},
            "side_data_list": [
                {
                    "side_data_type": "Display Matrix",
                    "rotation": 90,
                }
            ],
        }
        self.assertEqual(extract_rotation(stream), 90)

    def test_rotation_from_side_data_negative_ninety_normalizes_to_270(self) -> None:
        stream = {
            "side_data_list": [
                {
                    "side_data_type": "Display Matrix",
                    "rotation": -90,
                }
            ],
        }
        self.assertEqual(extract_rotation(stream), 270)

    def test_absent_rotation_returns_zero(self) -> None:
        self.assertEqual(extract_rotation({}), 0)

    def test_rotation_value_parsing_normalizes_numeric_and_string_inputs(self) -> None:
        self.assertEqual(parse_rotation_value("90"), 90)
        self.assertEqual(parse_rotation_value(180), 180)
        self.assertEqual(parse_rotation_value(-90), 270)
        self.assertEqual(parse_rotation_value("270.0"), 270)

    def test_narration_path_survives_simulated_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            narration_path = output_dir / "custom_narration.wav"
            create_silent_wav(narration_path, duration_seconds=3.0)
            stored_path = resolve_narration_path(output_dir=output_dir, stored_narration_path=narration_path)
            rerun_path = resolve_narration_path(output_dir=output_dir, stored_narration_path=stored_path)
            self.assertEqual(stored_path, narration_path)
            self.assertEqual(rerun_path, narration_path)

    def test_narration_path_falls_back_to_generation_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            narration_path = build_narration_audio_path(output_dir)
            create_silent_wav(narration_path, duration_seconds=3.0)
            resolved_path = resolve_narration_path(output_dir=output_dir, stored_narration_path=None)
            self.assertEqual(resolved_path, narration_path)


@unittest.skipUnless(ffmpeg_available(), "ffmpeg/ffprobe not available")
class KlingAssistedFFmpegTests(unittest.TestCase):
    def create_plan_and_assets(self, temp_dir: str):
        settings = build_video_settings(
            total_duration_seconds=10,
            preferred_scene_duration_seconds=5,
            video_mode=KLING_ASSISTED_MODE,
            visual_style="Simple Life Story",
        )
        plan = build_fallback_plan("A quiet tea moment between a woman and child", settings=settings)
        output_dir = Path(temp_dir)
        for scene in plan.scenes:
            Image.new("RGB", (1024, 1536), color="#DDE7D6").save(
                build_scene_image_path(output_dir, scene.scene_number),
                format="PNG",
            )
        create_tone_wav(build_narration_audio_path(output_dir), duration_seconds=9.0)
        return settings, plan, output_dir

    def create_video_fixture(
        self,
        output_path: Path,
        duration: float,
        width: int = 640,
        height: int = 360,
        frame_rate: int = 30,
        with_audio: bool = False,
        rotation: int | None = None,
    ) -> Path:
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=#88AADD:s={width}x{height}:d={duration}:r={frame_rate}",
        ]
        if with_audio:
            command.extend(["-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono", "-shortest"])
        command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        if with_audio:
            command.extend(["-c:a", "aac"])
        else:
            command.extend(["-an"])
        command.append(str(output_path))
        run_command(command)
        return output_path

    def create_rotated_video_fixture(
        self,
        output_path: Path,
        duration: float,
        width: int = 360,
        height: int = 640,
        frame_rate: int = 30,
        rotation: int = 90,
    ) -> tuple[Path, dict]:
        base_path = output_path.parent / "base.mp4"
        base_command = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=#88AADD:s={width}x{height}:d={duration}:r={frame_rate}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(base_path),
        ]
        run_command(base_command)

        candidate_commands = [
            [
                "ffmpeg",
                "-y",
                "-display_rotation",
                str(rotation),
                "-i",
                str(base_path),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                str(output_path),
            ],
            [
                "ffmpeg",
                "-y",
                "-i",
                str(base_path),
                "-c",
                "copy",
                "-metadata:s:v:0",
                f"rotate={rotation}",
                str(output_path),
            ],
        ]

        last_payload: dict = {}
        for command in candidate_commands:
            if output_path.exists():
                output_path.unlink()
            try:
                run_command(command)
            except RuntimeError:
                continue

            payload = probe_rotation_json(output_path)
            video_stream = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
            if collect_rotation_sources(video_stream):
                return output_path, payload
            last_payload = payload

        self.skipTest(
            "FFmpeg build did not emit MP4 rotation metadata via tested remux methods. "
            f"Last ffprobe payload: {json.dumps(last_payload)}"
        )

    def test_export_package_contains_expected_files_and_no_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings, plan, output_dir = self.create_plan_and_assets(temp_dir)
            zip_path = build_kling_package(plan=plan, settings=settings, output_dir=output_dir, generation_id=output_dir.name)
            self.assertTrue(zip_path.exists())
            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
                self.assertIn("project_manifest.json", names)
                self.assertIn("kling_instructions.txt", names)
                self.assertIn("narration/narration.txt", names)
                self.assertIn("narration/narration.wav", names)
                self.assertIn("scene_01/scene_01.png", names)
                self.assertIn("scene_01/scene_01_kling_prompt.txt", names)
                manifest = json.loads(archive.read("project_manifest.json"))
                self.assertEqual(manifest["video_mode"], KLING_ASSISTED_MODE)
                serialized = "\n".join(names) + archive.read("project_manifest.json").decode("utf-8")
                self.assertNotIn("OPENAI_API_KEY", serialized)
                self.assertNotIn("AUMSTATE_SPEECH_API_KEY", serialized)

    def test_valid_mp4_succeeds_and_short_clip_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = self.create_video_fixture(Path(temp_dir) / "valid.mp4", duration=5.2)
            short_path = self.create_video_fixture(Path(temp_dir) / "short.mp4", duration=4.7)
            valid_status, _, valid_error = validate_clip(valid_path, required_duration=5.0)
            short_status, _, short_error = validate_clip(short_path, required_duration=5.0)
            self.assertIn(valid_status, {STATUS_VALID, STATUS_READY})
            self.assertEqual(valid_error, "")
            self.assertEqual(short_status, STATUS_INVALID)
            self.assertIn("Tolerance: 0.15s", short_error)

    def test_missing_video_stream_and_corrupted_mp4_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_only = Path(temp_dir) / "audio_only.mp4"
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=22050:cl=mono",
                    "-t",
                    "5",
                    "-c:a",
                    "aac",
                    str(audio_only),
                ]
            )
            corrupted = Path(temp_dir) / "broken.mp4"
            corrupted.write_bytes(b"not an mp4")
            status_audio, _, error_audio = validate_clip(audio_only, required_duration=5.0)
            status_bad, _, error_bad = validate_clip(corrupted, required_duration=5.0)
            self.assertEqual(status_audio, STATUS_INVALID)
            self.assertIn("no readable video stream", error_audio.lower())
            self.assertEqual(status_bad, STATUS_INVALID)
            self.assertIn("corrupted or unreadable", error_bad.lower())

    def test_rotation_metadata_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rotated, payload = self.create_rotated_video_fixture(
                Path(temp_dir) / "rotated.mp4",
                duration=5.1,
                width=360,
                height=640,
                rotation=90,
            )
            video_stream = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
            rotation_sources = collect_rotation_sources(video_stream)
            status, details, _ = validate_clip(rotated, required_duration=5.0)
            self.assertIn(status, {STATUS_VALID, STATUS_READY})
            self.assertTrue(rotation_sources, payload)
            self.assertEqual(details["rotation"], 90)

    def test_concat_missing_input_is_detected_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "out.mp4"
            missing_clip = Path(temp_dir) / "missing.mp4"
            with self.assertRaisesRegex(FileNotFoundError, "Missing concat input"):
                concat_scene_clips([missing_clip], output_path)

    def test_normalization_matches_settings_and_removes_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings, plan, output_dir = self.create_plan_and_assets(temp_dir)
            source_path = self.create_video_fixture(output_dir / "source.mp4", duration=6.0, width=1280, height=720, frame_rate=24, with_audio=True)
            normalized_path = normalize_clip(
                source_path=source_path,
                output_dir=output_dir,
                scene_number=1,
                settings=settings,
                required_duration=5.0,
                trim_start=0.5,
            )
            payload = probe_json(normalized_path)
            video_stream = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
            self.assertEqual(video_stream["width"], settings.output_width)
            self.assertEqual(video_stream["height"], settings.output_height)
            self.assertEqual(video_stream["r_frame_rate"], f"{settings.frame_rate}/1")
            self.assertEqual(video_stream["pix_fmt"], "yuv420p")
            self.assertEqual(video_stream["sample_aspect_ratio"], "1:1")
            self.assertAlmostEqual(float(payload["format"]["duration"]), 5.0, delta=0.15)
            self.assertFalse(any(stream["codec_type"] == "audio" for stream in payload["streams"]))

    def test_assembly_creates_final_output_with_narration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings, plan, output_dir = self.create_plan_and_assets(temp_dir)
            entries = {}
            for scene in plan.scenes:
                source_path = self.create_video_fixture(
                    output_dir / f"scene_{scene.scene_number:02d}_source.mp4",
                    duration=scene.duration_seconds + 0.5,
                    with_audio=True,
                )
                entry = create_import_entry(
                    generation_id=output_dir.name,
                    scene_number=scene.scene_number,
                    source_filename=source_path.name,
                    stored_source_path=source_path,
                    required_duration=scene.duration_seconds,
                )
                status, details, error_summary = validate_clip(
                    clip_path=source_path,
                    required_duration=scene.duration_seconds,
                )
                entry = update_entry_from_validation(entry, source_path, status, details, error_summary)
                normalized = normalize_clip(
                    source_path=source_path,
                    output_dir=output_dir,
                    scene_number=scene.scene_number,
                    settings=settings,
                    required_duration=scene.duration_seconds,
                    trim_start=0.0,
                )
                entry.normalized_output_path = str(normalized)
                entry.validation_status = STATUS_READY
                entries[scene.scene_number] = entry
            stored_narration_path = output_dir / "stored_narration.wav"
            create_tone_wav(stored_narration_path, duration_seconds=9.0)
            final_path = assemble_kling_video(
                plan=plan,
                settings=settings,
                output_dir=output_dir,
                import_entries=entries,
                narration_path=resolve_narration_path(output_dir=output_dir, stored_narration_path=stored_narration_path),
            )
            payload = probe_json(final_path)
            self.assertTrue(final_path.exists())
            self.assertEqual(final_path.name, "final_kling_assisted.mp4")
            self.assertAlmostEqual(float(payload["format"]["duration"]), settings.total_duration_seconds, delta=0.25)
            audio_stream = next(stream for stream in payload["streams"] if stream["codec_type"] == "audio")
            self.assertEqual(audio_stream["codec_name"], "aac")
            self.assertTrue((output_dir / "final_kling_assisted_silent.mp4").exists())

    def test_missing_scene_blocks_final_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings, plan, output_dir = self.create_plan_and_assets(temp_dir)
            with self.assertRaisesRegex(ValueError, "Missing valid clips"):
                assemble_kling_video(
                    plan=plan,
                    settings=settings,
                    output_dir=output_dir,
                    import_entries={},
                    narration_path=build_narration_audio_path(output_dir),
                )

    def test_missing_narration_gives_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings, plan, output_dir = self.create_plan_and_assets(temp_dir)
            entries = {}
            for scene in plan.scenes:
                source_path = self.create_video_fixture(
                    output_dir / f"scene_{scene.scene_number:02d}_source.mp4",
                    duration=scene.duration_seconds + 0.5,
                )
                entry = create_import_entry(
                    generation_id=output_dir.name,
                    scene_number=scene.scene_number,
                    source_filename=source_path.name,
                    stored_source_path=source_path,
                    required_duration=scene.duration_seconds,
                )
                status, details, error_summary = validate_clip(source_path, scene.duration_seconds)
                entry = update_entry_from_validation(entry, source_path, status, details, error_summary)
                normalized = normalize_clip(
                    source_path=source_path,
                    output_dir=output_dir,
                    scene_number=scene.scene_number,
                    settings=settings,
                    required_duration=scene.duration_seconds,
                    trim_start=0.0,
                )
                entry.normalized_output_path = str(normalized)
                entry.validation_status = STATUS_READY
                entries[scene.scene_number] = entry

            missing_path = output_dir / "missing_narration.wav"
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Narration is enabled, but the generated narration file could not be found. Regenerate narration before assembling.",
            ):
                assemble_kling_video(
                    plan=plan,
                    settings=settings,
                    output_dir=output_dir,
                    import_entries=entries,
                    narration_path=missing_path,
                )

    def test_silent_narration_blocks_final_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings, plan, output_dir = self.create_plan_and_assets(temp_dir)
            entries = {}
            for scene in plan.scenes:
                source_path = self.create_video_fixture(
                    output_dir / f"scene_{scene.scene_number:02d}_source.mp4",
                    duration=scene.duration_seconds + 0.5,
                )
                entry = create_import_entry(
                    generation_id=output_dir.name,
                    scene_number=scene.scene_number,
                    source_filename=source_path.name,
                    stored_source_path=source_path,
                    required_duration=scene.duration_seconds,
                )
                status, details, error_summary = validate_clip(source_path, scene.duration_seconds)
                entry = update_entry_from_validation(entry, source_path, status, details, error_summary)
                normalized = normalize_clip(
                    source_path=source_path,
                    output_dir=output_dir,
                    scene_number=scene.scene_number,
                    settings=settings,
                    required_duration=scene.duration_seconds,
                    trim_start=0.0,
                )
                entry.normalized_output_path = str(normalized)
                entry.validation_status = STATUS_READY
                entries[scene.scene_number] = entry

            with patch("kling_assisted.inspect_narration_audio_file", side_effect=ValueError("Narration file is silent")):
                with self.assertRaisesRegex(ValueError, "Narration file is silent"):
                    assemble_kling_video(
                        plan=plan,
                        settings=settings,
                        output_dir=output_dir,
                        import_entries=entries,
                        narration_path=build_narration_audio_path(output_dir),
                    )

    def test_narration_disabled_output_has_no_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = build_video_settings(
                total_duration_seconds=10,
                preferred_scene_duration_seconds=5,
                video_mode=KLING_ASSISTED_MODE,
                visual_style="Simple Life Story",
                narration_enabled=False,
            )
            plan = build_fallback_plan("A quiet tea moment between a woman and child", settings=settings)
            output_dir = Path(temp_dir)
            entries = {}
            for scene in plan.scenes:
                source_path = self.create_video_fixture(
                    output_dir / f"scene_{scene.scene_number:02d}_source.mp4",
                    duration=scene.duration_seconds + 0.5,
                    with_audio=True,
                )
                entry = create_import_entry(
                    generation_id=output_dir.name,
                    scene_number=scene.scene_number,
                    source_filename=source_path.name,
                    stored_source_path=source_path,
                    required_duration=scene.duration_seconds,
                )
                status, details, error_summary = validate_clip(source_path, scene.duration_seconds)
                entry = update_entry_from_validation(entry, source_path, status, details, error_summary)
                normalized = normalize_clip(
                    source_path=source_path,
                    output_dir=output_dir,
                    scene_number=scene.scene_number,
                    settings=settings,
                    required_duration=scene.duration_seconds,
                    trim_start=0.0,
                )
                entry.normalized_output_path = str(normalized)
                entry.validation_status = STATUS_READY
                entries[scene.scene_number] = entry

            final_path = assemble_kling_video(
                plan=plan,
                settings=settings,
                output_dir=output_dir,
                import_entries=entries,
                narration_path=None,
            )
            payload = probe_json(final_path)
            self.assertEqual(final_path.name, "final_kling_assisted_silent.mp4")
            self.assertFalse(any(stream["codec_type"] == "audio" for stream in payload["streams"]))

    def test_import_state_persists_and_replacement_invalidates_only_one_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings, plan, output_dir = self.create_plan_and_assets(temp_dir)
            first_clip = self.create_video_fixture(output_dir / "scene1.mp4", duration=5.4)
            second_clip = self.create_video_fixture(output_dir / "scene2.mp4", duration=5.4)
            stored_one = store_uploaded_clip(output_dir, 1, first_clip.name, first_clip.read_bytes())
            stored_two = store_uploaded_clip(output_dir, 2, second_clip.name, second_clip.read_bytes())

            entries = {}
            for scene_number, stored_path in [(1, stored_one), (2, stored_two)]:
                required_duration = plan.scenes[scene_number - 1].duration_seconds
                entry = create_import_entry(
                    generation_id=output_dir.name,
                    scene_number=scene_number,
                    source_filename=stored_path.name,
                    stored_source_path=stored_path,
                    required_duration=required_duration,
                )
                status, details, error_summary = validate_clip(stored_path, required_duration)
                entry = update_entry_from_validation(entry, stored_path, status, details, error_summary)
                normalized = normalize_clip(
                    source_path=stored_path,
                    output_dir=output_dir,
                    scene_number=scene_number,
                    settings=settings,
                    required_duration=required_duration,
                    trim_start=0.0,
                )
                entry.normalized_output_path = str(normalized)
                entry.validation_status = STATUS_READY
                entries[scene_number] = entry

            save_import_state(output_dir, output_dir.name, entries)
            loaded = load_import_state(output_dir)
            self.assertEqual(set(loaded.keys()), {1, 2})
            self.assertFalse(needs_renormalization(loaded[1], Path(loaded[1].stored_source_path)))

            replacement = self.create_video_fixture(output_dir / "scene1_replacement.mp4", duration=5.6)
            stored_replacement = store_uploaded_clip(output_dir, 1, replacement.name, replacement.read_bytes())
            new_entry = create_import_entry(
                generation_id=output_dir.name,
                scene_number=1,
                source_filename=replacement.name,
                stored_source_path=stored_replacement,
                required_duration=plan.scenes[0].duration_seconds,
            )
            status, details, error_summary = validate_clip(stored_replacement, plan.scenes[0].duration_seconds)
            loaded[1] = update_entry_from_validation(new_entry, stored_replacement, status, details, error_summary)
            save_import_state(output_dir, output_dir.name, loaded)
            reloaded = load_import_state(output_dir)
            self.assertTrue(needs_renormalization(reloaded[1], Path(reloaded[1].stored_source_path)))
            self.assertFalse(needs_renormalization(reloaded[2], Path(reloaded[2].stored_source_path)))

    def test_basic_motion_renderer_still_attaches_narration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            settings = build_video_settings(total_duration_seconds=10, preferred_scene_duration_seconds=5)
            image_paths = []
            for scene_number in range(1, 3):
                image_path = output_dir / f"scene_{scene_number:02d}.png"
                Image.new("RGB", (1024, 1536), color="#DDE7D6").save(image_path, format="PNG")
                image_paths.append(image_path)
            narration_path = output_dir / "basic_motion_narration.wav"
            create_silent_wav(narration_path, duration_seconds=8.0)
            final_path = render_video(
                image_paths=image_paths,
                scene_durations=[5, 5],
                output_dir=output_dir,
                settings=settings,
                audio_path=narration_path,
            )
            payload = probe_json(final_path)
            self.assertEqual(final_path.name, "final_video.mp4")
            self.assertAlmostEqual(float(payload["format"]["duration"]), settings.total_duration_seconds, delta=0.25)
            self.assertTrue(any(stream["codec_type"] == "audio" for stream in payload["streams"]))


if __name__ == "__main__":
    unittest.main()
