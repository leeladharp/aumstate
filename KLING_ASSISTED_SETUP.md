# Kling Assisted Workflow

No Kling API key is required.

## Overview

`Basic Motion` remains the default mode and continues to render a full video locally with the existing FFmpeg still-image workflow.

`Kling Assisted` uses this flow:

1. Create a storyboard in AumState.
2. Generate scene images and narration in AumState.
3. Review the Kling-ready motion prompt for each scene.
4. Export the Kling package ZIP.
5. Open the Kling website manually and generate one clip per scene.
6. Rename each clip to the expected scene filename.
7. Upload the clips back into AumState.
8. Validate, trim, normalize, and assemble the final video in AumState.

## Export Kling Package

Use `Video generation mode -> Kling Assisted`, then:

1. Click `Create storyboard`.
2. Click `Generate scene images and narration`.
3. Click `Export Kling Package`.
4. Download the generated ZIP file.

The ZIP includes:

- `project_manifest.json`
- `kling_instructions.txt`
- `narration/narration.txt`
- `narration/narration.wav` when narration is enabled
- one folder per scene with:
  - `scene_XX.png`
  - `scene_XX_kling_prompt.txt`
  - `scene_XX_details.json`

## Generate One Scene On Kling

Recommended first pass:

1. Open the Kling website.
2. Select image-to-video.
3. Upload `scene_01.png`.
4. Paste `scene_01_kling_prompt.txt`.
5. Generate the clip.
6. Inspect the result before generating the remaining scenes.

Recommendations:

- use the same Kling model/settings across scenes where possible
- avoid automatic Kling text or subtitles
- avoid extending the clip unnecessarily
- preserve the uploaded image composition
- do not change character appearance between scenes

## Required Filenames

Expected imported filenames follow this pattern:

- `scene_01_kling.mp4`
- `scene_02_kling.mp4`

Supported filename variations also map automatically, including:

- `scene-01-kling.mp4`
- `scene_1_kling.mp4`
- `scene01.mp4`

Files that cannot be mapped automatically are shown under `Unmatched files` and can be assigned manually in the UI.

## Upload And Trim

Upload MP4 clips in the `Upload Kling clips` section.

Rules:

- all clips can be uploaded at once
- clips can be replaced scene by scene
- AumState keeps the active imported file in `outputs/<generation_id>/kling_imports/`
- imported clip audio is discarded during normalization

Each scene has a `Clip start position` control:

- default is `0.0`
- use it to choose the best segment when the imported Kling clip is longer than the storyboard scene
- AumState trims to the storyboard scene duration
- AumState does not loop clips
- AumState does not silently slow clips down

Duration tolerance:

- clips are accepted when source duration is at least `required_duration - 0.15 seconds`

## Final Assembly

After upload:

1. Click `Validate and assemble final video`.
2. AumState validates each clip with `ffprobe`.
3. Valid clips are normalized with FFmpeg to the approved output size, frame rate, SAR, pixel format, and exact scene duration.
4. AumState concatenates the normalized clips in storyboard order.
5. AumState adds AumState narration audio.
6. Final output is written inside `outputs/<generation_id>/`.

Imported Kling clip audio is removed by default so it does not conflict with narration.
