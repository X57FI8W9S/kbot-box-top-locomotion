# Policy Video Report

This report documents the KBot policy playback video maker, the HUD layout rules, and how to add or remove indicators without breaking visual comparability.

## Script

The policy video maker is:

```bash
scripts/rsl_rl/play_trailing.py
```

It loads an RSL-RL checkpoint, creates one Isaac Lab environment, runs the policy, renders two camera views, and writes one combined inspection video:

- left half: trailing/front-biased view
- right half: side view
- top band: one shared HUD overlay

The output is intentionally hard-coded to `1280x720` and `16:9`. Each view receives `640x720`, so the two halves remain comparable across runs. The command-line `--width` and `--height` arguments are still present for compatibility, but the writer now uses the authored constants:

```python
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
```

## Current HUD Layout

The HUD is one top panel from `x=18..1262`, `y=18..110`. Its purpose is to make policy behavior readable in a paused frame, not to be a full dashboard.

Current indicator groups:

- left block: speed plus fixed small-metric columns for command speed, yaw, root height, x/y distance, rolling window label, J/m, and `apv`
- lower left block: torso RMS, torso average, hip roll/yaw RMS, fsep, and ksep
- center/right block: averaged L/R joint positions
- right block: support percentages
- far-right block: gait timing, distance, and cadence

The layout is hard-coded because the video resolution is hard-coded. This is deliberate: dynamic placement made the columns drift and caused collisions like the `5cy avg` label overlapping the left ankle row.

## Fixed-Position Rules

Indicator numbers must have fixed places.

Do not draw a number by concatenating arbitrary text and placing it next to another indicator. A new digit or a minus sign must not shift a neighboring column.

Use fixed-width formatting:

```python
_format_float(value, width, precision)
_put_fixed(frame, text, origin, width=...)
```

For columns with decimals, align on the decimal point by giving all values in that column the same width and precision. Examples:

```python
_format_float(row["time_s"], 4, 2)
_format_float(row["length_m"], 5, 3)
_format_float(row["rate_hz"], 5, 2)
_format_float(float(joint_pos[index]), 5, 2)
```

For signed values, reserve room for the sign even when the value is positive. This keeps `0.12` and `-0.12` from changing the column width.

## Current Fixed Slots

The compact right-side HUD slots in `_draw_hud()` are:

```python
left_x = 735
left_val_x = 797
right_x = 855
right_val_x = 927
support_x = 985
support_l_x = 1036
support_r_x = 1088
gait_label_x = 1112
gait_t_x = 1150
gait_m_x = 1188
gait_hz_x = 1227
```

The left side of the overlay is organized as fixed columns:

```python
col0 = x
col1 = x + 114
col2 = x + 238
col3 = x + 354
col4 = x + 469
col5 = x + 574
```

Do not draw free-floating indicators between `col5` and the L/R joint block. The `apv` indicator is drawn in `col5` on the speed row with fixed-width formatting. It uses the same bootstrap reward-side gate as the current V2.5 step-margin experiment: alternating step root advance at least `0.008 m` and alternating-step duration at least `0.07 s`.

The support block uses `spt%` as a short header. Rows mean:

- `gnd`: percentage of the HUD window where each foot is in stance/contact
- `sgl`: percentage of the HUD window where each foot is the only supporting foot
- `dbl` / `air`: double-support percentage and airborne percentage

The `gait t m Hz` columns are intentionally close together and pushed far right. The L/R joint columns are right of the rolling-window label. The support block sits between joints and gait. Keep these groups separate unless the whole top band is redesigned.

## Adding an Indicator

1. Decide whether it is a per-frame value, rolling-window value, step/cycle value, or final JSON-only value.
2. Add the data collection in the simulation loop near the existing windows, for example `speed_window`, `root_x_window`, or `positive_work_window`.
3. Compute the HUD value immediately before `_draw_hud()`.
4. Add a new argument to `_draw_hud()` only if the value is visible in the video.
5. Draw it in an existing reserved slot or create a new fixed slot.
6. Use `_format_float()` and `_put_fixed()` for numbers.
7. Add the same value to the metrics JSON if it is useful for comparing runs.
8. Compile and visually check at least one frame or short video.

## Removing an Indicator

Remove in this order:

1. Remove the draw call from `_draw_hud()`.
2. Remove the `_draw_hud()` parameter.
3. Remove the value computation before `_draw_hud()`.
4. Remove the rolling window only if nothing else uses it.
5. Remove the JSON metric only if historical comparison is not needed.

Do not leave a dead slot that looks like missing data unless the missing slot itself is intentional.

## Compression Notes

The videos currently use OpenCV `VideoWriter` with `mp4v`. The visible compression artifacts are probably from that simple encoder path and its default bitrate/quality behavior. This is not the priority right now; the layout and metrics should be fixed first.

If compression becomes important later, likely fixes are:

- write a less compressed intermediate and transcode with `ffmpeg`
- switch the writer path to a backend that exposes bitrate/CRF
- increase output resolution after the HUD is redesigned for that resolution

For now, keep `1280x720` so metric positions stay stable.

## Maintenance Checklist

Before accepting a video-maker change:

- `python3 -m py_compile scripts/rsl_rl/play_trailing.py`
- Confirm the video is `1280x720`.
- Confirm `5cy avg` or any window label does not overlap L/R joint rows.
- Confirm L/R joint columns do not overlap gait or support columns.
- Confirm `gait t m Hz` columns are close but readable.
- Confirm negative signs do not move neighboring numbers.
- Confirm decimal columns stay visually aligned.
- Confirm the report is updated when the HUD layout changes.
