# K-LD7 Vertical Geometry Selector

This document describes the geometry-based K-LD7 vertical launch-angle selector.
It is intended to make the production behavior understandable without relying on
session-specific analysis or TrackMan comparisons.

## Goal

The K-LD7 raw RADC path measures target bearing from receive-channel phase
difference. A bearing is not the same thing as a launch angle: it is the angle
from the radar to the ball's current position at a specific time after impact.

The geometry selector estimates vertical launch angle by combining:

- K-LD7 per-frame bearing
- OPS243 ball speed
- impact timestamp
- radar-to-ball distance
- radar mount tilt
- radar/ball height relationship
- K-LD7 boresight or bearing offset

The selector prefers a two-frame trajectory fit. If only one strong in-flight
frame is available, it can produce a low-confidence single-frame geometry
fallback.

## Required Inputs

The geometry selector runs through `extract_launch_angle(...)` in
`src/openflight/kld7/radc.py`. The pure trajectory math lives in
`src/openflight/kld7/geometry.py` so RADC parsing and geometry fitting stay
separate.

Required for geometry mode:

- `frames`: K-LD7 frames containing raw `radc` payloads.
- `ops243_ball_speed_mph`: OPS243 measured ball speed.
- `shot_timestamp` or `impact_timestamp`: impact time in epoch seconds.
- `mount_deg`: physical vertical radar mount tilt in degrees.
- `distance_ft`: ball-to-radar-front distance in feet.
- `orientation="vertical"`.
- `vertical_estimator="geometry"`.

For live range sessions, the kiosk shortcut is:

```bash
scripts/start-kiosk.sh --kld7-geometry
```

This behaves like the normal K-LD7 startup path, including horizontal K-LD7,
but switches the vertical radar to geometry mode and forwards the current field
defaults: `--kld7-vertical-estimator geometry`, `--kld7-mount-tilt 10`,
`--kld7-ball-distance 5`, and `--kld7-angle-offset 2.5`. The individual
`--kld7-*` flags can still be passed to override the preset.

Plain `--kld7` intentionally stays on the legacy vertical estimator. Use it
when you want the existing K-LD7 path without the geometry selector. The
TrackMan collection preset enables geometry automatically:

```bash
scripts/start-kiosk.sh --trackman-test
```

That preset also enables horizontal K-LD7 and raw RADC payload logging so the
session can be replayed offline.

Important optional/configuration inputs:

- `angle_offset_deg`: bearing/boresight correction applied to raw K-LD7 angle
  before geometry fitting.
- `speed_tolerance_mph`: OPS-speed search half-width for candidate bins.
- `spectrum_source`: spectrum used to choose the Doppler bin. `sum12` combines
  the F1A and F2A receive-channel magnitudes before angle extraction.
- `ops_bin_outlier_tol`: maximum bin distance used by the OPS-near peak search.
- `ops_anchored_peak_min_snr`: SNR required for a primary OPS-bin anchor.
- `centroid_floor_frac`: controls centroid angle extraction around the selected
  spectral peak.

## Coordinate Terms

`mount_deg` and `angle_offset_deg` correct different things.

`mount_deg` is the mechanical direction of the radar relative to the room and
ball flight. It is used by the trajectory model when predicting what bearing
the radar should see for a given launch angle.

`angle_offset_deg` is a bearing correction applied to the raw phase-derived
K-LD7 angle:

```text
corrected_bearing = raw_kld7_angle + angle_offset_deg
```

This offset should be treated as a boresight/electrical-zero calibration term,
not as a launch-angle fudge. It accounts for the possibility that raw RADC phase
angle zero does not exactly equal the radar's mechanical zero.

## Frame Discovery

The selector starts by deriving a Doppler bin range from OPS ball speed. Golf
ball speeds can alias at the K-LD7 speed setting, so the expected bin may appear
in the wrapped portion of the FFT spectrum.

For each RADC frame:

1. Parse the RADC payload into F1A/F2A/F1B channels.
2. FFT the receive channels.
3. Build the configured spectrum (`f1a`, `f2a`, or `sum12`).
4. Search for a peak near the OPS-expected Doppler bin.
5. Compute SNR against the frame noise floor.
6. Compute per-bin angle from F1A/F2A phase difference.
7. Compute the selected peak angle, normally as a magnitude-weighted centroid
   around the spectral peak.

Before per-frame candidate selection, the code finds impact-energy frames and
groups nearby frames into shot events. For each group, it examines the impact
frames plus a small surrounding window.

## Candidate Gates

For vertical geometry, per-frame candidates are passed through a rule stack:

- Time gate: candidate must be roughly `20-100 ms` after impact.
- OPS-bin gate: candidate bin must be within `50` bins of the expected OPS bin.
- Strong-anchor requirement: at least one non-weak OPS-bin candidate is required.
- Weak adjacent candidates may be retained only as context around a strong
  anchor.

There are two SNR concepts:

- Strong anchor: must meet `ops_anchored_peak_min_snr`.
- Weak adjacent frame: may be kept at `SNR >= 3.0`, but only if it is adjacent
  to a strong anchor and passes the trajectory sanity checks.

Weak frames are never allowed to start a geometry fit by themselves.

## Production Selection Ladder

When geometry mode is enabled, the vertical extractor should be read as a
decision ladder:

1. Primary two-frame geometry: use two or more selected frames in the normal
   `20-100 ms` post-impact window when they match the OPS bin and pass the
   rising-bearing rule.
2. Early-assisted two-frame geometry: allow a `5-20 ms` frame only as adjacent
   context around a strong in-window anchor, and only when it matches the OPS
   bin for that same early timing range.
3. Single-frame geometry: if no valid pair survives, solve from one strong
   frame and mark the result as `geometry_single_frame`.
4. Naive rule-stack fallback: if geometry cannot run but the rule-stack group is
   still usable, return the weighted bearing average as `naive_rule_stack`.
5. Legacy naive suspect: if only a broader legacy group remains, cap confidence
   and mark the result as `legacy_naive_suspect`.

`select_best_shot_result(...)` ranks those paths in that same spirit:
`geometry_primary`, then `geometry_early_assisted`, then
`geometry_single_frame`, then naive paths. This prevents a later clutter group
from overriding an earlier geometry candidate from the actual ball flight.

## Anchor Selection

After primary gates pass, the selector chooses one strong anchor frame.

Anchor sorting prefers:

1. smallest OPS-bin error
2. highest SNR
3. highest phase coherence

The anchor is the most trusted frame for the shot.

## Two-Frame Geometry

The selector tries to pair the anchor with an adjacent frame:

- previous gated frame, if present
- otherwise next gated frame, if present

The pair must pass a rising-bearing rule:

```text
later_bearing > earlier_bearing
```

For a ball launched upward, the vertical bearing should increase over early
flight. If the adjacent frame moves the bearing downward, it is rejected as
likely clutter, multipath, club contamination, or timing-adjacent noise.

The adjacent frame SNR threshold is:

- weak adjacent: `>= 3.0`
- normal adjacent: `>= max(2.0, 50% of anchor SNR)`

If a valid pair is found, the selected per-frame data is fit with
`fit_launch_angle_geometric(...)`.

The fit searches launch angle from `0-45 deg` in `0.1 deg` steps and minimizes
weighted bearing residual:

```text
predicted_bearing(alpha, t, speed, distance, mount)
```

The trajectory model treats the ball as moving in a straight line over the short
in-flight window:

```text
x = distance_ft + speed_ft_s * cos(alpha) * t
y = ball_above_radar_ft + speed_ft_s * sin(alpha) * t

bearing = atan2(y, x) - mount_deg
```

The current default vertical origin assumes the ball starts about `4 in` below
the radar center:

```text
ball_above_radar_ft = -4 / 12
```

Gravity is ignored because the modeled time window is very short.

If a weak adjacent frame participates in the two-frame fit, the fit is accepted
only when the geometry RMSE is small. A high-RMSE weak pair is rejected and the
selector can fall back to the single-frame path.

## Single-Frame Geometry Fallback

If a two-frame geometry fit is unavailable, the selector can attempt a
single-frame geometry fallback.

This path is intentionally narrow:

- geometry mode must be enabled
- vertical orientation is required
- ball speed, impact timing, mount tilt, and distance must all be present
- exactly one strong non-weak frame must remain after rule selection
- weak adjacent frames cannot create a single-frame result
- the solved angle must stay inside the physical vertical bounds

The single-frame solver uses the same predicted-bearing model as the two-frame
fit, but with one bearing observation. It grid-searches `0-45 deg` and accepts
the angle only if the predicted bearing can match the observed bearing within
the configured residual limit.

The result is marked separately:

```text
estimator = "geometry_single_frame"
```

Single-frame geometry is lower confidence because bearing, timing, and setup
errors cannot be averaged across multiple frames. The confidence is capped below
the strict-accept threshold so downstream selection treats it as a soft candidate.

## Fallbacks And Bounds

If geometry cannot run, the function can still return the legacy naive estimate:

```text
naive_angle = weighted_average_raw_bearing + angle_offset_deg
```

For vertical orientation, final output is bounded to:

```text
0 deg <= launch_angle <= 45 deg
```

Angles outside that range are rejected.

## Confidence

Confidence is estimator-dependent.

Two-frame geometry:

- blends average SNR with fit RMSE
- lower RMSE increases confidence

Single-frame geometry:

- uses SNR and one-frame residual
- confidence is capped below strict radar acceptance

Naive:

- follows legacy SNR/frame-count confidence behavior

The estimator name is included in the result payload so callers can distinguish
between true trajectory geometry, single-frame fallback, and legacy naive output.

## Server Acceptance

The extractor can return a candidate that the server still refuses to use as the
user-facing launch angle. This is deliberate: the extractor answers "what did
the radar see?" while the server answers "is this plausible enough to put on the
shot?"

Vertical radar acceptance has three confidence bands:

- `strict_accept`: confidence is at least `0.80`.
- `soft_accept`: confidence is at least `0.68` and the candidate passes the
  soft guardrails.
- `low_confidence_accept`: confidence is at least `0.65` and every other
  guardrail passes. This lets the UI/session log preserve marginal but aligned
  radar shots instead of silently replacing them with the estimator.

Common rejection reasons:

- `implausible_launch`: the launch plausibility model rejected the radar angle
  for the club, ball speed, club speed, and spin context.
- `low_confidence`: confidence was below the low-confidence floor.
- `outside_soft_lane`: the angle was outside the broad club-family launch lane
  used for soft/low-confidence candidates.
- `estimator_delta_too_large`: the radar angle disagreed too much with the
  independent launch estimator.
- `no_candidate_frames`: the extractor returned no usable selected frames.
- `suspicious_frame_span`: the candidate used many frames and still disagreed
  with the estimator, which often points at clutter rather than ball flight.

If a candidate is rejected, the session can still contain useful
`radc_selection` diagnostics. Do not treat a rejected radar candidate as
worthless until you inspect the selected frames, timing, bin errors, and SNR.

## Logging And Diagnostics

The selector logs:

- rule-stack pass/fail reasons for candidate frames
- anchor frame choice
- pair-rule failures
- weak adjacent candidate retention
- geometry RMSE rejection
- single-frame geometry fallback details
- OPS-bin penalty warnings when selected bins are far from the OPS-expected bin

Useful result fields include:

| Field | Meaning |
| --- | --- |
| `estimator` | `geometry`, `geometry_single_frame`, or `naive`. |
| `selection_path` | More specific path such as `geometry_primary`, `geometry_early_assisted`, `geometry_single_frame`, `naive_rule_stack`, or `legacy_naive_suspect`. |
| `selected_frame_indices` | K-LD7 frame indices selected inside the shot window. |
| `selected_t_ms` | Selected frame times relative to the impact timestamp used for K-LD7. |
| `selected_bin_errors` | Absolute bin distance from the OPS-expected Doppler bin. |
| `geom_fit_rmse_deg` | Two-frame geometry bearing-fit residual. High values usually mean timing, clutter, or bad pairing. |
| `geom_single_frame_resid_deg` | One-frame geometry bearing residual. This can look excellent even when the shot is underconstrained. |
| `weak_adjacent_frame_used` | True when a weak adjacent frame contributed to the selected fit. |
| `raw_angle_deg` | Weighted bearing before final geometry interpretation. |
| `angle_offset_deg` | Bearing offset applied to raw K-LD7 angle. |
| `ball_speed_mph` | Speed implied by the selected K-LD7 Doppler bins. Compare with OPS/TrackMan. |
| `confidence` | Extractor confidence before server acceptance guardrails. |
| `frame_count` | Number of frames used by the selected result. |
| `avg_snr_db` | Average SNR of the selected frames. |
| `impact_frames` | Internal impact-energy group used for this result. |

In console logs, look for lines like:

```text
[RADC-RULES] frame=39 t_ms=50.9 bin=1879 bin_err=1 snr=16.25 angle=-3.93 coh=0.99 -> PASS
[KLD7] RADC: angle=15.8° ... est=geometry_single_frame path=geometry_single_frame selected_frames=[39] selected_t_ms=[50.9]
```

Those two lines usually tell you whether the radar had a credible target and
whether the selector had enough frames to make a constrained geometry estimate.

## Timing Debug Workflow

The most important lesson from the TrackMan sessions is that a miss is not
automatically bad radar data. Several misses had good OPS-bin matches and good
SNR, but the useful K-LD7 frames landed slightly outside the selector's timing
window or produced a different launch angle when replayed with a shifted impact
instant.

Use this workflow before changing thresholds:

1. Confirm raw RADC exists. The session must have been collected with
   `--trackman-test` or `--experimental-kld7-raw-radc-logging`.
2. Confirm both K-LD7 streams were healthy. If stream health is far below
   roughly `35 Hz`, or the log shows repeated short RADC payload reads, fix USB
   and serial reliability before tuning selection logic.
3. Read the live selected result:
   `estimator`, `selection_path`, `selected_t_ms`, `selected_bin_errors`,
   `avg_snr_db`, and `geom_fit_rmse_deg`.
4. Scan nearby frames from about `-100 ms` to `+100 ms` around the K-LD7 impact
   timestamp. Look for frames with low bin error, decent SNR, coherent phase,
   and rising bearing.
5. Replay trial impact shifts such as `-40`, `-30`, `-20`, `-10`, `+10`, `+20`,
   `+30`, and `+40 ms`. Recompute frame times and geometry angle for the same
   physical frames.
6. Categorize the shot:
   - production two-frame geometry selected correctly
   - two-frame geometry recoverable by timing shift
   - one-frame radar evidence only
   - no usable radar signal, clutter, or off-boresight miss

A timing shift that makes a two-frame pair match TrackMan is strong evidence
that the radar saw the ball and the impact alignment is wrong. A timing shift
that makes a single frame match TrackMan is weaker evidence: with one bearing
observation, timing can move the solved launch angle enough to "fit" the target.
Keep those shots in a separate one-frame diagnostic bucket.

When investigating timing, compare against OPS transition diagnostics in
`rolling_buffer_capture`:

- `impact_source`
- `impact_timestamp_ms`
- `impact_offset_from_trigger_ms`
- `impact_last_club_center_ms`
- `impact_first_ball_center_ms`
- `impact_speed_delta_mph`
- `impact_transition_gap_ms`
- `impact_reason`

`impact_source="ops_transition"` means the K-LD7 path used the midpoint between
the last club-like OPS frame and first ball-like OPS frame. If the speed jump is
below the threshold or either side is missing, the code falls back to the sound
trigger timing.

## Calibration Notes

The geometry selector depends on physical setup measurements. The most important
ones are:

- impact timestamp
- radar-to-ball distance
- radar mount tilt
- radar center height relative to ball height
- boresight/bearing offset

The boresight offset should be calibrated with a known reflector position when
possible. For a reflector at known height and distance:

```text
true_bearing = atan2(reflector_height - radar_height, distance) - mount_deg
offset_needed = true_bearing - measured_raw_angle
```

If the offset is stable across several known reflector positions, it is a real
boresight/electrical-zero correction. If it changes substantially by height,
floor position, or distance, the setup may be affected by multipath or alignment
error.

## Known Gaps And Focus Areas

- Per-shot timing alignment remains the largest open risk. The current OPS
  transition estimate is better than blindly using the sound-trigger offset, but
  TrackMan review still found shots where replayed timing shifts recovered the
  geometry.
- A one-frame result is underconstrained. It is useful evidence that the radar
  saw something plausible, but it should not be counted the same way as
  two-frame geometry when judging algorithm accuracy.
- Second-frame selection needs more review. Some reviewed shots had an adjacent
  frame just below the relaxed SNR threshold or a high-RMSE pair where the
  strongest single frame looked better. Avoid loosening this globally until the
  TrackMan bucket analysis supports it.
- High-launch and off-boresight shots need more data. Shots launched around
  `20 deg+`, pushed far right, or curving right can reduce vertical SNR and make
  pair selection brittle.
- The indoor screen/net distance matters. With a 12 ft ball-to-screen flight,
  later frames can include screen, wall, or return clutter. Prefer early
  in-flight evidence before the expected screen-time boundary.
- USB reliability is part of angle accuracy. Two K-LD7 radars at 3 Mbaud plus
  OPS243 must all stream cleanly. Repeated short RADC payloads or a horizontal
  stream near `2 Hz` make the session unsuitable for selector tuning.
- Incorrect radar distance, mount tilt, height offset, or boresight offset can
  bias geometry even when timing is correct.
- Ground, mat, net, and side-wall multipath can bias measured bearing before the
  selector sees it.
- Driver and low-launch/high-speed data are still thin compared with 7-iron
  testing.

## Follow-Up TODOs

- Revisit single-frame anchor choice when no clean rising pair exists. The
  current anchor ranking prefers smallest OPS-bin error before SNR/coherence.
  TrackMan review of the 2026-05-30 7-iron session showed cases where that can
  select a later, lower-SNR frame over an earlier strong in-flight frame with
  comparable bin error. Candidate rule: when no adjacent rising pair survives,
  prefer an earlier strong frame if it is before the screen-time boundary, has a
  comparable bin error, and has materially better SNR/coherence.
- Tighten early-assisted pair acceptance. Shot review showed cases where an
  early frame alone matched TrackMan, but pairing it with the next frame pushed
  launch high even though RMSE stayed below the high-RMSE fallback threshold.
  Candidate rule: if the paired result diverges sharply from the strongest
  single-frame candidate or requires a large bearing jump, fall back to the
  strongest/earliest single frame and mark it as `geometry_single_frame`.
- Revisit second-frame SNR selection. A few reviewed shots had otherwise useful
  adjacent frames near the relaxed threshold. Any change should be tested by
  bucket: primary two-frame, timing-recoverable two-frame, one-frame diagnostic,
  and no-signal/clutter.
- Add local F1B ball-bin diagnostics for range-assisted selection. The current
  analysis CSV reports the strongest F1B peak and its bin error, but reviewed
  shots showed that the global F1B peak can land on near-DC/clutter while F1B
  still has usable SNR at or near the selected F1A/OPS ball bin. Track both the
  global F1B peak and a local F1B peak constrained around the selected ball bin
  before using F1B range as a confidence signal for one-frame recovery.
