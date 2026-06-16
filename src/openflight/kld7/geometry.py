"""Geometry helpers for K-LD7 vertical launch-angle estimation."""

from __future__ import annotations

import math

MPH_TO_FTS = 1.46667
GEOM_BALL_ABOVE_RADAR_FT = -4.0 / 12.0  # ball sits ~4" below the radar center
GEOM_FLIGHT_T_MAX_S = 0.150  # ignore frames beyond plausible in-net flight time
GEOM_ALPHA_MIN_DEG = 0.0
GEOM_ALPHA_MAX_DEG = 45.0
GEOM_ALPHA_STEP_DEG = 0.1
GEOM_PAIR_SINGLE_FRAME_FALLBACK_RMSE_DEG = 4.0
GEOM_SINGLE_FRAME_MAX_BEARING_RESID_DEG = 1.0
GEOM_SINGLE_FRAME_CONFIDENCE_MAX = 0.72


def predicted_bearing_deg(
    alpha_deg: float,
    flight_time_s: float,
    ball_speed_mph: float,
    distance_ft: float,
    mount_deg: float,
    ball_above_radar_ft: float = GEOM_BALL_ABOVE_RADAR_FT,
) -> float:
    """Bearing the radar should measure for a ball launched at ``alpha_deg``.

    Models the ball as a point launched at the tee (``distance_ft`` downrange,
    ``ball_above_radar_ft`` vertically relative to the radar center) flying in a
    straight line at ``ball_speed_mph``. Gravity is negligible over the short
    K-LD7 in-flight window. Returns bearing in the radar frame with mount tilt
    subtracted.
    """
    v_fts = ball_speed_mph * MPH_TO_FTS
    alpha_rad = math.radians(alpha_deg)
    x_ft = distance_ft + v_fts * math.cos(alpha_rad) * flight_time_s
    y_ft = ball_above_radar_ft + v_fts * math.sin(alpha_rad) * flight_time_s
    return math.degrees(math.atan2(y_ft, x_ft)) - mount_deg


def fit_launch_angle_geometric(
    per_frame: list[tuple[float, float, float]],
    ball_speed_mph: float,
    distance_ft: float,
    mount_deg: float,
    ball_above_radar_ft: float = GEOM_BALL_ABOVE_RADAR_FT,
) -> tuple[float, float, int] | None:
    """Fit launch angle from per-frame ``(flight_time_s, bearing_deg, weight)``.

    Grid-searches the launch angle whose predicted bearing trajectory minimizes
    the weight-scaled squared bearing residual. Only frames inside the plausible
    in-flight window are used. Returns ``(alpha_deg, rmse_deg, n_used)`` or
    ``None`` when the fit is underdetermined.
    """
    pts = [
        (time_s, bearing_deg, max(float(weight), 0.0))
        for (time_s, bearing_deg, weight) in per_frame
        if time_s is not None and 0.0 < time_s <= GEOM_FLIGHT_T_MAX_S
    ]
    weight_sum = sum(weight for _, _, weight in pts)
    if len(pts) < 2 or weight_sum <= 0.0:
        return None

    best_alpha = GEOM_ALPHA_MIN_DEG
    best_ss = math.inf
    steps = int(round((GEOM_ALPHA_MAX_DEG - GEOM_ALPHA_MIN_DEG) / GEOM_ALPHA_STEP_DEG))
    for i in range(steps + 1):
        alpha = GEOM_ALPHA_MIN_DEG + i * GEOM_ALPHA_STEP_DEG
        ss = 0.0
        for time_s, bearing_deg, weight in pts:
            resid = bearing_deg - predicted_bearing_deg(
                alpha,
                time_s,
                ball_speed_mph,
                distance_ft,
                mount_deg,
                ball_above_radar_ft,
            )
            ss += weight * resid * resid
        if ss < best_ss:
            best_ss = ss
            best_alpha = alpha

    rmse = math.sqrt(best_ss / weight_sum)
    return float(best_alpha), float(rmse), len(pts)


def fit_launch_angle_single_frame_geometric(
    frame: tuple[float, float, float],
    ball_speed_mph: float,
    distance_ft: float,
    mount_deg: float,
    ball_above_radar_ft: float = GEOM_BALL_ABOVE_RADAR_FT,
) -> tuple[float, float] | None:
    """Estimate launch angle from one in-flight bearing.

    This is lower confidence than the multi-frame trajectory fit because timing
    and bearing noise cannot be averaged out. Returns
    ``(alpha_deg, bearing_residual_deg)`` when the residual is acceptable.
    """
    time_s, bearing_deg, weight = frame
    if time_s is None or not (0.0 < time_s <= GEOM_FLIGHT_T_MAX_S) or weight <= 0.0:
        return None

    best_alpha = GEOM_ALPHA_MIN_DEG
    best_resid = math.inf
    steps = int(round((GEOM_ALPHA_MAX_DEG - GEOM_ALPHA_MIN_DEG) / GEOM_ALPHA_STEP_DEG))
    for i in range(steps + 1):
        alpha = GEOM_ALPHA_MIN_DEG + i * GEOM_ALPHA_STEP_DEG
        resid = abs(
            bearing_deg
            - predicted_bearing_deg(
                alpha,
                time_s,
                ball_speed_mph,
                distance_ft,
                mount_deg,
                ball_above_radar_ft,
            )
        )
        if resid < best_resid:
            best_resid = resid
            best_alpha = alpha

    if best_resid > GEOM_SINGLE_FRAME_MAX_BEARING_RESID_DEG:
        return None
    return float(best_alpha), float(best_resid)
