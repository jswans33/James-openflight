"""Tests for the geometric vertical launch-angle fit."""

import math

from openflight.kld7.geometry import (
    GEOM_BALL_ABOVE_RADAR_FT,
    fit_launch_angle_geometric,
    predicted_bearing_deg,
)

V_MPH = 100.0
D_FT = 5.5
MOUNT_DEG = 18.0


def _synthetic_frames(alpha_true, times_s, v=V_MPH, d=D_FT, mount=MOUNT_DEG, weight=1.0):
    """Per-frame (t, bearing, weight) generated from a known launch angle."""
    return [(t, predicted_bearing_deg(alpha_true, t, v, d, mount), weight) for t in times_s]


def test_predicted_bearing_is_shallower_than_launch_angle():
    # Core physics: early in flight the bearing (in world frame = measured+mount)
    # is well below the true launch angle because the ball is still low and close.
    alpha = 16.0
    t = 0.045
    measured = predicted_bearing_deg(alpha, t, V_MPH, D_FT, MOUNT_DEG)
    world_bearing = measured + MOUNT_DEG  # undo the mount frame
    assert world_bearing < alpha - 3.0  # several degrees shallow (the ~7° effect)


def test_round_trip_recovers_launch_angle():
    for alpha_true in (8.0, 12.0, 16.0, 22.0, 30.0):
        frames = _synthetic_frames(alpha_true, [0.045, 0.070])
        result = fit_launch_angle_geometric(frames, V_MPH, D_FT, MOUNT_DEG)
        assert result is not None
        alpha_fit, rmse, n_used = result
        assert n_used == 2
        assert abs(alpha_fit - alpha_true) < 0.2  # within grid resolution
        assert rmse < 1e-3  # noiseless data fits essentially perfectly


def test_distance_is_a_weak_lever():
    # Generate bearings at the true distance, then fit with D off by +/-1 ft.
    alpha_true = 16.0
    true_d = 6.0
    frames = _synthetic_frames(alpha_true, [0.045, 0.070], d=true_d)
    for wrong_d in (true_d - 1.0, true_d + 1.0):
        alpha_fit, _, _ = fit_launch_angle_geometric(frames, V_MPH, wrong_d, MOUNT_DEG)
        # A 1 ft distance error should move the estimate by well under a degree.
        assert abs(alpha_fit - alpha_true) < 1.0


def test_returns_none_with_fewer_than_two_inflight_frames():
    # Single in-flight frame -> underdetermined -> None.
    one = _synthetic_frames(16.0, [0.050])
    assert fit_launch_angle_geometric(one, V_MPH, D_FT, MOUNT_DEG) is None
    # Empty -> None.
    assert fit_launch_angle_geometric([], V_MPH, D_FT, MOUNT_DEG) is None


def test_frames_outside_flight_window_are_ignored():
    good = _synthetic_frames(16.0, [0.045, 0.070])
    # Add a pre-impact frame (t<=0) and a far-future frame (> window); both ignored.
    noisy = [(-0.010, 5.0, 1.0), (0.500, 40.0, 1.0)] + good
    result = fit_launch_angle_geometric(noisy, V_MPH, D_FT, MOUNT_DEG)
    assert result is not None
    alpha_fit, _, n_used = result
    assert n_used == 2  # only the two in-window frames counted
    assert abs(alpha_fit - 16.0) < 0.2


def test_zero_weight_frames_do_not_dominate():
    # A wildly wrong bearing with zero weight must not pull the fit.
    frames = _synthetic_frames(16.0, [0.045, 0.070]) + [(0.060, -45.0, 0.0)]
    alpha_fit, _, _ = fit_launch_angle_geometric(frames, V_MPH, D_FT, MOUNT_DEG)
    assert abs(alpha_fit - 16.0) < 0.3


def test_ball_above_radar_default_is_four_inches_below():
    assert math.isclose(GEOM_BALL_ABOVE_RADAR_FT, -4.0 / 12.0)
