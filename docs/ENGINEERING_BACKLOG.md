# Engineering Backlog

Last updated: February 25, 2026
Source: Deep-dive code review of the `openflight` fork

## Overall Assessment

The core radar and signal-processing direction is strong, but the repository has elevated deployment risk from unauthenticated network control paths and moderate maintainability risk from a monolithic server module.

## Prioritized Backlog

### P0 - Lock down network control surface (security)

Status: Open

Issue:
- The web server currently binds with permissive CORS and exposes hardware-control Socket.IO events without authentication.

Evidence:
- `src/openflight/server.py:48`
- `src/openflight/server.py:544`
- `src/openflight/server.py:545`

Why this matters:
- Any host on the same network can potentially send commands that modify radar behavior.

Acceptance criteria:
- Socket events that change hardware/config state require authentication and authorization.
- `cors_allowed_origins` is restricted to trusted origins.
- Security behavior is covered by automated tests.

### P0 - Protect camera streaming and camera controls (security/privacy)

Status: Open

Issue:
- Camera stream and camera toggle pathways are network-accessible without auth gating.

Evidence:
- `src/openflight/server.py:255`
- `src/openflight/server.py:268`
- `src/openflight/server.py:289`
- `src/openflight/server.py:290`

Why this matters:
- Remote clients can enable/disable streaming and access live camera frames.

Acceptance criteria:
- Camera stream endpoint requires authenticated access.
- Camera toggle events are permission-gated.
- Unauthorized access attempts are rejected and logged.

### P1 - Ensure rolling-buffer trigger resets on failed captures (correctness)

Status: Open

Issue:
- The capture loop can `continue` without trigger reset when capture processing fails, which can leave trigger state stale.

Evidence:
- `src/openflight/rolling_buffer/monitor.py:387`
- `src/openflight/rolling_buffer/monitor.py:390`
- `src/openflight/rolling_buffer/monitor.py:420`
- `src/openflight/rolling_buffer/monitor.py:562`

Why this matters:
- A failure path can degrade or stall shot detection until manual recovery.

Acceptance criteria:
- Trigger reset executes on all exit paths in the capture cycle (including errors/early continues).
- Add regression test that simulates `process_capture(...)` returning `None`.

### P1 - Expand integration tests around server + hardware orchestration

Status: Open

Issue:
- Current tests are strong on pure logic but light on end-to-end server behavior and hardware-facing orchestration.

Evidence:
- `tests/test_server.py:1`
- `src/openflight/server.py:239`
- `src/openflight/server.py:321`
- `src/openflight/ops243.py:762`
- `src/openflight/ops243.py:1269`

Why this matters:
- Regressions in high-risk runtime paths may pass CI undetected.

Acceptance criteria:
- Add integration tests for Socket.IO handlers and critical HTTP routes.
- Add tests for OPS243 streaming and non-blocking read orchestration with mocks/fakes.
- CI gates include these tests.

### P2 - Decompose `server.py` and reduce global mutable state (maintainability)

Status: Open

Issue:
- `server.py` currently mixes web routing, lifecycle, camera management, monitor startup, and session concerns with module-level mutable state.

Evidence:
- `src/openflight/server.py:51`
- `src/openflight/server.py:58`
- `src/openflight/server.py:691`
- `src/openflight/server.py:900`

Why this matters:
- Increases regression risk and makes focused unit testing/refactoring harder.

Acceptance criteria:
- Split responsibilities into smaller modules (for example: server app, monitor lifecycle, camera manager, session orchestration).
- Replace module-level globals with explicit application context/state objects where feasible.
- Preserve existing external behavior and CLI options.

### P2 - Decouple shot processing from side-effect logging/output (maintainability/testability)

Status: Open

Issue:
- `LaunchMonitor` shot processing includes substantial direct `print(...)` side effects inside core flow.

Evidence:
- `src/openflight/launch_monitor.py:444`
- `src/openflight/launch_monitor.py:550`
- `src/openflight/launch_monitor.py:631`

Why this matters:
- Harder to reuse/test core logic without inheriting presentation/logging noise.

Acceptance criteria:
- Move verbose output to structured logger with configurable levels.
- Keep core shot computation path deterministic and side-effect-light.
- Add tests that assert behavior independent of logging output.

### P3 - Sync architecture docs with runtime reality

Status: Open

Issue:
- README architecture narrative presents a cleaner callback-only model than current runtime responsibilities.

Evidence:
- `README.md:214`
- `README.md:235`
- `src/openflight/server.py:691`

Why this matters:
- New contributors can be misled about actual boundaries and coupling.

Acceptance criteria:
- Update README architecture section to reflect current modules and responsibilities.
- Keep diagrams and text aligned with implemented server flow.

### P2 - Integrate OpenGolfCoach for physics-based shot analysis

Status: Open
Fork: https://github.com/jswans33/open-golf-coach (forked from OpenLaunchLabs/open-golf-coach)
License: Apache 2.0

Issue:
- OpenFlight currently uses a simple ball-speed-only carry distance estimate. No spin decomposition, shot classification, or physics trajectory modeling.

What OpenGolfCoach provides:
- Physics-based trajectory: carry distance, total distance, offline deviation, hang time, peak height, descent angle, landing position/velocity — from ball speed, launch angle, and spin data.
- Spin analysis: converts between total spin/axis and backspin/sidespin components.
- Shot classification: deterministic shot names (Straight, Draw, Fade, Hook, Slice, Duck Hook, Shank, etc.), quality ranks (S+ through E), and hex colors for UI rendering.
- Club analytics: estimated club speed, smash factor, club path, face-to-path angle.
- Python bindings via PyO3 (published as `opengolfcoach` on PyPI, also buildable from our fork with `maturin`).

Integration plan:
1. Add `opengolfcoach` to `pyproject.toml` dependencies (use PyPI package, fall back to fork build if needed on Pi/ARM).
2. In `launch_monitor.py` where `Shot` objects are created, call `opengolfcoach.calculate_derived_values()` with available shot data (ball_speed, club_speed, launch_angle if camera provides it, spin_rpm if rolling buffer provides it).
3. Extend `Shot` dataclass with new fields: carry_distance_physics, total_distance, offline_distance, shot_name, shot_rank, shot_color, backspin, sidespin, hang_time, peak_height.
4. Emit enriched shot data over WebSocket.
5. Update React UI to display shot classification (name + rank + color), physics-based distances, and spin breakdown.
6. Graceful degradation: if only ball_speed is available (no camera/spin), pass what we have — OpenGolfCoach will calculate what it can.

Why this matters:
- Transforms OpenFlight from a speed-only monitor into a full shot analyzer as more sensors come online (camera for launch angle, rolling buffer for spin).

Acceptance criteria:
- Shots include physics-based carry/total distance when launch angle data is available.
- Shots include shot classification name, rank, and color in WebSocket events and UI.
- Spin decomposition shown in UI when spin data is available.
- Falls back gracefully to current estimation when minimal data is available.
- Unit tests cover the integration boundary (mock OpenGolfCoach responses).
- ARM (Pi) build verified — either PyPI wheel available or fork builds with maturin.

## Suggested Execution Order

1. Security hardening for Socket.IO and camera endpoints (both P0 items).
2. Rolling-buffer trigger reset fix + regression coverage.
3. Server and OPS243 integration tests for critical paths.
4. `server.py` decomposition and logging decoupling in focused refactors.
5. OpenGolfCoach integration for physics-based shot analysis.
6. Documentation synchronization after architecture refactor boundaries are settled.
