# Contributing to OpenFlight

Thank you for your interest in contributing to OpenFlight! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Node.js 20+ (for UI development)
- Git
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jewbetcha/openflight.git
   cd openflight
   ```

2. **Install dependencies**
   ```bash
   make install
   ```

   This installs Python dependencies (including dev tools) and UI dependencies.

   Or manually:
   ```bash
   uv sync --group dev
   cd ui && npm install
   ```

3. **Install pre-commit hooks**
   ```bash
   make hooks
   ```

   This runs ruff, pylint, and ESLint automatically before each commit.

4. **Build the UI** (for frontend development)
   ```bash
   make build-ui
   ```

### Running in Development

```bash
# Run server in mock mode (no hardware needed)
make dev

# Or manually:
scripts/start-kiosk.sh --mock

# Run UI development server (separate terminal)
cd ui && npm run dev
```

## Code Quality Standards

All checks can be run at once with:

```bash
make lint
```

### Python

We use **pylint** for linting with a minimum score of **9.0** and **ruff** for formatting.

```bash
# Check code quality
uv run pylint src/openflight/

# Auto-format with ruff
make format
```

### TypeScript/React

```bash
cd ui
npm run lint      # ESLint
npm run build     # Type check + build
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_launch_monitor.py -v

# Run with coverage (if pytest-cov installed)
uv run pytest tests/ --cov=src/openflight --cov-report=html
```

**All tests must pass before submitting a PR.**

## Submitting Changes

### Reporting Issues

Use the [issue templates](https://github.com/jewbetcha/openflight/issues/new/choose) to file bugs, request features, or get help with hardware setup. Check existing issues before creating new ones.

### Pull Request Process

1. **Fork the repository** and create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, focused commits

3. **Ensure quality checks pass**
   ```bash
   make test
   make lint
   cd ui && npm run build
   ```

4. **Update documentation** if needed
   - Update README.md for user-facing changes
   - Update relevant docs in `docs/`
   - Add entry to `docs/CHANGELOG.md` under `[Unreleased]`

5. **Submit a pull request** and fill out the PR template

### Commit Messages

Use clear, descriptive commit messages:

```
Add ball detection indicator to UI header

- Create BallDetectionIndicator component
- Add shot data to useSocket hook
- Update App.tsx to display indicator
```

### What We're Looking For

**High-priority contributions:**
- Bug fixes with tests
- Documentation improvements
- Performance optimizations
- Test coverage improvements

**Feature ideas:**
- Launch angle detection improvements
- Better carry distance models
- Mobile app / Bluetooth support
- Integration with golf simulation software

## Project Structure

```
openflight/
├── src/openflight/       # Python package
│   ├── ops243.py         # Radar driver
│   ├── launch_monitor.py # Shot detection
│   ├── server.py         # WebSocket server
│   ├── kld7/             # K-LD7 angle radar
│   └── rolling_buffer/   # Spin detection
├── ui/                   # React frontend
│   └── src/
│       ├── components/   # UI components
│       └── hooks/        # React hooks
├── tests/                # Test suite
├── scripts/
│   ├── start-kiosk.sh    # Main startup script
│   ├── analysis/         # Post-session analysis & data capture
│   ├── hardware-test/    # Radar & trigger testing/debugging
│   ├── setup/            # Pi setup, systemd, deployment
│   └── vision/           # Camera, YOLO, ML training
├── models/               # ML models
└── docs/                 # Documentation
```

## Testing Without Hardware

OpenFlight supports **mock mode** for development without hardware:

```bash
make dev
```

The `MockLaunchMonitor` class simulates realistic shot data based on TrackMan averages.

## Questions?

- Use the [issue templates](https://github.com/jewbetcha/openflight/issues/new/choose) for bugs, features, or hardware help
- Check existing issues before creating new ones
- Be respectful and constructive in discussions

## License

By contributing, you agree that your contributions will be licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
