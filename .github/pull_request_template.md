## What does this PR do?

<!-- Describe the change and why it's needed. Reference related issues with "Fixes #123". -->

## How was this tested?

<!-- How did you verify this works? Mock mode, hardware, specific test cases? -->

## Checklist

- [ ] Python tests pass (`uv run pytest tests/ -v`)
- [ ] Pylint passes (`uv run pylint src/openflight/ --fail-under=9`)
- [ ] Ruff passes (`uv run ruff check src/openflight/`)
- [ ] UI builds (`cd ui && npm run build`)
- [ ] UI lint passes (`cd ui && npm run lint`)
- [ ] Updated docs or CHANGELOG if needed
- [ ] No unrelated changes mixed in
