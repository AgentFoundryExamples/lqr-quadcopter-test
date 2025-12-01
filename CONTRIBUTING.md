# Contributing

Thank you for your interest in contributing to the Quadcopter Target Tracking project!

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install development dependencies: `make dev-install`
4. Create a feature branch: `git checkout -b feature/your-feature`

## Development Workflow

### Running Tests

```bash
make test
```

### Linting and Formatting

```bash
make lint
make format
```

## Release Process

### Version Conflict Resolution

When multiple teams work on releases simultaneously:

1. Always pull the latest `main` branch before cutting a release
2. Use `git merge --no-ff` to preserve release history
3. If version conflicts occur, increment to the next available patch version
4. Update CHANGELOG entries to reflect the merged changes

### Cutting a Release

1. Ensure all tests pass: `make test`
2. Update version in `pyproject.toml`
3. Add changelog entry in `CHANGELOG.md`
4. Update `ROADMAP.md` if milestones are completed
5. Update version badges in `README.md`
6. Verify docs and Make targets reference the correct version:
   - Check `README.md` version badge matches `pyproject.toml`
   - Run `make help` and verify output reflects current features
   - Confirm example commands in docs work with the new version
7. Create a git tag: `git tag v<version>`
8. Push the tag: `git push origin v<version>`

## Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Write docstrings for public functions and classes
- Keep line length under 88 characters (configured in ruff)

## Submitting Changes

1. Commit your changes with clear, descriptive messages
2. Push to your fork
3. Open a Pull Request against the `main` branch
4. Ensure CI checks pass
5. Request review from maintainers
