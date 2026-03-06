# Contributing to Experanto

## Code of Conduct

This project follows the [Contributor Covenant v3.0](https://www.contributor-covenant.org/version/3/0/code_of_conduct/). By participating, you are expected to uphold it.

## Setup

```bash
git clone https://github.com/sensorium-competition/experanto.git
cd experanto
pip install -e ".[dev]"
```

## Code Style

We use [black](https://black.readthedocs.io/), [isort](https://pycqa.github.io/isort/), and [pyright](https://github.com/microsoft/pyright) — all configured in `pyproject.toml`. Docstrings should follow NumPy style.

## Tests

Tests live in `tests/`. Synthetic data fixtures are created via context managers in `tests/create_*.py` — follow this pattern for new tests so no real data files are left on disk.

```bash
pytest                                       # all tests
pytest tests/test_sequence_interpolator.py  # single file
```

## Documentation

All sources are in the `docs/` folder. To build locally:

1. Install Sphinx and the ReadTheDocs theme:
   ```bash
   pip install sphinx sphinx-rtd-theme
   ```
2. From inside the `docs/` folder:
   ```bash
   make clean html
   ```
3. Open `docs/build/html/index.html` in your browser.

## Pull Requests

- Keep PRs focused; one logical change per PR.
- All existing tests must pass; add tests for new behaviour.
- Code style checks and type checking must pass.
- Keep documentation up to date — add, update, or remove docstrings and `docs/source/` pages to reflect your changes.