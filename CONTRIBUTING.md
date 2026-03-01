# Contributing

Thanks for your interest in contributing to `crewai-a2a-settlement`.

This package bridges [CrewAI](https://github.com/crewAIInc/crewAI) with the [A2A Settlement Exchange](https://github.com/a2a-settlement/a2a-settlement). It follows the same governance and contribution patterns as the core a2a-settlement org.

## Where to contribute

- **Bugs**: open an Issue with reproduction steps.
- **Feature requests**: open a Discussion first if it affects the API surface.
- **SDK/client improvements**: PRs welcome for `crewai_a2a_settlement/`.

## How to propose a new feature

1. Open a GitHub Discussion with a description and motivation.
2. Get feedback from maintainers.
3. Submit an implementation PR with tests.
4. Link to any related a2a-settlement spec changes if applicable.

## Development setup

Use a clean Python environment (venv/conda):

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Running tests

```bash
python -m pytest -q
```

With coverage:

```bash
python -m pytest --cov=crewai_a2a_settlement --cov-report=xml --cov-report=term-missing -q
```

CI-parity run (recommended before push):

```bash
./ci-local.sh
```

## Code style

- Keep code small and readable.
- Run `ruff check crewai_a2a_settlement/ tests/` before committing.
- Avoid heavy dependencies unless they materially improve clarity or correctness.
