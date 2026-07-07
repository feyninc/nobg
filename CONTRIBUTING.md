# Contributing to nobg

## Setup

```bash
git clone https://github.com/feyninc/nobg.git
cd nobg
uv sync
```

## Development

### Running tests

```bash
uv run pytest tests/
```

### Linting

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uvx ty check src/
```

### Adding a new model

1. Create `src/nobg/modeling_<name>.py`
2. Define a config dataclass and model class inheriting from `nn.Module` and `Revised_Mixin`
3. Use `model_card_template()` from `utils.py` for the HuggingFace model card
4. Register the model in `__init__.py` and `auto.py`
5. Add tests in `tests/`

## Pull requests

- Run the full lint and test suite before opening a PR
- Keep PRs focused on a single change
- Add tests for new functionality
