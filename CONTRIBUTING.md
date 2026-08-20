# Contributing to nobg

Thanks for helping out. This file covers the mechanics; [`AGENTS.md`](AGENTS.md) is the normative
document for the architectural invariants a new model has to respect, and the same material is
published as a page on the docs site.

## Setup

```bash
git clone https://github.com/feyninc/nobg.git
cd nobg
uv sync
```

`uv sync` installs the `dev` group, which is where `torch`, `torchvision` and the ONNX extras live.
They are deliberately not `[project] dependencies` — an install picks up whatever torch build is
already in the environment instead of resolving one — so anything that actually runs the library needs
the dev group synced.

## Development

### Running tests

```bash
uv run pytest tests/
```

Tests are local-only: no network calls, `tmp_path` for filesystem work, and `@pytest.fixture` with
reduced-size configs for anything expensive. The ONNX round-trip tests in `tests/test_mixin.py` skip
themselves when `onnxruntime`/`onnxscript` are missing.

### Linting and type checking

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uvx ty check src/
```

All three, plus the test suite, run in CI on every push and pull request to `main`.

### Adding a new model

The short version — see [`AGENTS.md`](AGENTS.md) for the rules behind each step:

1. Create `src/nobg/<model_name>/modeling_<model_name>.py`.
2. Define a `<ModelName>Config` dataclass (every field defaulted, primitives and simple containers
   only) and a model class inheriting from `nn.Module` and `Revised_Mixin`. `forward` returns a dict
   with raw `(B, 1, H, W)` `logits`.
3. Pass `model_card_template()` from `utils.py` as the `model_card_template` class kwarg — do not write
   a custom template.
4. Add `src/nobg/<model_name>/image_processing_<model_name>.py`, exposing
   `post_process_alpha_matting`, `refine_foreground` and `cutout` by delegating to `utils.py`.
5. Give the model `predict(processor, image, ...)`, `process(image, ...)` and `default_processor()`.
6. Export the model and processor from `__init__.py`, and register both in `auto.py` (a new
   `elif "<tag>" in tags` branch plus a `PROCESSOR_TYPES` entry).
7. Add `tests/test_<model_name>.py`, `tests/test_image_processing_<model_name>.py`, and a class in
   `tests/test_mixin.py` with a torch-vs-onnxruntime parity assertion on `logits`. The required test
   matrix is in [`AGENTS.md`](AGENTS.md#tests).

For ONNX, a new model implements none of the export methods. It overrides at most
`onnx_dummy_inputs(batch_size)` (if `forward` needs more than `pixel_values`) and `onnx_dynamo` (if the
default exporter fails on it — record which error forced it, in a comment).

## Documentation

The docs site is a [Fumadocs](https://fumadocs.dev) app in [`docs/`](docs), deployed on Vercel.

```bash
cd docs
npm install
npm run dev     # http://localhost:3000
```

| Path                      | What it holds                                                  |
| ------------------------- | -------------------------------------------------------------- |
| `docs/content/docs/`      | Every page, as MDX. The file path is the URL                    |
| `docs/content/docs/*/meta.json` | Sidebar order and section titles                         |
| `docs/lib/shared.ts`      | Site name, repo coordinates, external links                     |
| `docs/lib/layout.shared.tsx` | Nav title, logo, top-level nav links                         |
| `docs/app/(home)/page.tsx`| Landing page                                                    |

Front matter is `title`, `description` and an optional `icon` (any [lucide](https://lucide.dev/icons)
export name). `Card`, `Cards` and `Callout` need no import; `Tabs`, `Steps`, `TypeTable` and
`Accordions` are registered in `docs/components/mdx.tsx` — add new components there rather than
importing them per page.

Before opening a docs PR:

```bash
cd docs
npm run build          # also type-checks
```

If you change the public API, update the matching page under `docs/content/docs/reference/`. Every page
has an "Edit on GitHub" link that points at its source file, so the fastest fix is often from the site
itself.

### Deployment

Vercel builds from this repo with **Root Directory** set to `docs`; the framework, build command and
output directory are all detected. `main` deploys to production and every other branch gets a preview
URL. Nothing in the Python package is part of that build.

## Pull requests

- Run the full lint, type and test suite before opening one.
- Keep each PR focused on a single change.
- Add tests for new functionality.
- Update the docs in the same PR when behaviour or the public API changes.
