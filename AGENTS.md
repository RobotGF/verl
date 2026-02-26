# AGENTS.md

## Cursor Cloud specific instructions

### Overview

**verl** (Volcano Engine Reinforcement Learning for LLMs) is a Python library for RL post-training of large language models. It is **not** a web application or microservice — it is a single installable Python package.

### Development setup

Install in editable mode: `pip install -e ".[test]"` followed by `pip install --upgrade "transformers<5.0.0"`. The transformers pin is required because the package declares `transformers<5.0.0` but pip may resolve a newer version.

Also install `pyzmq` which is an undeclared transitive dependency needed by some vLLM-related modules.

### Linting

- Linter: **ruff** (configured in `pyproject.toml`). Run: `ruff check verl/ tests/ examples/ scripts/`
- Full pre-commit suite (ruff, ruff-format, mypy, autogen-trainer-cfg, check-docstrings, check-license, compileall): `pre-commit run --all-files`
- See `CONTRIBUTING.md` for details.

### Testing

- **CPU tests** (no GPU required): create a `pytest.ini` with `python_files = *_on_cpu.py`, then run `pytest -s --asyncio-mode=auto tests/`
- Skip `tests/workers/rollout/test_vllm_cli_args_on_cpu.py` — it fails to collect without vLLM installed.
- Some CPU tests require pre-downloaded HuggingFace models at `~/models/` and datasets; these will fail in environments without those assets. This is expected.
- **GPU tests** require NVIDIA GPUs and are not runnable in this cloud environment.

### Key gotchas

- **No GPU in Cloud Agent VM**: This codebase is GPU-focused (RL training for LLMs). Full end-to-end training and GPU tests cannot run without NVIDIA GPUs. CPU unit tests and data preprocessing scripts are the primary validation path.
- **Ray warnings about /dev/shm**: Ray will warn about small `/dev/shm` in Docker containers. This is cosmetic and does not affect CPU test correctness.
- The `transformers` version must be pinned to `<5.0.0`; the latest `transformers` 5.x breaks compatibility.
