# AGENTS.md

## Cursor Cloud specific instructions

### Overview

**verl** (Volcano Engine Reinforcement Learning for LLMs) is a Python RL training library for large language models. It is a single Python package (not a web app or microservice) — there is no server to start. Development involves editing Python code, running lint, and running tests.

### Key Commands

- **Lint**: `pre-commit run --all-files` (runs ruff, ruff-format, mypy, autogen-trainer-cfg, check-docstrings, check-license, compileall). Individual hooks: `pre-commit run ruff --all-files`
- **CPU unit tests**: Create a temporary `pytest.ini` with `python_files = *_on_cpu.py`, then run `pytest -s -x --asyncio-mode=auto tests/`. See `.github/workflows/cpu_unit_tests.yml` for the CI pattern.
- **Install (editable)**: `pip install -e ".[test]"` then `pip install "transformers<5.0.0"` (the upper bound is important — `setup.py` declares `transformers` without an upper bound but the project requires `<5.0.0`).

### Caveats (CPU-only environment)

- **No GPU available**: GPU-dependent tests (under `tests/special_distributed/`, `tests/special_e2e/`, `tests/special_standalone/`) cannot run. Only `*_on_cpu.py` tests run.
- **vLLM/SGLang not installed**: `tests/workers/rollout/test_vllm_cli_args_on_cpu.py` will fail to collect because it imports from the vLLM rollout module which requires vLLM. Exclude it with `--ignore=tests/workers/rollout/test_vllm_cli_args_on_cpu.py`.
- **flash_attn not installed**: `tests/utils/test_padding_on_cpu.py` requires `flash_attn` (GPU-only C extension). Exclude it.
- **Local model/dataset tests**: Several dataset tests (`test_multiturn_sft_dataset_on_cpu.py`, `test_rl_dataset_on_cpu.py`, `test_sft_dataset_on_cpu.py`) and `test_rate_limited_reward_manager_on_cpu.py` expect pre-downloaded models/datasets at `~/models/`. The CI downloads them first; without those files, these tests fail. Exclude them in a CPU-only environment without models.
- **PyTorch CPU-only**: Install with `pip install torch --index-url https://download.pytorch.org/whl/cpu` to avoid pulling CUDA dependencies.
- **pyzmq**: Required by some verl modules (vLLM rollout uses ZMQ). Install `pyzmq` even without full vLLM.
- **pre-commit hooks**: Run `git config --unset-all core.hooksPath` before `pre-commit install` if the git config has `core.hooksPath` set.
- **transformers version**: Always pin `transformers<5.0.0` after installing verl. The `setup.py` base deps don't enforce the upper bound, but the project needs it.
