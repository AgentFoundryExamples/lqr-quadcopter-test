# Dependency Graph

Intra-repository dependency analysis for Python and JavaScript/TypeScript files.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 19
- **Intra-repo dependencies**: 26
- **External stdlib dependencies**: 19
- **External third-party dependencies**: 9

## External Dependencies

### Standard Library / Core Modules

Total: 19 unique modules

- `argparse`
- `collections.abc.Callable`
- `csv`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime`
- `datetime.datetime`
- `datetime.timezone`
- `json`
- `logging`
- `math`
- `os`
- `pathlib.Path`
- `sys`
- `tempfile`
- `typing.Any`
- `typing.Literal`
- `typing.Protocol`
- `unittest.mock`

### Third-Party Packages

Total: 9 unique packages

- `dotenv.load_dotenv`
- `matplotlib`
- `matplotlib.pyplot`
- `numpy`
- `pytest`
- `torch`
- `torch.nn`
- `torch.optim`
- `yaml`

## Most Depended Upon Files (Intra-Repo)

- `src/quadcopter_tracking/env/__init__.py` (6 dependents)
- `src/quadcopter_tracking/controllers/__init__.py` (4 dependents)
- `src/quadcopter_tracking/utils/metrics.py` (3 dependents)
- `src/quadcopter_tracking/utils/diagnostics.py` (3 dependents)
- `src/quadcopter_tracking/utils/losses.py` (3 dependents)
- `src/quadcopter_tracking/utils/__init__.py` (2 dependents)
- `src/quadcopter_tracking/env/config.py` (1 dependents)
- `src/quadcopter_tracking/env/target_motion.py` (1 dependents)
- `src/quadcopter_tracking/controllers/deep_tracking_policy.py` (1 dependents)
- `src/quadcopter_tracking/eval.py` (1 dependents)

## Files with Most Dependencies (Intra-Repo)

- `tests/test_training_loop.py` (5 dependencies)
- `src/quadcopter_tracking/train.py` (4 dependencies)
- `tests/test_eval.py` (4 dependencies)
- `src/quadcopter_tracking/__init__.py` (3 dependencies)
- `src/quadcopter_tracking/eval.py` (3 dependencies)
- `src/quadcopter_tracking/utils/__init__.py` (3 dependencies)
- `src/quadcopter_tracking/env/__init__.py` (2 dependencies)
- `tests/test_config.py` (1 dependencies)
- `tests/test_env_dynamics.py` (1 dependencies)
