# Dependency Graph

Intra-repository dependency analysis for Python and JavaScript/TypeScript files.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 25
- **Intra-repo dependencies**: 39
- **External stdlib dependencies**: 21
- **External third-party dependencies**: 11

## External Dependencies

### Standard Library / Core Modules

Total: 21 unique modules

- `argparse`
- `collections.abc.Callable`
- `csv`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime`
- `datetime.datetime`
- `datetime.timezone`
- `itertools.product`
- `json`
- `logging`
- `math`
- `os`
- `pathlib.Path`
- `signal`
- `sys`
- `tempfile`
- `typing.Any`
- `typing.Literal`
- `typing.Protocol`
- ... and 1 more (see JSON for full list)

### Third-Party Packages

Total: 11 unique packages

- `dotenv.load_dotenv`
- `matplotlib`
- `matplotlib.pyplot`
- `numpy`
- `pytest`
- `scipy.linalg.solve_discrete_are`
- `torch`
- `torch.nn`
- `torch.nn.functional`
- `torch.optim`
- `yaml`

## Most Depended Upon Files (Intra-Repo)

- `src/quadcopter_tracking/controllers/__init__.py` (7 dependents)
- `src/quadcopter_tracking/env/__init__.py` (7 dependents)
- `src/quadcopter_tracking/utils/metrics.py` (4 dependents)
- `src/quadcopter_tracking/controllers/tuning.py` (3 dependents)
- `src/quadcopter_tracking/utils/__init__.py` (3 dependents)
- `src/quadcopter_tracking/utils/diagnostics.py` (3 dependents)
- `src/quadcopter_tracking/utils/losses.py` (3 dependents)
- `src/quadcopter_tracking/utils/coordinate_frame.py` (2 dependents)
- `src/quadcopter_tracking/train.py` (2 dependents)
- `src/quadcopter_tracking/eval.py` (2 dependents)

## Files with Most Dependencies (Intra-Repo)

- `tests/test_env_dynamics.py` (6 dependencies)
- `tests/test_training_loop.py` (5 dependencies)
- `src/quadcopter_tracking/train.py` (4 dependencies)
- `src/quadcopter_tracking/utils/__init__.py` (4 dependencies)
- `tests/test_eval.py` (4 dependencies)
- `src/quadcopter_tracking/__init__.py` (3 dependencies)
- `src/quadcopter_tracking/controllers/tuning.py` (3 dependencies)
- `src/quadcopter_tracking/eval.py` (3 dependencies)
- `src/quadcopter_tracking/controllers/__init__.py` (2 dependencies)
- `src/quadcopter_tracking/env/__init__.py` (2 dependencies)
