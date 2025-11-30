# Dependency Graph

Intra-repository dependency analysis for Python and JavaScript/TypeScript files.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 10
- **Intra-repo dependencies**: 7
- **External stdlib dependencies**: 10
- **External third-party dependencies**: 5

## External Dependencies

### Standard Library / Core Modules

Total: 10 unique modules

- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime`
- `json`
- `logging`
- `math`
- `os`
- `pathlib.Path`
- `typing.Protocol`
- `unittest.mock`

### Third-Party Packages

Total: 5 unique packages

- `dotenv.load_dotenv`
- `matplotlib.pyplot`
- `numpy`
- `pytest`
- `yaml`

## Most Depended Upon Files (Intra-Repo)

- `src/quadcopter_tracking/env/__init__.py` (2 dependents)
- `src/quadcopter_tracking/utils/__init__.py` (2 dependents)
- `src/quadcopter_tracking/controllers/__init__.py` (1 dependents)
- `src/quadcopter_tracking/env/config.py` (1 dependents)
- `src/quadcopter_tracking/env/target_motion.py` (1 dependents)

## Files with Most Dependencies (Intra-Repo)

- `src/quadcopter_tracking/__init__.py` (3 dependencies)
- `src/quadcopter_tracking/env/__init__.py` (2 dependencies)
- `tests/test_config.py` (1 dependencies)
- `tests/test_env_dynamics.py` (1 dependencies)
