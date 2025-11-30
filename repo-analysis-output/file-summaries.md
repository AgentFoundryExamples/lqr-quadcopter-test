# File Summaries

Heuristic summaries of source files based on filenames, extensions, and paths.

Schema Version: 2.0

Total files: 10

## src/quadcopter_tracking/__init__.py
**Language:** Python  
**Role:** module-init  
**Role Justification:** module initialization file '__init__'  
**Size:** 0.83 KB  
**LOC:** 33  
**TODOs/FIXMEs:** 0  

## src/quadcopter_tracking/controllers/__init__.py
**Language:** Python  
**Role:** module-init  
**Role Justification:** module initialization file '__init__'  
**Size:** 4.46 KB  
**LOC:** 113  
**TODOs/FIXMEs:** 0  
**Declarations:** 3  
**Top-level declarations:**
  - class BaseController
  - class LQRController
  - class PIDController

## src/quadcopter_tracking/env/__init__.py
**Language:** Python  
**Role:** module-init  
**Role Justification:** module initialization file '__init__'  
**Size:** 1.94 KB  
**LOC:** 62  
**TODOs/FIXMEs:** 0  

## src/quadcopter_tracking/env/config.py
**Language:** Python  
**Role:** configuration  
**Role Justification:** configuration file name 'config'  
**Size:** 8.92 KB  
**LOC:** 179  
**TODOs/FIXMEs:** 0  
**Declarations:** 6  
**Top-level declarations:**
  - class QuadcopterParams
  - class SimulationParams
  - class TargetParams
  - class SuccessCriteria
  - class LoggingParams
  - class EnvConfig
**External Dependencies:**
  - **Stdlib:** `dataclasses.dataclass`, `dataclasses.field`

## src/quadcopter_tracking/env/quadcopter_env.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 20.82 KB  
**LOC:** 470  
**TODOs/FIXMEs:** 0  
**Declarations:** 1  
**Top-level declarations:**
  - class QuadcopterEnv
**External Dependencies:**
  - **Stdlib:** `logging`, `math`
  - **Third-party:** `numpy`

## src/quadcopter_tracking/env/target_motion.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 12.18 KB  
**LOC:** 335  
**TODOs/FIXMEs:** 0  
**Declarations:** 7  
**Top-level declarations:**
  - class MotionPattern
  - class LinearMotion
  - class CircularMotion
  - class SinusoidalMotion
  - class Figure8Motion
  - class StationaryMotion
  - class TargetMotion
**External Dependencies:**
  - **Stdlib:** `math`, `typing.Protocol`
  - **Third-party:** `numpy`

## src/quadcopter_tracking/utils/__init__.py
**Language:** Python  
**Role:** module-init  
**Role Justification:** module initialization file '__init__'  
**Size:** 11.56 KB  
**LOC:** 306  
**TODOs/FIXMEs:** 0  
**Declarations:** 7  
**Top-level declarations:**
  - function get_default_config
  - function load_config
  - function _deep_merge
  - function _json_serializer
  - function _apply_env_overrides
  - class DataLogger
  - class Plotter
**External Dependencies:**
  - **Stdlib:** `datetime`, `json`, `logging`, `os`, `pathlib.Path`
  - **Third-party:** `dotenv.load_dotenv`, `matplotlib.pyplot`, `numpy`, `yaml`

## tests/__init__.py
**Language:** Python  
**Role:** test  
**Role Justification:** located in 'tests' directory  
**Size:** 0.06 KB  
**LOC:** 1  
**TODOs/FIXMEs:** 0  

## tests/test_config.py
**Language:** Python  
**Role:** test  
**Role Justification:** filename starts with 'test_'  
**Size:** 1.83 KB  
**LOC:** 39  
**TODOs/FIXMEs:** 0  
**Declarations:** 5  
**Top-level declarations:**
  - function test_get_default_config_has_required_keys
  - function test_get_default_config_values
  - function test_load_config_without_file
  - function test_load_config_env_override
  - function test_load_config_target_env_override
**External Dependencies:**
  - **Stdlib:** `os`, `unittest.mock`

## tests/test_env_dynamics.py
**Language:** Python  
**Role:** test  
**Role Justification:** filename starts with 'test_'  
**Size:** 17.36 KB  
**LOC:** 396  
**TODOs/FIXMEs:** 0  
**Declarations:** 4  
**Top-level declarations:**
  - class TestTargetMotion
  - class TestQuadcopterEnv
  - class TestEnvConfig
  - class TestIntegration
**External Dependencies:**
  - **Third-party:** `numpy`, `pytest`
