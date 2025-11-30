# File Summaries

Heuristic summaries of source files based on filenames, extensions, and paths.

Schema Version: 2.0

Total files: 6

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
**Size:** 3.32 KB  
**LOC:** 91  
**TODOs/FIXMEs:** 0  
**Declarations:** 2  
**Top-level declarations:**
  - class QuadcopterEnv
  - class TargetMotion

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
