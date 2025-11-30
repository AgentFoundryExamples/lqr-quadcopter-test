# File Summaries

Heuristic summaries of source files based on filenames, extensions, and paths.

Schema Version: 2.0

Total files: 15

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
**Size:** 3.43 KB  
**LOC:** 89  
**TODOs/FIXMEs:** 0  
**Declarations:** 2  
**Top-level declarations:**
  - class LQRController
  - class PIDController

## src/quadcopter_tracking/controllers/base.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 1.29 KB  
**LOC:** 36  
**TODOs/FIXMEs:** 0  
**Declarations:** 1  
**Top-level declarations:**
  - class BaseController

## src/quadcopter_tracking/controllers/deep_tracking_policy.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 12.93 KB  
**LOC:** 310  
**TODOs/FIXMEs:** 0  
**Declarations:** 3  
**Top-level declarations:**
  - class PolicyNetwork
  - class DeepTrackingPolicy
  - function create_controller_from_config
**External Dependencies:**
  - **Stdlib:** `json`, `logging`, `pathlib.Path`, `typing.Literal`
  - **Third-party:** `numpy`, `torch`, `torch.nn`

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
**Size:** 8.89 KB  
**LOC:** 177  
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
**Size:** 21.23 KB  
**LOC:** 481  
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
**Size:** 12.38 KB  
**LOC:** 336  
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

## src/quadcopter_tracking/train.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 25.55 KB  
**LOC:** 562  
**TODOs/FIXMEs:** 0  
**Declarations:** 5  
**Top-level declarations:**
  - class TrainingConfig
  - class Trainer
  - function load_checkpoint_and_resume
  - function parse_args
  - function main
**External Dependencies:**
  - **Stdlib:** `argparse`, `csv`, `datetime`, `json`, `logging`
    _(and 3 more)_
  - **Third-party:** `numpy`, `torch`, `torch.optim`, `yaml`

## src/quadcopter_tracking/utils/__init__.py
**Language:** Python  
**Role:** module-init  
**Role Justification:** module initialization file '__init__'  
**Size:** 11.80 KB  
**LOC:** 322  
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

## src/quadcopter_tracking/utils/losses.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 16.16 KB  
**LOC:** 395  
**TODOs/FIXMEs:** 0  
**Declarations:** 5  
**Top-level declarations:**
  - class TrackingLoss
  - class RewardShapingLoss
  - class CombinedLoss
  - function create_loss_from_config
  - class LossLogger
**External Dependencies:**
  - **Stdlib:** `logging`, `typing.Literal`
  - **Third-party:** `numpy`, `torch`, `torch.nn`

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

## tests/test_training_loop.py
**Language:** Python  
**Role:** test  
**Role Justification:** filename starts with 'test_'  
**Size:** 17.87 KB  
**LOC:** 417  
**TODOs/FIXMEs:** 0  
**Declarations:** 8  
**Top-level declarations:**
  - class TestPolicyNetwork
  - class TestDeepTrackingPolicy
  - class TestTrackingLoss
  - class TestCombinedLoss
  - class TestLossLogger
  - class TestTrainingConfig
  - class TestTrainer
  - class TestIntegration
**External Dependencies:**
  - **Stdlib:** `pathlib.Path`, `tempfile`
  - **Third-party:** `numpy`, `pytest`, `torch`
