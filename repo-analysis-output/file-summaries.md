# File Summaries

Heuristic summaries of source files based on filenames, extensions, and paths.

Schema Version: 2.0

Total files: 21

## scripts/generate_comparison_report.py
**Language:** Python  
**Role:** script  
**Role Justification:** located in 'scripts' directory  
**Size:** 3.10 KB  
**LOC:** 88  
**TODOs/FIXMEs:** 0  
**Declarations:** 1  
**Top-level declarations:**
  - function main
**External Dependencies:**
  - **Stdlib:** `argparse`, `json`, `pathlib.Path`, `sys`

## src/quadcopter_tracking/__init__.py
**Language:** Python  
**Role:** module-init  
**Role Justification:** module initialization file '__init__'  
**Size:** 1.15 KB  
**LOC:** 44  
**TODOs/FIXMEs:** 0  

## src/quadcopter_tracking/controllers/__init__.py
**Language:** Python  
**Role:** module-init  
**Role Justification:** module initialization file '__init__'  
**Size:** 25.86 KB  
**LOC:** 438  
**TODOs/FIXMEs:** 0  
**Declarations:** 4  
**Top-level declarations:**
  - function _validate_observation
  - function _ensure_array
  - class PIDController
  - class LQRController
**External Dependencies:**
  - **Third-party:** `numpy`

## src/quadcopter_tracking/controllers/base.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 4.23 KB  
**LOC:** 103  
**TODOs/FIXMEs:** 0  
**Declarations:** 3  
**Top-level declarations:**
  - class ActionLimits
  - function validate_action
  - class BaseController
**External Dependencies:**
  - **Stdlib:** `dataclasses.dataclass`
  - **Third-party:** `numpy`

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

## src/quadcopter_tracking/controllers/riccati_lqr.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 19.38 KB  
**LOC:** 407  
**TODOs/FIXMEs:** 0  
**Declarations:** 6  
**Top-level declarations:**
  - function _is_positive_semidefinite
  - function _is_positive_definite
  - function solve_dare
  - function build_linearized_system
  - function _validate_observation
  - class RiccatiLQRController
**External Dependencies:**
  - **Stdlib:** `logging`
  - **Third-party:** `numpy`, `scipy.linalg.solve_discrete_are`

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
**Size:** 8.90 KB  
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

## src/quadcopter_tracking/eval.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 25.46 KB  
**LOC:** 650  
**TODOs/FIXMEs:** 0  
**Declarations:** 6  
**Top-level declarations:**
  - class Evaluator
  - function load_controller
  - function run_hyperparameter_sweep
  - function parse_args
  - function _load_eval_config
  - function main
**External Dependencies:**
  - **Stdlib:** `argparse`, `collections.abc.Callable`, `datetime.datetime`, `datetime.timezone`, `json`
    _(and 3 more)_
  - **Third-party:** `matplotlib`, `matplotlib.pyplot`, `numpy`, `yaml`

## src/quadcopter_tracking/train.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 45.26 KB  
**LOC:** 938  
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
  - **Third-party:** `numpy`, `torch`, `torch.nn.functional`, `torch.optim`, `yaml`

## src/quadcopter_tracking/utils/__init__.py
**Language:** Python  
**Role:** module-init  
**Role Justification:** module initialization file '__init__'  
**Size:** 12.80 KB  
**LOC:** 362  
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

## src/quadcopter_tracking/utils/diagnostics.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 27.97 KB  
**LOC:** 692  
**TODOs/FIXMEs:** 0  
**Declarations:** 9  
**Top-level declarations:**
  - class DiagnosticsConfig
  - class GradientStats
  - class StepDiagnostics
  - class EpochDiagnostics
  - function _safe_float
  - function compute_gradient_stats
  - function compute_observation_stats
  - function compute_action_stats
  - class Diagnostics
**External Dependencies:**
  - **Stdlib:** `csv`, `dataclasses.dataclass`, `dataclasses.field`, `json`, `logging`
    _(and 4 more)_
  - **Third-party:** `matplotlib`, `matplotlib.pyplot`, `numpy`, `torch`

## src/quadcopter_tracking/utils/losses.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 16.11 KB  
**LOC:** 392  
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

## src/quadcopter_tracking/utils/metrics.py
**Language:** Python  
**Role:** implementation  
**Role Justification:** general implementation file (default classification)  
**Size:** 14.77 KB  
**LOC:** 352  
**TODOs/FIXMEs:** 0  
**Declarations:** 10  
**Top-level declarations:**
  - class SuccessCriteria
  - class EpisodeMetrics
  - class EvaluationSummary
  - function compute_tracking_error
  - function compute_on_target_ratio
  - function compute_control_effort
  - function detect_overshoots
  - function compute_episode_metrics
  - function compute_evaluation_summary
  - function format_metrics_report
**External Dependencies:**
  - **Stdlib:** `dataclasses.dataclass`, `dataclasses.field`, `logging`
  - **Third-party:** `numpy`

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
**Size:** 113.91 KB  
**LOC:** 2326  
**TODOs/FIXMEs:** 0  
**Declarations:** 16  
**Top-level declarations:**
  - class TestTargetMotion
  - class TestQuadcopterEnv
  - class TestEnvConfig
  - class TestIntegration
  - class TestPIDController
  - class TestLQRController
  - class TestClassicalControllerIntegration
  - function create_hover_observation
  - function create_hover_env_config
  - class TestHoverThrustIntegration
  - ... and 6 more
**External Dependencies:**
  - **Third-party:** `numpy`, `pytest`

## tests/test_eval.py
**Language:** Python  
**Role:** test  
**Role Justification:** filename starts with 'test_'  
**Size:** 25.39 KB  
**LOC:** 563  
**TODOs/FIXMEs:** 0  
**Declarations:** 10  
**Top-level declarations:**
  - class TestMetrics
  - class TestEpisodeMetrics
  - class TestEvaluationSummary
  - class TestSuccessCriteria
  - class TestEvaluator
  - class TestLoadController
  - class TestIntegration
  - class TestControllerSelectionEval
  - class TestActionSchema
  - class TestControllerConfigPropagation
**External Dependencies:**
  - **Third-party:** `matplotlib.pyplot`, `numpy`, `pytest`

## tests/test_training_loop.py
**Language:** Python  
**Role:** test  
**Role Justification:** filename starts with 'test_'  
**Size:** 40.29 KB  
**LOC:** 889  
**TODOs/FIXMEs:** 0  
**Declarations:** 11  
**Top-level declarations:**
  - class TestPolicyNetwork
  - class TestDeepTrackingPolicy
  - class TestTrackingLoss
  - class TestCombinedLoss
  - class TestLossLogger
  - class TestTrainingConfig
  - class TestTrainer
  - class TestTrainingModes
  - class TestControllerSelection
  - class TestIntegration
  - ... and 1 more
**External Dependencies:**
  - **Stdlib:** `pathlib.Path`, `tempfile`
  - **Third-party:** `numpy`, `pytest`, `torch`
