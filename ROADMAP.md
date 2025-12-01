# Roadmap and Future Designs

This document contains detailed design proposals and pseudocode for planned future features. These are **not yet implemented** in the codebase.

## Completed in v0.3

The following features from the original roadmap have been implemented:

| Feature | Status | Notes |
|---------|--------|-------|
| Riccati-LQR Controller | ✅ Complete | DARE-based optimal feedback gains |
| Controller Auto-Tuning | ✅ Complete | Grid/random search for PID/LQR/Riccati |
| Feedforward Support | ✅ Complete | Optional velocity/acceleration feedforward |
| Documentation Overhaul | ✅ Complete | Consolidated workflows and config docs |
| Release Validation Workflow | ✅ Complete | Baseline regeneration guidance |

**Milestone Summary**: The v0.3 release delivers mathematically optimal control via the Riccati-LQR controller, automated gain tuning, and comprehensive documentation for reproducible research workflows. All v0.2.x configurations remain compatible.

## Completed in v0.2

The following features from the original roadmap have been implemented:

| Feature | Status | Notes |
|---------|--------|-------|
| PID/LQR Classical Controllers | ✅ Complete | Full evaluation support |
| Training Diagnostics | ✅ Complete | Step/epoch logging, gradient tracking |
| Imitation Learning Mode | ✅ Complete | Supervisor-based training |
| Reproducible Workflows | ✅ Complete | Three documented workflows |
| Controller Comparison | ✅ Complete | Automated comparison reports |

## Next Focus Areas (v0.4+)

The following features are prioritized for the next major release:

| Feature | Priority | Complexity | Notes |
|---------|----------|------------|-------|
| Observation Noise | High | Low | Add configurable sensor noise to observations |
| State Estimation (Kalman) | High | Medium | Requires observation noise first |
| Recurrent Policies (LSTM/GRU) | Medium | Medium | For partial observability |
| Transfer Learning | Medium | Low | Trainer checkpoint loading enhancements |

## Deferred to Future Releases

The following features are designed but not yet implemented:

| Feature | Target | Reason for Deferral |
|---------|--------|---------------------|
| Observation Noise | v0.4+ | Requires environment modification |
| State Estimation (Kalman) | v0.4+ | Depends on observation noise |
| Recurrent Policies (LSTM/GRU) | v0.4+ | Research complexity |
| Transfer Learning | v0.4+ | Trainer modification needed |
| Reinforcement Learning (PPO/SAC) | v0.5+ | Significant implementation effort |

## Imperfect Information Controllers

*Design proposal for handling partial observability.*

For controllers that must handle partial observability:

1. **State estimation**: Implement filtering (Kalman, particle filter)
2. **Memory-based architectures**: Use LSTM/GRU networks
3. **Uncertainty-aware policies**: Output distributions over actions

### Proposed Implementation Structure

```python
# Pseudocode - components to implement
import copy

class PartialObservabilityController(BaseController):
    """Example structure for handling partial observability."""
    def __init__(self, config=None):
        super().__init__(name="partial_obs", config=config)
        # Implement your own state estimator (e.g., Kalman filter)
        self.state_estimator = None  # KalmanFilter(...)
        # Implement a recurrent policy network
        self.policy = None  # RecurrentPolicy(...)
    
    def compute_action(self, observation):
        # Add simulated noise for training
        noisy_obs = self._add_observation_noise(observation)
        # Estimate true state using your filter
        estimated_state = self.state_estimator.update(noisy_obs)
        # Compute action from estimated state
        return self.policy(estimated_state)
    
    def _add_observation_noise(self, observation):
        # Add Gaussian noise to simulate sensor imperfections
        import numpy as np
        noisy = copy.deepcopy(observation)
        noisy["target"]["position"] += np.random.normal(0, 0.1, 3)
        return noisy
```

## Observation Noise Configuration

*Proposed configuration format for adding sensor noise to observations.*

```yaml
# Proposed future extension - requires environment modification
# This configuration format is a design reference, not currently functional
observation_noise:
  enabled: true
  position_stddev: 0.1  # meters
  velocity_stddev: 0.05  # m/s

# To implement: Modify QuadcopterEnv._get_observation() to add noise
# when observation_noise.enabled is true in config
```

## Transfer Learning Support

*Design proposal for enabling transfer learning in the Trainer class.*

### Planned Approach

To adapt trained controllers to new scenarios:

1. Load pretrained checkpoint
2. Freeze early layers (optional)
3. Fine-tune on new target motion patterns

### Proposed API

```python
# Future implementation concept (not currently available)
# This shows the desired API for transfer learning support

config = TrainingConfig()
config.target_motion_type = "figure8"
config.learning_rate = 0.0001  # Lower LR for fine-tuning
config.epochs = 50
config.pretrained_checkpoint = "checkpoints/pretrained.pt"  # Future feature
config.freeze_layers = [0, 1]  # Future feature: freeze early layers

trainer = Trainer(config)
trainer.train()
```

## Implementation Priority (v0.4+)

| Feature | Priority | Complexity | Dependencies |
|---------|----------|------------|--------------|
| Observation Noise | High | Low | Environment modification |
| State Estimation (Kalman) | High | Medium | numpy/scipy, observation noise |
| Recurrent Policies (LSTM/GRU) | Medium | Medium | PyTorch |
| Transfer Learning | Medium | Low | Trainer modification |
| Uncertainty-aware Policies | Low | High | Research |
| Reinforcement Learning (PPO/SAC) | Medium | High | Stable training pipeline |
