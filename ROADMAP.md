# Roadmap and Future Designs

This document contains detailed design proposals and pseudocode for planned future features. These are **not yet implemented** in the codebase.

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

## Implementation Priority

| Feature | Priority | Complexity | Dependencies |
|---------|----------|------------|--------------|
| Observation Noise | High | Low | Environment modification |
| State Estimation (Kalman) | High | Medium | numpy/scipy |
| Recurrent Policies (LSTM/GRU) | Medium | Medium | PyTorch |
| Transfer Learning | Medium | Low | Trainer modification |
| Uncertainty-aware Policies | Low | High | Research |
