# RL Generalization Benchmark

This project implements a reinforcement learning benchmark for testing agent generalization across environment variations.

## Features

- **Environment Wrappers**: Support for both DM-Control Suite and OpenAI Gymnasium/Gym
- **Automatic Fallback**: Falls back to Gymnasium if DM-Control/MuJoCo fails (e.g., on Windows)
- **Domain Randomization**: Test generalization with environment variations (gravity, friction, etc.)
- **Multiple Algorithms**: Includes DDPG and baseline random agent
- **Visualization**: Automatic plotting of training curves and generalization results

## Installation

### Option 1: With DM-Control (Linux/Mac recommended)
```bash
pip install dm-control[mujoco] gymnasium matplotlib torch numpy
```

### Option 2: Gymnasium only (Windows compatible)
```bash
pip install gymnasium[classic_control] matplotlib torch numpy
```

## Quick Start

### 1. Simple Test
```python
python quick_test.py
```

### 2. Run Full Benchmark
```python
from src.bench import GeneralizationBenchmark
from src.rl import DDPG, RandomAgent

# Create benchmark
benchmark = GeneralizationBenchmark(
    domain_name='Pendulum-v1',  # Gymnasium environment
    env_type='gymnasium'        # Use 'dmcontrol' for DM-Control
)

# Define agents to compare
agent_classes = {
    'DDPG': DDPG,
    'Random': RandomAgent
}

# Run benchmark
results = benchmark.run_benchmark(agent_classes, 
                                num_train_episodes=100, 
                                num_eval_episodes=10)

# Plot results
benchmark.plot_results(results)
```

## Supported Environments

### Gymnasium (Recommended for Windows)
- `Pendulum-v1` - Continuous control pendulum
- `CartPole-v1` - Discrete control cart-pole  
- `Acrobot-v1` - Continuous control acrobot
- `MountainCarContinuous-v0` - Drive up mountain

### DM-Control Suite (Linux/Mac)
- `cartpole/balance` - Balance cartpole
- `cartpole/swingup` - Swing up cartpole
- `pendulum/swingup` - Swing up pendulum
- `acrobot/swingup` - Swing up acrobot

## Environment Variations

The benchmark tests generalization by training on the default environment and testing on:

- **Default**: Standard environment parameters
- **High Gravity**: Increased gravity (15.0 vs ~10.0)  
- **Low Friction**: Reduced friction/force effectiveness

## Troubleshooting

### DM-Control/MuJoCo Issues on Windows
The most common issue is MuJoCo DLL loading problems on Windows. The system automatically falls back to Gymnasium in this case.

To fix MuJoCo on Windows (optional):
1. Install Visual C++ Redistributable
2. Try: `pip install --upgrade mujoco dm-control[mujoco]`
3. Or just use Gymnasium fallback (recommended)

## Results Interpretation

The benchmark reports:
- **Training Performance**: How well agents learn during training
- **Default Score**: Performance on standard environment  
- **Generalization Score**: Average performance on varied environments
- **Generalization Gap**: Difference between default and varied performance

A smaller generalization gap indicates better generalization ability.

## File Structure

```
src/
├── env.py          # Environment wrappers and factory
├── rl.py           # RL algorithms (DDPG, RandomAgent)  
├── bench.py        # Benchmark framework
└── mbrl.py         # Model-based RL algorithms (future)

quick_test.py       # Quick functionality test
run_benchmark.py    # Full benchmark runner
```
