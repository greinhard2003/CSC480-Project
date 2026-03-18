# Super Mario Bros Reinforcement Learning

## Project Information

**Course**: CSC 480 - Artificial Intelligence, Cal Poly

**Instructor**: Rodrigo Canaan

**Team Members**:
- Jared Hammett
- Garrett Reinhard
- Peter Chinh
- Alyssa Gerardo
- Liza Znamerovskaya
- Matthew Stavros

## Acknowledgments

This project uses the following external libraries and resources:

- **[gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)** - OpenAI Gym environment for Super Mario Bros
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** - High-quality implementations of reinforcement learning algorithms
- **JoypadSpace compatibility fix** - Code snippet from [StackOverflow (aaron, 2023)](https://stackoverflow.com/questions/76509663/typeerror-joypadspace-reset-got-an-unexpected-keyword-argument-seed-when-i), License: CC BY-SA 4.0

## Project Overview

Train an AI agent to play Super Mario Bros using PPO (Proximal Policy Optimization) with customizable reward functions. The project includes two distinct reward strategies: one optimized for speed completion and another for coin collection.

## Prerequisites

**Python version:** <= 3.11.9

**Required packages:**
```bash
pip install gym-super-mario-bros
pip install stable-baselines3
pip install stable-baselines3[extra]
pip install Shimmy
pip install opencv-python
```

**Important:** gym-super-mario-bros may require Visual Studio Build Tools for C++

**NumPy version:** < 2.0
```bash
# The above installs may update numpy, so run this after:
pip install --force-reinstall "numpy<2.0"
```
## Branch Structure

- `main` - Coin-Based model
- `trained_model_v1` - Right-Only base model that completes 1-1
- `trained_model_world_1` - World 1 Completion Model
- `trained_model_score` - Score-Based model (completes World 1)
- `trained_model_world_2` - World 1+2 Model (completes World 1, 2-3, 2-4)

## Project Structure

- `game.py` - Environment testing script (runs random actions, NO training)
- `train.py` - **Train the RL agent** (saves models to `./models/`)
- `test_model.py` - Watch a trained agent play
- `models/` - Saved model checkpoints (created during training)
- `logs/` - TensorBoard logs (created during training)

## Usage

### 1. Test the Environment Setup
```bash
python game.py
```
This runs random actions to verify the environment works. Press ESC to quit.

### 2. Train the Agent

#### Basic Training (Default - Coins Mode)
```bash
python train.py
```
This will:
- Train for 15,000,000 timesteps (configurable via `TOTAL_TIMESTEPS` in train.py)
- Save checkpoints every 500,000 steps to `./models/` (configurable via `SAVE_FREQ` in train.py)
- Save final model as `mario_ppo_final.zip`
- Use the "coins" reward function (prioritizes coin collection)
- Can take several hours depending on your hardware

#### Training with Different Reward Functions

The project includes two reward strategies that can be configured in `train.py` on line 54:

**Option 1: Coins Mode (Default)** - Optimizes for collecting coins
```python
# In train.py, line 54:
env_fns = [make_mario_env(frame_skip=FRAME_SKIP, use_custom_reward=USE_CUSTOM_REWARD, reward_mode="coins") for _ in range(NUM_ENV)]
```
- Rewards: +5.0 per coin collected, +100.0 for level completion
- Penalties: -5.0 for death
- Small exploration bonus: +0.005 per unit of forward progress

**Option 2: Speed Mode** - Optimizes for fast level completion
```python
# In train.py, line 54:
env_fns = [make_mario_env(frame_skip=FRAME_SKIP, use_custom_reward=USE_CUSTOM_REWARD, reward_mode="speed") for _ in range(NUM_ENV)]
```
- Rewards: +0.025 per unit of forward progress, +0.05 bonus for reaching new maximum position, +100.0 for level completion
- Penalties: -10.0 for death, -0.01 per second elapsed
- Encourages fast completion with minimal time spent

#### Additional Training Parameters

You can customize training by modifying these variables in `train.py`:

- `NUM_ENV` (line 20): Number of parallel environments (default: cpu_count - 2)
- `TOTAL_TIMESTEPS` (line 21): Total training steps (default: 15,000,000)
- `SAVE_FREQ` (line 22): Model checkpoint frequency (default: 500,000)
- `FRAME_SKIP` (line 27): Action repeat duration (default: 4, higher = longer jumps)
- `USE_CUSTOM_REWARD` (line 28): Enable/disable custom rewards (default: True)

#### Monitor Training Progress
```bash
tensorboard --logdir ./logs
```
Then open http://localhost:6006 in your browser to view:
- Episode reward over time
- Episode length
- Loss curves
- Other training metrics

### 3. Test the Trained Agent
```bash
python test_model.py
```
Or specify a specific checkpoint:
```bash
python test_model.py ./models/mario_ppo_500000_steps.zip
```

## Reproducing Experimental Results

### Comparing Reward Functions

To reproduce the main experiments comparing "coins" vs "speed" reward functions:

1. **Train agent with coins reward function**:
   ```bash
   # Edit train.py line 54 to use reward_mode="coins"
   python train.py
   # Models will be saved to ./models/mario_ppo_*.zip
   ```

2. **Train agent with speed reward function**:
   ```bash
   # Edit train.py line 54 to use reward_mode="speed"
   python train.py
   # Models will be saved to ./models/ (rename or move previous models first)
   ```

3. **Evaluate trained agents**:
   ```bash
   # Test coins-trained agent
   python test_model.py ./models/mario_ppo_final.zip

   # Test speed-trained agent
   python test_model.py ./models/mario_ppo_final_speed.zip
   ```

4. **Compare performance metrics**:
   - View TensorBoard logs for both training runs
   - Compare: total episode reward, level completion rate, time to completion, coins collected
   - Logs are saved in `./logs/` directory with timestamps

### Expected Training Time

**Note**: Full training is computationally expensive:
- 15,000,000 timesteps takes approximately 20 hours but will vary drastically depending on hardware
- GPU acceleration recommended for faster training (set `device="cuda"` in train.py line 75)
- Checkpoints are saved every 500,000 steps for intermediate evaluation

## Features

### Dual Reward Functions

The project implements two distinct reward strategies:

**Coins Mode** (see `game.py:121-130`):
- Primary objective: Maximize coin collection
- +5.0 reward per coin collected
- +100.0 reward for level completion
- -5.0 penalty for death
- Small forward progress bonus (+0.005) to encourage exploration

**Speed Mode** (see `game.py:104-118`):
- Primary objective: Complete level as fast as possible
- +0.025 reward per unit of forward progress
- +0.05 bonus for reaching new maximum x-position
- +100.0 reward for level completion
- -10.0 penalty for death (less severe than coins mode)
- -0.01 penalty per second elapsed

### Frame Skip Wrapper

Actions are held for 4 frames by default (see `game.py:17-51`), allowing Mario to:
- Jump higher by holding the jump button longer
- Move more consistently across frames
- Have better momentum control

Adjust frame skip in `train.py` line 27 or when calling `make_mario_env()`:
```python
env = make_mario_env(frame_skip=6)  # Higher = longer jumps/actions
```

## Hyperparameters

Default PPO settings configured in `train.py` (lines 70-86):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 2.5e-4 | Step size for policy gradient updates |
| `n_steps` | 2048 | Steps to collect before policy update |
| `batch_size` | 256 | Minibatch size for SGD updates |
| `n_epochs` | 10 | Number of epochs per policy update |
| `gamma` | 0.99 | Discount factor for future rewards |
| `gae_lambda` | 0.95 | Lambda parameter for GAE (Generalized Advantage Estimation) |
| `clip_range` | 0.2 | PPO clipping parameter |
| `ent_coef` | 0.01 | Entropy coefficient for exploration |
| `vf_coef` | 0.5 | Value function coefficient in loss |
| `max_grad_norm` | 0.5 | Maximum gradient norm for clipping |

These hyperparameters are optimized for stable training. Modifications can be made in `train.py` to experiment with different values.

### Key Implementation Details

- **Parallel Training**: Uses `SubprocVecEnv` to run multiple Mario environments in parallel, significantly speeding up data collection
- **Modular Design**: Reward functions are implemented as gym wrappers, making it easy to add new reward strategies
- **Robust Step Handling**: Code handles both gym and gymnasium API formats for compatibility
- **Checkpoint System**: Automatic model saving during training allows for recovery and intermediate evaluation

## Troubleshooting

**Issue**: "gym-super-mario-bros installation fails"
- **Solution**: Install Visual Studio Build Tools for C++ (required for compilation)

**Issue**: "NumPy version incompatibility"
- **Solution**: Ensure NumPy < 2.0 with `pip install --force-reinstall "numpy<2.0"`

**Issue**: "Training is slow"
- **Solution**: Increase `NUM_ENV` for more parallel environments, or enable GPU with `device="cuda"` in train.py

**Issue**: "Mario doesn't jump high enough"
- **Solution**: Increase `FRAME_SKIP` parameter to hold actions longer (try 6 or 8)

## License

This project is for educational purposes as part of CSC 480 at California Polytechnic University San Luis Obispo.
