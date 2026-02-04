# Super Mario Bros Reinforcement Learning

Train an AI agent to play Super Mario Bros using PPO (Proximal Policy Optimization).

## Prerequisites

**Python version:** <= 3.11.9

**Required packages:**
```bash
pip install gym-super-mario-bros
pip install stable-baselines3
pip install Shimmy
pip install opencv-python
```

**Important:** gym-super-mario-bros may require Visual Studio Build Tools for C++

**NumPy version:** < 2.0
```bash
# The above installs may update numpy, so run this after:
pip install --force-reinstall "numpy<2.0"
```

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
```bash
python train.py
```
This will:
- Train for 1,000,000 timesteps (adjust `TOTAL_TIMESTEPS` in the script)
- Save checkpoints every 50,000 steps to `./models/`
- Save final model as `mario_ppo_final.zip`
- Can take several hours depending on your hardware

**Monitor training progress:**
```bash
tensorboard --logdir ./logs
```
Then open http://localhost:6006 in your browser

### 3. Test the Trained Agent
```bash
python test_model.py
```
Or specify a specific checkpoint:
```bash
python test_model.py ./models/mario_ppo_500000_steps.zip
```

## Features

### Custom Reward Function
The agent is trained with a custom reward function that:
- Rewards forward progress (+0.1 per unit)
- Penalizes backward movement (-0.2 per unit)
- Rewards score increases (+0.01 per point)
- Rewards coin collection (+1.0 per coin)
- Heavily penalizes death (-50)
- Big reward for level completion (+100)
- Small time penalty to encourage speed (-0.01 per second)

### Frame Skip
Actions are held for 4 frames by default, allowing Mario to:
- Jump higher (hold jump button longer)
- Move more consistently
- Have better momentum control

Adjust in `train.py` or `test_model.py`:
```python
env = make_mario_env(frame_skip=6)  # Higher = longer jumps
```

## Hyperparameters

Default PPO settings in `train.py`:
- Learning rate: 3e-4
- Batch size: 64
- Number of epochs: 10
- Discount factor (gamma): 0.99
- Entropy coefficient: 0.01

Tune these for better performance!
