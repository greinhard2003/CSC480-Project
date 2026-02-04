"""
Quick test to compare trained model vs random actions
"""
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
import numpy as np

from game import FrameSkipWrapper, CustomRewardWrapper

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

def make_mario_env():
    env = gym.make("SuperMarioBros-v0", render_mode="rgb_array", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = FrameSkipWrapper(env, skip=4)
    env = CustomRewardWrapper(env)
    return env

def safe_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def safe_step(env, action):
    out = env.step(action)
    if len(out) == 4:
        return out
    elif len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, terminated or truncated, info
    return out

def test_policy(env, model=None, num_steps=500):
    """Test a policy and return average x_pos"""
    obs = safe_reset(env)
    total_reward = 0
    max_x = 0

    for _ in range(num_steps):
        if model is not None:
            # Trained model
            if len(obs.shape) == 3 and obs.shape[2] == 3:
                obs_model = obs.transpose(2, 0, 1)
            else:
                obs_model = obs
            obs_model = np.ascontiguousarray(obs_model)
            action, _ = model.predict(obs_model, deterministic=True)
            action = int(action)
        else:
            # Random
            action = env.action_space.sample()

        obs, reward, done, info = safe_step(env, action)
        total_reward += reward
        max_x = max(max_x, info.get('x_pos', 0))

        if done:
            break

    return total_reward, max_x

if __name__ == "__main__":
    print("="*60)
    print("Comparing Random vs Trained Policy")
    print("="*60)

    # Test random policy
    print("\n1. Testing RANDOM policy...")
    env = make_mario_env()
    random_rewards = []
    random_x_pos = []

    for i in range(5):
        reward, x_pos = test_policy(env, model=None, num_steps=500)
        random_rewards.append(reward)
        random_x_pos.append(x_pos)
        print(f"   Run {i+1}: Reward={reward:.1f}, Max X={x_pos}")

    print(f"\n   Average: Reward={np.mean(random_rewards):.1f}, Max X={np.mean(random_x_pos):.1f}")

    # Test trained model
    print("\n2. Testing TRAINED model...")
    try:
        model = PPO.load("./models/mario_ppo_50000_steps.zip")
        trained_rewards = []
        trained_x_pos = []

        for i in range(5):
            reward, x_pos = test_policy(env, model=model, num_steps=500)
            trained_rewards.append(reward)
            trained_x_pos.append(x_pos)
            print(f"   Run {i+1}: Reward={reward:.1f}, Max X={x_pos}")

        print(f"\n   Average: Reward={np.mean(trained_rewards):.1f}, Max X={np.mean(trained_x_pos):.1f}")

        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        print(f"Random:  Reward={np.mean(random_rewards):.1f}, X={np.mean(random_x_pos):.1f}")
        print(f"Trained: Reward={np.mean(trained_rewards):.1f}, X={np.mean(trained_x_pos):.1f}")

        if np.mean(trained_x_pos) > np.mean(random_x_pos):
            print("\n✓ Model is LEARNING (better than random)")
        else:
            print("\n✗ Model is NOT better than random (needs more training)")

    except FileNotFoundError:
        print("   ERROR: Model not found!")

    env.close()
