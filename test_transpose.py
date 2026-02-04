"""
Test both transpose directions to see which is correct
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

def test_with_transpose(env, model, transpose_mode, num_steps=300):
    """Test with different transpose modes"""
    obs = safe_reset(env)
    total_reward = 0
    max_x = 0
    actions_taken = []

    for _ in range(num_steps):
        # Apply transpose based on mode
        if transpose_mode == "none":
            obs_model = obs
        elif transpose_mode == "2,0,1":  # Current: HWC -> CHW
            obs_model = obs.transpose(2, 0, 1)
        elif transpose_mode == "0,1,2":  # No change
            obs_model = obs.transpose(0, 1, 2)
        elif transpose_mode == "1,0,2":  # Swap H and W
            obs_model = obs.transpose(1, 0, 2)
        else:
            obs_model = obs

        obs_model = np.ascontiguousarray(obs_model)

        try:
            action, _ = model.predict(obs_model, deterministic=True)
            action = int(action)
            actions_taken.append(action)
        except Exception as e:
            print(f"   Error with transpose {transpose_mode}: {e}")
            return 0, 0, []

        obs, reward, done, info = safe_step(env, action)
        total_reward += reward
        max_x = max(max_x, info.get('x_pos', 0))

        if done:
            break

    return total_reward, max_x, actions_taken

if __name__ == "__main__":
    print("="*60)
    print("Testing Different Transpose Modes")
    print("="*60)

    try:
        model = PPO.load("./models/mario_ppo_50000_steps.zip")
        env = make_mario_env()

        obs = safe_reset(env)
        print(f"\nOriginal observation shape: {obs.shape}")
        print("Expected: (240, 256, 3) = (Height, Width, Channels)")

        transpose_modes = [
            ("none", "No transpose (240, 256, 3)"),
            ("2,0,1", "Current: (2,0,1) -> (3, 240, 256)"),
            ("0,1,2", "Identity: (0,1,2) -> (240, 256, 3)"),
            ("1,0,2", "Swap H/W: (1,0,2) -> (256, 240, 3)"),
        ]

        print("\n" + "="*60)
        print("Testing each transpose mode:")
        print("="*60)

        results = []
        for mode, description in transpose_modes:
            print(f"\nTesting: {description}")
            reward, x_pos, actions = test_with_transpose(env, model, mode, num_steps=300)

            # Count action distribution
            action_names = ['NOOP', 'right', 'right+A', 'right+B', 'right+A+B', 'A', 'left']
            action_counts = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1

            most_common = max(action_counts.items(), key=lambda x: x[1]) if action_counts else (0, 0)
            most_common_name = action_names[most_common[0]] if most_common[0] < len(action_names) else "?"

            print(f"  Reward: {reward:.1f}, Max X: {x_pos}, Most common action: {most_common_name} ({most_common[1]} times)")
            results.append((mode, description, reward, x_pos, most_common_name, most_common[1]))

        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        for mode, desc, reward, x_pos, action, count in results:
            print(f"{desc:40} -> X={x_pos:4}, R={reward:6.1f}, Action={action}")

        best = max(results, key=lambda x: x[3])  # Max x_pos
        print(f"\n✓ BEST: {best[1]} (X={best[3]})")

        env.close()

    except FileNotFoundError:
        print("ERROR: Model not found!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
