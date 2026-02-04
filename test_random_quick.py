"""Quick test to see if random actions also go left"""
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from game import FrameSkipWrapper, CustomRewardWrapper

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = gym.make("SuperMarioBros-v0", render_mode="rgb_array", apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = FrameSkipWrapper(env, skip=4)
env = CustomRewardWrapper(env)

obs = env.reset()
obs = obs[0] if isinstance(obs, tuple) else obs

action_counts = {}
total_reward = 0
max_x = 0

print("Testing 500 RANDOM actions...")
for i in range(500):
    action = env.action_space.sample()
    action_counts[action] = action_counts.get(action, 0) + 1

    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result

    total_reward += reward
    max_x = max(max_x, info.get('x_pos', 0))

    if done:
        break

print(f"\nTotal reward: {total_reward:.1f}")
print(f"Max X position: {max_x}")
print("\nAction distribution:")
action_names = ['NOOP', 'right', 'right+A', 'right+B', 'right+A+B', 'A', 'left']
for action_id in sorted(action_counts.keys()):
    count = action_counts[action_id]
    name = action_names[action_id]
    print(f"  {name}: {count} times")

env.close()
