import time
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
import numpy as np
import cv2

# Source - https://stackoverflow.com/questions/76509663/typeerror-joypadspace-reset-got-an-unexpected-keyword-argument-seed-when-i
# Posted by aaron
# Retrieved 2026-01-27, License - CC BY-SA 4.0
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)


class FrameSkipWrapper(gym.Wrapper):
    """
    Repeat each action for N frames. This allows Mario to hold jump longer
    and reach maximum jump height.
    """
    def __init__(self, env, skip=4):
        super(FrameSkipWrapper, self).__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.skip):
            step_result = self.env.step(action)

            if len(step_result) == 4:
                obs, reward, done, info = step_result
                terminated = done
                truncated = False
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected step result format with {len(step_result)} values")

            total_reward += reward

            if done:
                break

        return obs, total_reward, terminated, truncated, info


class CustomRewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper for Super Mario Bros.
    Modifies the reward function to encourage desired behaviors.
    """
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.prev_x_pos = 0
        self.prev_score = 0
        self.prev_coins = 0
        self.prev_time = 400

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Reset tracking variables
        self.prev_x_pos = 0
        self.prev_score = 0
        self.prev_coins = 0
        self.prev_time = 400
        return obs

    def step(self, action):
        # Handle both gym (4-tuple) and gymnasium (5-tuple) formats
        step_result = self.env.step(action)

        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            raise ValueError(f"Unexpected step result format with {len(step_result)} values")

        # Extract info from the environment
        x_pos = info.get('x_pos', 0)
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        time_left = info.get('time', 400)
        status = info.get('status', 'small')

        # Custom reward calculation
        custom_reward = 0

        # 1. Reward forward progress (most important)
        x_progress = x_pos - self.prev_x_pos
        custom_reward += x_progress * 0.1  # Small reward for moving right

        # 2. Penalize backward movement
        if x_progress < 0:
            custom_reward += x_progress * 0.2  # Stronger penalty for going left

        # 3. Reward score increase (coins, enemies defeated, etc.)
        score_increase = score - self.prev_score
        custom_reward += score_increase * 0.01

        # 4. Reward coin collection
        coin_increase = coins - self.prev_coins
        custom_reward += coin_increase * 1.0

        # 5. Penalize death heavily
        if done and x_pos < 3161:  # 3161 is roughly the flag position
            custom_reward -= 50

        # 6. Reward reaching the flag
        if done and x_pos >= 3161:
            custom_reward += 100

        # 7. Small time penalty to encourage speed
        time_penalty = self.prev_time - time_left
        if time_penalty > 1:  # Normal time passage
            custom_reward -= 0.01

        # Update previous values
        self.prev_x_pos = x_pos
        self.prev_score = score
        self.prev_coins = coins
        self.prev_time = time_left

        # Always return gymnasium format (5-tuple) for compatibility with Stable-Baselines3
        return obs, custom_reward, terminated, truncated, info

def make_mario_env(render_mode="rgb_array", use_custom_reward=True, frame_skip=4):
    def _init():
        env = gym.make(
            "SuperMarioBros-v0",
            render_mode=render_mode,
            apply_api_compatibility=True,
        )
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        # Apply frame skip wrapper to hold actions for multiple frames
        # This allows Mario to jump higher by holding the jump button
        if frame_skip > 1:
            env = FrameSkipWrapper(env, skip=frame_skip)

        # Apply custom reward wrapper if enabled
        if use_custom_reward:
            env = CustomRewardWrapper(env)

        return env
    return _init


# Helper to stack frames in grid for rendering
def stack_frames_grid(obs, rows, cols):
    """
    Stack multiple frames (C,H,W) into a grid (H_total, W_total, 3) for display.
    """
    # Convert channel-first to HWC
    frames = [np.transpose(f, (1, 2, 0)) for f in obs]  # (H,W,C)
    # RGB -> BGR for OpenCV
    frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]

    # Stack into rows
    h, w, c = frames[0].shape
    grid_rows = []
    for r in range(rows):
        row_frames = frames[r*cols:(r+1)*cols]
        # Pad missing frames with black images
        if len(row_frames) < cols:
            row_frames += [np.zeros_like(frames[0])]*(cols - len(row_frames))
        grid_rows.append(np.hstack(row_frames))
    return np.vstack(grid_rows)

if __name__ == "__main__":
    """
    ============================================================
    NOTE: This script is for TESTING the environment setup only.
    It runs random actions and does NOT train an RL agent.

    To actually TRAIN an agent, use: python train.py
    To TEST a trained agent, use: python test_model.py
    ============================================================
    """
    # Print the mapping so you know which index equals which button combo
    print("SIMPLE_MOVEMENT mapping (index -> button tuple):")
    for i, action in enumerate(SIMPLE_MOVEMENT):
        print(f"{i}: {action}")
    print("0 is usually NOOP. Try 1 or 2 for moving right (depends on the mapping above).")

    # Helper to handle both gym and gymnasium style resets/steps
    def safe_reset(e):
        out = e.reset()
        # gym returns obs; gymnasium returns (obs, info)
        if isinstance(out, tuple) and len(out) >= 1:
            return out[0]
        return out

    def safe_step(e, action):
        out = e.step(action)
        # gym: (obs, reward, done, info)
        # gymnasium: (obs, reward, terminated, truncated, info)
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, done, info
        elif len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
            return obs, reward, done, info
        else:
            # unexpected format — just return what we get
            return out

    # Main loop with FPS limiter
    FPS = 60.0
    frame_dt = 1.0 / FPS

    vectorize = True
    NUM_ENV = 4 # NUMBER OF ENVIRONMENTS TO VECTORIZE

    # Choose a deterministic action index to try moving right.
    # If you printed the mapping above, set `ACTION_TO_TRY` to the index that corresponds to 'right' or 'right + A'.
    ACTION_TO_TRY = 1   # change this if your mapping shows a different index for 'right'

    if (vectorize):
        render = True # Flag to render all environments
        random = True # Flag to test if all environments are different

        env_fns = [make_mario_env() for _ in range (NUM_ENV)]
        vec_env = SubprocVecEnv(env_fns)
        vec_env = VecTransposeImage(vec_env)

        obs = vec_env.reset()
        done_flags = np.array([False]*NUM_ENV)

        try:
            while True:
                # Step all envs
                if (random):
                    actions = [vec_env.action_space.sample() for _ in range(NUM_ENV)]
                    obs, rewards, dones, infos = vec_env.step(actions)
                    for i, r in enumerate(rewards):
                        print(f"env {i}: reward = {r}")

                else:
                    actions = [ACTION_TO_TRY]*NUM_ENV
                    obs, reward, dones, infos = vec_env.step(actions)

                if (render):
                    # Stack frames into grid
                    stacked = stack_frames_grid(obs, rows=2, cols=2)
                    cv2.imshow("Multiple Mario Envs", stacked)

                    # Break on ESC
                    if cv2.waitKey(int(frame_dt*1000)) & 0xFF == 27:
                        break

        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            vec_env.close()
            cv2.destroyAllWindows()

    else:
        env_func = make_mario_env()
        env = env_func("human")
        obs = safe_reset(env)
        done = False

        try:
            t0 = time.time()
            for step in range(5000):
                if done:
                    obs = safe_reset(env)

                # use deterministic action instead of random sampling
                action = ACTION_TO_TRY
                obs, reward, done, info = safe_step(env, action)

                env.render()

                # accurate real-time limiting
                t1 = time.time()
                elapsed = t1 - t0
                to_sleep = frame_dt - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
                t0 = time.time()

        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            env.close()


