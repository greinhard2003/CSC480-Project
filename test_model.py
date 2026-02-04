"""
Test a trained Super Mario Bros model and watch it play
"""
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
import numpy as np
import cv2
import time

# Import the wrappers from game.py
from game import FrameSkipWrapper, CustomRewardWrapper

# Monkey patch for JoypadSpace compatibility
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)


def make_mario_env(render_mode="rgb_array", use_custom_reward=True, frame_skip=4):
    """Create a Super Mario Bros environment with wrappers"""
    env = gym.make(
        "SuperMarioBros-v0",
        render_mode=render_mode,
        apply_api_compatibility=True,
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Apply frame skip wrapper to hold actions for multiple frames
    if frame_skip > 1:
        env = FrameSkipWrapper(env, skip=frame_skip)

    # Apply custom reward wrapper if enabled
    if use_custom_reward:
        env = CustomRewardWrapper(env)

    return env


def safe_reset(env):
    """Handle both gym and gymnasium reset formats"""
    out = env.reset()
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out


def safe_step(env, action):
    """Handle both gym and gymnasium step formats"""
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, info
    elif len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    else:
        return out


if __name__ == "__main__":
    import sys

    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "./models/mario_ppo_50000_steps.zip"

    print("="*60)
    print("Super Mario Bros - Trained Model Testing")
    print("="*60)
    print(f"Loading model from: {model_path}")

    try:
        # Load the trained model
        model = PPO.load(model_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"\nERROR: Model not found at {model_path}")
        print("\nAvailable models:")
        import os
        if os.path.exists("./models"):
            models = [f for f in os.listdir("./models") if f.endswith(".zip")]
            for m in models:
                print(f"  - ./models/{m}")
        else:
            print("  No models directory found. Train a model first with: python train.py")
        sys.exit(1)

    # Create environment (use "human" render mode for direct rendering)
    print("Creating environment...")
    # Note: We'll use rgb_array mode and manually render with cv2
    env = make_mario_env(render_mode="rgb_array", frame_skip=4)

    print("\nStarting evaluation...")
    print("Press ESC to quit\n")
    print("="*60)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("="*60)

    FPS = 60.0
    frame_dt = 1.0 / FPS

    episode = 0
    total_episodes = 5  # Number of episodes to run

    try:
        while episode < total_episodes:
            obs = safe_reset(env)
            done = False
            episode_reward = 0
            step_count = 0
            info = {}  # Initialize info dict

            episode += 1
            print(f"\nEpisode {episode}/{total_episodes}")

            # Debug: print observation shape on first episode
            if episode == 1:
                print(f"Debug - Raw obs shape: {obs.shape}")

            # Track action distribution
            action_counts = {}

            while not done:
                # Transpose observation from HWC to CHW to match training format
                # (VecTransposeImage does this during training)
                if len(obs.shape) == 3 and obs.shape[2] == 3:  # HWC format
                    obs_for_model = obs.transpose(2, 0, 1)  # Convert to CHW
                else:
                    obs_for_model = obs

                # Debug: print transposed shape on first step
                if episode == 1 and step_count == 0:
                    print(f"Debug - Transposed obs shape for model: {obs_for_model.shape}")

                # Ensure observation is contiguous in memory
                obs_for_model = np.ascontiguousarray(obs_for_model)

                # Predict action using the trained model
                action, _states = model.predict(obs_for_model, deterministic=True)

                # Convert action from numpy array to int
                action = int(action)

                # Track action frequency
                action_counts[action] = action_counts.get(action, 0) + 1

                # Step the environment
                obs, reward, done, info = safe_step(env, action)
                episode_reward += reward
                step_count += 1

                # Use observation as the frame (it's already in HWC format: 240x256x3)
                frame = obs.copy()

                # Ensure frame is uint8 (should already be from observation space)
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)

                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Scale up for better visibility
                frame = cv2.resize(frame, (512, 480), interpolation=cv2.INTER_NEAREST)

                # Add info overlay
                x_pos = info.get('x_pos', 0)
                cv2.putText(frame, f"Episode: {episode}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Reward: {episode_reward:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"X Position: {x_pos}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Steps: {step_count}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Trained Mario Agent", frame)

                # FPS limiting
                if cv2.waitKey(int(frame_dt * 1000)) & 0xFF == 27:
                    print("\n\nStopped by user (ESC pressed)")
                    done = True
                    episode = total_episodes  # Exit outer loop too

            print(f"Episode finished! Total reward: {episode_reward:.1f}, Steps: {step_count}, Final X: {info.get('x_pos', 0)}")

            # Print action distribution
            print("Action distribution:")
            action_names = ['NOOP', 'right', 'right+A', 'right+B', 'right+A+B', 'A', 'left']
            for action_id in sorted(action_counts.keys()):
                count = action_counts[action_id]
                percentage = (count / step_count) * 100
                action_name = action_names[action_id] if action_id < len(action_names) else f"unknown_{action_id}"
                print(f"  {action_name}: {count} times ({percentage:.1f}%)")

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")

    finally:
        env.close()
        cv2.destroyAllWindows()
        print("\nEnvironment closed.")
        print("="*60)
