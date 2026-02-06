"""
Test a trained Super Mario Bros model and watch it play
"""
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
import numpy as np
import cv2
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Import the wrappers from game.py
from game import make_mario_env

# Monkey patch for JoypadSpace compatibility
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

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
        model_path = "./models/mario_ppo_final.zip"

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
    env_func = make_mario_env(render_mode="rgb_array", frame_skip=4)
    env = DummyVecEnv([env_func])
    env = VecFrameStack(env, n_stack=4)

    display_env = make_mario_env(render_mode="rgb_array", frame_skip=4, gray=False, resize=False)()

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

    # Track stats across all episodes
    all_action_counts = {}
    episode_stats = []

    try:
        while episode < total_episodes:
            obs = safe_reset(env)
            display_obs = safe_reset(display_env)
            done = False
            episode_reward = 0
            step_count = 0
            info = {}  # Initialize info dict

            episode += 1
            print(f"\nEpisode {episode}/{total_episodes}")

            # Debug: print observation shape on first episode
            if episode == 1:
                print(f"Debug - Raw obs shape: {obs.shape}")

            # Track action distribution for this episode
            action_counts = {}

            while not done:
                # Transpose to CHW format to match training (SB3 auto-adds VecTransposeImage)
                if len(obs.shape) == 3 and obs.shape[2] == 3:  # HWC format
                    obs_for_model = obs.transpose(2, 0, 1)  # Convert to CHW
                else:
                    obs_for_model = obs

                # Debug: validate observations on first step
                if episode == 1 and step_count == 0:
                    print(f"Debug - obs shape: {obs.shape} -> transposed: {obs_for_model.shape}")
                    print(f"Debug - obs range: [{obs.min()}, {obs.max()}], std: {obs.std():.1f}")
                    print(f"Debug - obs IS varying: {obs.std() > 10} (should be True)")

                # Ensure observation is contiguous in memory
                obs_for_model = np.ascontiguousarray(obs_for_model)

                # Predict action using the trained model
                action, _states = model.predict(obs_for_model, deterministic=True)

                # Convert action from numpy array to int
                action = int(action)

                # Track action frequency
                action_counts[action] = action_counts.get(action, 0) + 1

                # Step the environment
                obs, rewards, dones, infos = safe_step(env, [action])
                reward = rewards[0]
                done = dones[0]
                info = infos[0]
                episode_reward += reward
                step_count += 1

                display_obs, _, _, _ = safe_step(display_env, action)

                # Use observation as the frame (it's already in HWC format: 240x256x3)
                frame = display_obs.copy()

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
                padding_top = 100  # space from top
                line_height = 30  # vertical spacing between lines

                cv2.putText(frame, f"Episode: {episode}", (10, padding_top),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Reward: {episode_reward:.1f}", (10, padding_top + line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"X Position: {x_pos}", (10, padding_top + 2*line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Steps: {step_count}", (10, padding_top + 3*line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


                cv2.imshow("Trained Mario Agent", frame)

                # FPS limiting
                if cv2.waitKey(int(frame_dt * 1000)) & 0xFF == 27:
                    print("\n\nStopped by user (ESC pressed)")
                    done = True
                    episode = total_episodes  # Exit outer loop too

            # Save episode stats
            episode_stats.append({
                'episode': episode,
                'reward': episode_reward,
                'steps': step_count,
                'final_x': info.get('x_pos', 0),
                'actions': action_counts.copy()
            })

            # Update overall action counts
            for action_id, count in action_counts.items():
                all_action_counts[action_id] = all_action_counts.get(action_id, 0) + count

            # Find most common action this episode
            action_names = ['NOOP', 'right', 'right+A', 'right+B', 'right+A+B', 'A', 'left']
            most_common = max(action_counts.items(), key=lambda x: x[1]) if action_counts else (0, 0)
            most_common_name = action_names[most_common[0]] if most_common[0] < len(action_names) else "?"

            print(f"  Reward: {episode_reward:.1f}, Steps: {step_count}, Final X: {info.get('x_pos', 0)}, Most used: {most_common_name}")

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")

    finally:
        env.close()
        cv2.destroyAllWindows()

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        if episode_stats:
            avg_reward = sum(s['reward'] for s in episode_stats) / len(episode_stats)
            avg_x = sum(s['final_x'] for s in episode_stats) / len(episode_stats)
            max_x = max(s['final_x'] for s in episode_stats)

            print(f"\nEpisodes completed: {len(episode_stats)}")
            print(f"Average reward: {avg_reward:.1f}")
            print(f"Average final X: {avg_x:.1f}")
            print(f"Best final X: {max_x}")

            # Overall action distribution
            print("\nOverall action distribution:")
            action_names = ['NOOP', 'right', 'right+A', 'right+B', 'right+A+B', 'A', 'left']
            total_actions = sum(all_action_counts.values())
            for action_id in sorted(all_action_counts.keys()):
                count = all_action_counts[action_id]
                percentage = (count / total_actions) * 100
                action_name = action_names[action_id] if action_id < len(action_names) else f"unknown_{action_id}"
                print(f"  {action_name}: {count} times ({percentage:.1f}%)")

        print("\nEnvironment closed.")
        print("="*60)
