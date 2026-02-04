"""
Training script for Super Mario Bros using PPO (Proximal Policy Optimization)
"""
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Import the wrappers from game.py
from game import FrameSkipWrapper, CustomRewardWrapper

# Monkey patch for JoypadSpace compatibility
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)


def make_mario_env(render_mode="rgb_array", use_custom_reward=True, frame_skip=4):
    def _init():
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
    return _init


if __name__ == "__main__":
    # Configuration
    NUM_ENV = 8  # Number of parallel environments
    TOTAL_TIMESTEPS = 1_000_000  # Total training steps (increase for better results)
    SAVE_FREQ = 50_000  # Save model every N steps
    MODEL_DIR = "./models"  # Directory to save models
    LOG_DIR = "./logs"  # Directory for tensorboard logs

    # Speed optimizations
    FRAME_SKIP = 4  # Increase to 6 for faster training (less steps needed)
    USE_CUSTOM_REWARD = True  # Set to False to speed up if custom reward is slow

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Check system resources
    cpu_count = os.cpu_count()

    print("="*60)
    print("Super Mario Bros - Reinforcement Learning Training")
    print("="*60)
    print(f"System Info:")
    print(f"  CPU cores available: {cpu_count}")
    print(f"  Number of parallel environments: {NUM_ENV}")
    if NUM_ENV < cpu_count - 2:
        print(f"  💡 Tip: You could increase NUM_ENV to {cpu_count - 2} for faster training")
    print(f"\nTraining Configuration:")
    print(f"  Total training timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Model save frequency: {SAVE_FREQ:,}")
    print(f"  Models will be saved to: {MODEL_DIR}")
    print(f"  Logs will be saved to: {LOG_DIR}")
    print("="*60)

    # Create vectorized environment
    print("\nCreating vectorized environment...")
    env_fns = [make_mario_env(frame_skip=FRAME_SKIP, use_custom_reward=USE_CUSTOM_REWARD) for _ in range(NUM_ENV)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecTransposeImage(vec_env)

    # Create checkpoint callback to save models during training
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // NUM_ENV,  # Divide by NUM_ENV because it counts per environment
        save_path=MODEL_DIR,
        name_prefix="mario_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Create PPO model
    print("\nCreating PPO model...")
    model = PPO(
        "CnnPolicy",  # CNN policy for image-based observations
        vec_env,
        verbose=1,  # Print training progress
        tensorboard_log=LOG_DIR,
        device="auto",  # "auto", "cuda", or "cpu" - auto detects GPU
        learning_rate=3e-4,  # Learning rate
        n_steps=2048,  # Steps to collect before updating (increased for speed)
        batch_size=512,  # Batch size for training (increased for speed)
        n_epochs=4,  # Number of epochs per update (reduced for speed)
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda parameter
        clip_range=0.2,  # PPO clipping parameter
        ent_coef=0.01,  # Entropy coefficient (encourages exploration)
    )

    print("\nStarting training...")
    print("You can monitor progress with: tensorboard --logdir ./logs")
    print("Press Ctrl+C to stop training early\n")

    try:
        # Train the model
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True,
        )

        # Save the final model
        final_model_path = f"{MODEL_DIR}/mario_ppo_final.zip"
        model.save(final_model_path)
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Final model saved to: {final_model_path}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_model_path = f"{MODEL_DIR}/mario_ppo_interrupted.zip"
        model.save(interrupted_model_path)
        print(f"Model saved to: {interrupted_model_path}")

    finally:
        vec_env.close()
        print("\nEnvironment closed.")
