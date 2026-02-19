"""
Training script for Super Mario Bros using PPO (Proximal Policy Optimization)
"""
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
import os
import sys

# Import the env factory from game.py
from MultiEvalCallback import MultiLevelEvalCallback
from game import make_mario_env, make_mario_level_env

# Monkey patch for JoypadSpace compatibility
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

if __name__ == "__main__":
    # Configuration
    NUM_ENV = 16  # Number of parallel environments
    TOTAL_TIMESTEPS = 15_000_000  # Total training steps (increase for better results)
    SAVE_FREQ = 50_000  # Save model every N steps
    MODEL_DIR = "./models"  # Directory to save models
    LOG_DIR = "./logs"  # Directory for tensorboard logs

    # Speed optimizations
    FRAME_SKIP = 4  # Increase to 6 for faster training (less steps needed)
    USE_CUSTOM_REWARD = True  # Set to False to speed up if custom reward is slow

    # Create directories
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

    levels = ["SuperMarioBros-1-1-v0"]

    # Create vectorized environment
    print("\nCreating vectorized environment...")
    env_fns = [make_mario_level_env(
        level=levels[i % len(levels)],
        frame_skip=FRAME_SKIP,
        use_custom_reward=USE_CUSTOM_REWARD,
        action_repeat_prob=0.02,  # 2% action repeat during training
    ) for i in range(NUM_ENV)]
    vec_env = SubprocVecEnv(env_fns)
    # REMOVED VecTransposeImage - it was breaking training!
    # vec_env = VecTransposeImage(vec_env)

    vec_env = VecFrameStack(vec_env, n_stack=4)

    # Create checkpoint callback to save models during training
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // NUM_ENV,  # Divide by NUM_ENV because it counts per environment
        save_path=MODEL_DIR,
        name_prefix="mario_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_env = DummyVecEnv([make_mario_level_env(
        level="SuperMarioBros-1-1-v0",
        frame_skip=FRAME_SKIP,
        use_custom_reward=USE_CUSTOM_REWARD,
        action_repeat_prob=0.0,
    )])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    n_steps = 2048

    # Create PPO model
    print("\nCreating PPO model...")
    model = PPO(
        "CnnPolicy",  # CNN policy for image-based observations
        vec_env,
        verbose=1,  # Print training progress
        tensorboard_log=LOG_DIR,
        device="auto",  # "auto", "cuda", or "cpu" - auto detects GPU
        learning_rate=1e-4,  # FIXED: Standard rate for PPO
        n_steps=n_steps,  # Steps to collect before updating
        batch_size=256,  # FIXED: Smaller for more stable updates
        n_epochs=10,  # FIXED: More epochs for better learning
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda parameter
        clip_range=0.2,  # PPO clipping parameter
        ent_coef=0.02,  # FIXED: Standard exploration (0.05 was too high!)
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping for stability
        target_kl=0.05
    )
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        try:
            # Load the trained model
            model.set_parameters(model_path, exact_match=True, device="auto")
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

    eval_levels = [
        "SuperMarioBros-1-1-v0"
    ]

    eval_callback = MultiLevelEvalCallback(
        model=model,
        levels=eval_levels,
        n_eval_episodes=3,
        best_model_save_path="./best_model",
        eval_freq=2048,
        verbose=1
    )
    
    callback = CallbackList([eval_callback, checkpoint_callback])


    print("\nStarting training...")
    print("You can monitor progress with: tensorboard --logdir ./logs")
    print("Press Ctrl+C to stop training early\n")
    try:
        # Train the model
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False
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
