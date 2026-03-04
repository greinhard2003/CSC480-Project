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
import random  # ADDED (for optional shuffle)

# Import the env factory from game.py
from MultiEvalCallback import MultiLevelEvalCallback
from game import make_mario_env, make_mario_level_env  # <-- CHANGED: removed make_mario_multi_level_env

# Monkey patch for JoypadSpace compatibility
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

if __name__ == "__main__":
    # Configuration
    NUM_ENV = 16  # Number of parallel environments
    TOTAL_TIMESTEPS = 6_000_000  # Total training steps (increase for better results)
    SAVE_FREQ = 200_000  # Save model every N steps
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

    # =========================
    # 50/50 level mix
    # =========================
    levels_w1 = [
        "SuperMarioBros-1-1-v0",
        "SuperMarioBros-1-2-v0",
        "SuperMarioBros-1-3-v0",
        "SuperMarioBros-1-4-v0",
    ]

    levels_w2 = [
        "SuperMarioBros-2-1-v0",
        "SuperMarioBros-2-2-v0",
        "SuperMarioBros-2-3-v0",
        "SuperMarioBros-2-4-v0",
    ]

    levels = levels_w1 + levels_w2  # 8 total levels

    # Create vectorized environment
    print("\nCreating vectorized environment...")

    # ============================================================
    # NEW: Fixed assignment sampler (2 envs per level)
    # - avoids uneven sampling and helps with plateaus/forgetting
    # ============================================================
    assert NUM_ENV % len(levels) == 0, (
        f"NUM_ENV ({NUM_ENV}) must be a multiple of number of levels ({len(levels)})."
    )
    envs_per_level = NUM_ENV // len(levels)  # for 16 envs and 8 levels -> 2 each

    env_fns = []
    for level in levels:
        for _ in range(envs_per_level):
            env_fns.append(
                make_mario_level_env(
                    level=level,
                    frame_skip=FRAME_SKIP,
                    use_custom_reward=USE_CUSTOM_REWARD
                )
            )

    # Optional: shuffle so workers aren't grouped by level in process order
    random.shuffle(env_fns)

    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    # Optional explicit transpose (SB3 often auto-wraps for CnnPolicy, but explicit is fine):
    # vec_env = VecTransposeImage(vec_env)

    # Create checkpoint callback to save models during training
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // NUM_ENV,  # Divide by NUM_ENV because it counts per environment
        save_path=MODEL_DIR,
        name_prefix="mario_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Eval env (fixed one env per level)
    eval_env = DummyVecEnv([
        make_mario_level_env(
            level,
            frame_skip=FRAME_SKIP,
            use_custom_reward=USE_CUSTOM_REWARD
        )
        for level in levels
    ])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    # eval_env = VecTransposeImage(eval_env)

    n_steps = 2048

    # Create PPO model
    print("\nCreating PPO model...")
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
        learning_rate=1e-4,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02
    )

    # Load baseline weights if provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        try:
            model.set_parameters(model_path, exact_match=True, device="auto")
            print("Model loaded successfully!")
        except FileNotFoundError:
            print(f"\nERROR: Model not found at {model_path}")
            print("\nAvailable models:")
            if os.path.exists("./models"):
                models = [f for f in os.listdir("./models") if f.endswith(".zip")]
                for m in models:
                    print(f"  - ./models/{m}")
            else:
                print("  No models directory found. Train a model first with: python train.py")
            sys.exit(1)

    eval_levels = [
        "SuperMarioBros-1-1-v0",
        "SuperMarioBros-1-2-v0",
        "SuperMarioBros-1-3-v0",
        "SuperMarioBros-1-4-v0",
        "SuperMarioBros-2-1-v0",
        "SuperMarioBros-2-2-v0",
        "SuperMarioBros-2-3-v0",
        "SuperMarioBros-2-4-v0",
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
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False
        )

        final_model_path = f"{MODEL_DIR}/best_world2.zip"
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