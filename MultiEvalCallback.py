from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from game import make_mario_level_env

# Vibed this one out
class MultiLevelEvalCallback(BaseCallback):
    def __init__(self, model, levels, n_eval_episodes=1, best_model_save_path="./best_model/",
                 eval_freq=2048, verbose=1):
        super().__init__(verbose)
        self.model = model
        self.levels = levels
        self.n_eval_episodes = n_eval_episodes
        self.best_mean = -float("inf")
        self.best_model_save_path = best_model_save_path
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        # Check if it is time to evaluate
        if self.n_calls % self.eval_freq == 0:
            level_rewards = []
            for level in self.levels:
                eval_env = DummyVecEnv([ make_mario_level_env(level)])
                eval_env = VecFrameStack(eval_env, n_stack=4)
                mean_reward, _ = evaluate_policy(
                    self.model,
                    eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=True,
                    render=False
                )
                eval_env.close()
                level_rewards.append(mean_reward)
            overall_mean = sum(level_rewards) / len(level_rewards)
            if self.verbose > 0:
                print(f"\nEval overall mean reward across levels: {overall_mean:.2f}")
            
            # Save model if best
            if overall_mean > self.best_mean:
                self.best_mean = overall_mean
                self.model.save(self.best_model_save_path + "/best_mean_model.zip")
                if self.verbose > 0:
                    print(f"New best model saved with mean reward {self.best_mean:.2f}")
        return True
