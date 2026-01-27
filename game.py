import time
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Create env and bind the simple action set
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

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

# Choose a deterministic action index to try moving right.
# If you printed the mapping above, set `ACTION_TO_TRY` to the index that corresponds to 'right' or 'right + A'.
ACTION_TO_TRY = 1   # change this if your mapping shows a different index for 'right'

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


