import gym
import gym_minigrid

from gym_minigrid.wrappers import *
def make_env(env_key, seed=None,blocked=None):
    try:
        if blocked is None:
            env = gym.make(env_key)
        else:
            env = gym.make(env_key,blocked=blocked)
    except (TypeError, gym.error.Error):
        env = gym.make(env_key)
    # env = RGBImgPartialObsWrapper(env) # Get pixel observations
    # env = ImgObsWrapper(env) # Get rid of the 'mission' field
    # obs = env.reset() # This now produces an RGB tensor only
    env.seed(seed)
    return env