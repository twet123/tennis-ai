import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from config import AgentParams
from model import create_model
from train import train
import os.path
import keras

keras.config.enable_unsafe_deserialization()

# playing

import numpy as np

env = gym.make("TennisNoFrameskip-v4", render_mode="human")
env = AtariPreprocessing(
    env, frame_skip=4
)  # setting frame_skip to 1 because frame-skipping of 4 is present in the original env (when set to tennis v5)
env = FrameStack(env, 4)
actions = env.action_space.n
env.reset(seed=AgentParams.seed)

loaded_model = keras.saving.load_model("tennis-model.keras")
model = create_model(actions)
model.set_weights(loaded_model.get_weights())

done = False
score = 0
env.render()
n_state, reward, done, _, _ = env.step(1)
while not done:
    n_state = np.array(n_state)
    state_tensor = keras.ops.convert_to_tensor(n_state)
    state_tensor = keras.ops.expand_dims(state_tensor, 0)
    env.render()
    if reward == 1 or reward == -1:
        action = 1
    else:
        action_probs = model(state_tensor, training=False)
        action = keras.ops.argmax(action_probs[0]).numpy()
    print(action)
    n_state, reward, done, _, _ = env.step(action)
    score += reward

print(f"Game over! Score {score}")

env.close()
