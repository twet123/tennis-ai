import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from config import AgentParams
from model import create_model
from train import train
import os.path
import keras

keras.config.enable_unsafe_deserialization()

# training

env = gym.make("ALE/Tennis-v5")
env = AtariPreprocessing(
    env, frame_skip=1
)  # setting frame_skip to 1 because frame-skipping of 4 is present in the original env
env = FrameStack(env, 4)
env.reset(seed=AgentParams.seed)
actions = env.action_space.n

model = create_model(actions)
target_model = create_model(actions)

if os.path.isfile("tennis-model.keras"):
    loaded_model = keras.saving.load_model("tennis-model.keras")
    model.set_weights(loaded_model.get_weights())
    target_model.set_weights(loaded_model.get_weights())

train(env, actions, model, target_model)
model.save("tennis-model.keras")

env.close()

# playing

# import numpy as np

# env = gym.make("ALE/Tennis-v5", render_mode="human")
# env = AtariPreprocessing(
#     env, frame_skip=1
# )  # setting frame_skip to 1 because frame-skipping of 4 is present in the original env
# env = FrameStack(env, 4)
# actions = env.action_space.n
# n_state, _ = env.reset(seed=AgentParams.seed)

# loaded_model = keras.saving.load_model("tennis-model.keras")
# model = create_model(actions)
# model.set_weights(loaded_model.get_weights())

# done = False
# score = 0
# while not done:
#     env.render()
#     action = model.predict(np.expand_dims(n_state, 0))
#     action = keras.ops.argmax(action[0]).numpy()
#     n_state, reward, done, _, _ = env.step(action)
#     score += reward

# print(f"Game over! Score {score}")

# env.close()
