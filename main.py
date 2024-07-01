import gymnasium as gym
import random
from agents.keras_dqn_agent import build_model, build_agent
from keras import optimizers

env = gym.make("ALE/Tennis-v5")
height, width, channels = env.observation_space.shape
actions = env.action_space.n

# episodes = 5
# for episode in range(episodes):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action = random.randrange(actions)
#         n_state, reward, terminated, truncated, info = env.step(action)
#         score += reward
#         done = terminated or truncated

#     print(f"Episode: {episode}, Score: {score}")

model = build_model(height, width, channels, actions)
model.summary()

dqn = build_agent(model, actions)
dqn.compile(optimizers.Adam(learning_rate=1e-4))
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

env.close()
