import keras
from config import AgentParams
import numpy as np
import tensorflow as tf
import h5py
import os.path


def load_h5(file_name):
    if os.path.isfile(f"{file_name}.h5"):
        with h5py.File(f"{file_name}.h5", "r") as hf:
            arr = hf[file_name][:]
            return arr.tolist()

    return []


def save_h5(file_name, arr):
    with h5py.File(f"{file_name}.h5", "w") as hf:
        hf.create_dataset(file_name, data=np.asarray(arr))


def train(env, actions, model, target_model):
    optimizer = keras.optimizers.Adam(
        learning_rate=AgentParams.learning_rate, clipnorm=1.0
    )
    loss_function = keras.losses.Huber()

    # replay memory
    action_history = load_h5("action_history")
    state_history = load_h5("state_history")
    state_next_history = load_h5("state_next_history")
    rewards_history = load_h5("rewards_history")
    print(rewards_history)
    done_history = load_h5("done_history")
    print(done_history)
    episode_reward_history = load_h5("episode_reward_history")
    print(episode_reward_history)

    running_reward = 0
    episode_count = 0
    frame_count = 0

    while True:
        observation, _ = env.reset()
        state = np.array(observation)
        episode_reward = 0

        for timestep in range(1, AgentParams.max_steps_per_episode):
            frame_count += 1

            # exploration
            if (
                frame_count < AgentParams.epsilon_random_frames
                or AgentParams.epsilon > np.random.rand(1)[0]
            ):
                action = np.random.choice(actions)
            # exploitation
            else:
                state_tensor = keras.ops.convert_to_tensor(state)
                state_tensor = keras.ops.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                action = keras.ops.argmax(action_probs[0]).numpy()

            AgentParams.epsilon -= (
                AgentParams.epsilon_interval / AgentParams.epsilon_greedy_frames
            )
            AgentParams.epsilon = max(AgentParams.epsilon, AgentParams.epsilon_min)

            state_next, reward, done, _, _ = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            if (
                frame_count % AgentParams.update_after_actions == 0
                and len(done_history) > AgentParams.batch_size
            ):
                indicies = np.random.choice(
                    range(len(done_history)), size=AgentParams.batch_size
                )

                state_sample = np.array([state_history[i] for i in indicies])
                state_next_sample = np.array([state_next_history[i] for i in indicies])
                rewards_sample = [rewards_history[i] for i in indicies]
                action_sample = [action_history[i] for i in indicies]
                done_sample = keras.ops.convert_to_tensor(
                    [float(done_history[i]) for i in indicies]
                )

                future_rewards = target_model.predict(state_next_sample)
                # q value = reward + discount * expected future reward
                updated_q_values = rewards_sample + AgentParams.gamma * keras.ops.amax(
                    future_rewards, axis=1
                )

                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                masks = keras.ops.one_hot(action_sample, actions)

                with tf.GradientTape() as tape:
                    q_values = model(state_sample)

                    q_action = keras.ops.sum(
                        keras.ops.multiply(q_values, masks), axis=1
                    )
                    loss = loss_function(updated_q_values, q_action)

                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % AgentParams.update_target_network == 0:
                target_model.set_weights(model.get_weights())
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

                model.save("tennis-model.keras")
                save_h5("action_history", action_history)
                save_h5("state_history", state_history)
                save_h5("state_next_history", state_next_history)
                save_h5("rewards_history", rewards_history)
                save_h5("done_history", done_history)
                save_h5("episode_reward_history", episode_reward_history)

            if len(rewards_history) > AgentParams.max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

        if running_reward >= 24:
            print("Solved at episode {}!".format(episode_count))
            break

        if (
            AgentParams.max_episodes > 0 and episode_count >= AgentParams.max_episodes
        ):  # Maximum number of episodes reached
            print(
                "Stopped at episode {}, running reward: {}!".format(
                    episode_count, running_reward
                )
            )
            break
