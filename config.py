class AgentParams:
    seed = 8
    gamma = 0.99  # Disount factor (zanemarivanje nagrade)
    epsilon = 1.0  # Epsilon greedy
    epsilon_min = 0.1
    epsilon_max = 1.0
    epsilon_interval = (
        epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    batch_size = 32  # Size of batch taken from replay memory (buffer)
    max_steps_per_episode = (
        10000  # 10000 for testing/showcase, 10000000 for actual training
    )
    max_episodes = 10  # 10 for testing/showcase, more for actual training
