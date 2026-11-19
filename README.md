# IMPLEMENTATION OF ON-POLICY MONTE CARLO CONTROL USING GYMNASIUM FROZENLAKE-V1
# 2. Initialize Q-table
# ------------------------------------------------------------

Q = np.zeros((n_states, n_actions))


# ------------------------------------------------------------
# 3. Parameters
# ------------------------------------------------------------

alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

num_episodes = 10000

# Store reward obtained in every episode
episode_rewards = []


# ------------------------------------------------------------
# 4. Epsilon-Greedy Action Selection
# ------------------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    return np.argmax(Q[state])


# ------------------------------------------------------------
# 5. Monte Carlo Control
# ------------------------------------------------------------

for episode in range(num_episodes):

    # Reset environment
    state, info = env.reset()

    # Store:
    # (state, action, reward)
    episode_data = []

    total_reward = 0

    terminated = False
    truncated = False


    # --------------------------------------------------------
    # Generate Complete Episode
    # --------------------------------------------------------

    while not terminated and not truncated:

        # Select action using epsilon-greedy policy
        action = epsilon_greedy_action(
            state,
            epsilon
        )

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        # Store experience
        episode_data.append(
            (state, action, reward)
        )

        # Accumulate reward
        total_reward += reward

        # Move to next state
        state = next_state


    # --------------------------------------------------------
    # Calculate Return and Update Q-table
    # --------------------------------------------------------

    G = 0

    # Traverse the episode backwards
