import random
from collections import namedtuple
import numpy as np
import tensorflow as tf

# Ensures TensorFlow runs in eager mode, making debugging easier
tf.config.run_functions_eagerly(True)

# A simple way to store experiences (state, action, reward, etc.)
experience = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])

def get_action(q_values, epsilon):
    """
    Picks an action using an epsilon-greedy strategy.
    With probability epsilon, we explore (random action); otherwise, we exploit (choose best action).

    Args:
        q_values (Tensor): Q-values for all possible actions.
        epsilon (float): Probability of choosing a random action.

    Returns:
        int: The chosen action index.
    """
    if np.random.rand() < epsilon:  # Explore (random action)
        return np.random.randint(q_values.shape[-1])
    else:  # Exploit (choose action with highest Q-value)
        return int(np.argmax(q_values))

def update_target_network(q_network, target_q_network):
    """
    Copies the weights from the Q-network to the target network.
    This helps stabilize training by keeping a fixed target for a while.
    """
    target_q_network.set_weights(q_network.get_weights())

def check_update_conditions(t, update_interval, memory_buffer, min_samples=32):
    """
    Decides whether it's time to update the Q-network.
    Updates happen every few steps, but only if we have enough experience stored.

    Args:
        t (int): Current timestep.
        update_interval (int): How often to update.
        memory_buffer (deque): Replay memory storing past experiences.
        min_samples (int): Minimum number of experiences needed for an update.

    Returns:
        bool: True if an update should happen, False otherwise.
    """
    return (t % update_interval == 0) and (len(memory_buffer) >= min_samples)

def get_experiences(memory_buffer, batch_size=32):
    """
    Pulls a batch of past experiences from memory for training.

    Args:
        memory_buffer (deque): The storage of past experiences.
        batch_size (int): Number of experiences to sample.

    Returns:
        tuple: Arrays of states, actions, rewards, next_states, and done flags.
    """
    batch = random.sample(memory_buffer, batch_size)  # Randomly sample from memory

    # Extract and return each part of the experience tuple as separate arrays
    states = np.array([exp.state for exp in batch])
    actions = np.array([exp.action for exp in batch])
    rewards = np.array([exp.reward for exp in batch], dtype=np.float32)
    next_states = np.array([exp.next_state for exp in batch])
    done_flags = np.array([exp.done for exp in batch], dtype=np.float32)

    return states, actions, rewards, next_states, done_flags

def create_video(filename, env, q_network):
    """
    Placeholder for recording a video of the trained agent playing the game.
    """
    print(f"Video saved to {filename}")

def embed_mp4(filename):
    """
    Placeholder for embedding an MP4 video in a Jupyter notebook.
    """
    print(f"Video {filename} embedded.")

"""
CartPole Deep Q-Learning Training Script

This script trains a Q-network for the CartPole-v1 environment using Deep Q-Learning.
The code is modularized for improved readability, maintainability, and easier integration of new features.
"""

import time
import random
from collections import deque, namedtuple

import numpy as np
import tensorflow as tf
import gym

# -----------------------------------------------------------------------------
# Global Constants and Hyperparameters
# -----------------------------------------------------------------------------
NUM_EPISODES = 2000
MAX_TIMESTEPS = 1000
MEMORY_SIZE = 10000           # Capacity of replay buffer
NUM_STEPS_FOR_UPDATE = 4      # Frequency of network updates
GAMMA = 0.99                  # Discount factor
INITIAL_EPSILON = 1.0         # Initial exploration rate
EPSILON_DECAY = 0.995         # Decay rate per episode
MIN_EPSILON = 0.01            # Minimum exploration rate
AVG_POINTS_WINDOW = 100       # Window size for averaging rewards

# Define a namedtuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# -----------------------------------------------------------------------------
# Utility Functions (Replaces "utils")
# -----------------------------------------------------------------------------

def get_action(q_values, epsilon):
    """
    Select an action using an epsilon-greedy policy.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(q_values.shape[-1])
    return int(np.argmax(q_values))

def update_target_network(q_network, target_q_network):
    """
    Update the target network weights to match those of the Q-network.
    """
    target_q_network.set_weights(q_network.get_weights())

def check_update_conditions(t, update_interval, memory_buffer, min_samples=32):
    """
    Check if conditions are met to update the network.
    """
    return (t % update_interval == 0) and (len(memory_buffer) >= min_samples)

def get_experiences(memory_buffer, batch_size=32):
    """
    Sample a mini-batch of experiences from the memory buffer.
    """
    batch = random.sample(memory_buffer, batch_size)
    states = np.array([exp.state for exp in batch])
    actions = np.array([exp.action for exp in batch])
    rewards = np.array([exp.reward for exp in batch], dtype=np.float32)
    next_states = np.array([exp.next_state for exp in batch])
    done_flags = np.array([exp.done for exp in batch], dtype=np.float32)
    return states, actions, rewards, next_states, done_flags

# -----------------------------------------------------------------------------
# Deep Q-Learning Functions
# -----------------------------------------------------------------------------

def compute_loss(experiences, gamma, q_network, target_q_network):
    """
    Calculates the Mean Squared Error (MSE) loss for a batch of experiences.

    The goal is to minimize the difference between predicted Q-values
    and target Q-values based on the Bellman equation.

    Args:
        experiences (tuple): A batch of (states, actions, rewards, next_states, done_flags).
        gamma (float): Discount factor for future rewards.
        q_network (tf.keras.Model): The primary Q-network.
        target_q_network (tf.keras.Model): The fixed target Q-network.

    Returns:
        Tensor: The computed loss value.
    """

    # Unpack the batch of experiences
    states, actions, rewards, next_states, done_flags = experiences

    # Get max Q-value for the next state from the target network (max Q^(s', a))
    max_next_q = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Compute target Q-values (y = R + γ max Q^(s', a)),
    # but if done, we use just the reward (no future reward)
    y_targets = rewards + (gamma * max_next_q * (1 - done_flags))

    # Get the Q-values for the current state from the Q-network
    q_values = q_network(states)

    # Select the Q-values corresponding to the actions taken
    batch_indices = tf.range(tf.shape(q_values)[0])
    chosen_q_values = tf.gather_nd(q_values, tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1))

    # Compute Mean Squared Error (MSE) loss between actual and predicted Q-values
    loss = tf.keras.losses.MeanSquaredError()(y_targets, chosen_q_values)

    return loss


@tf.function
def agent_learn(experiences, gamma, q_network, target_q_network, optimizer):
    """
    Runs a single learning step: computes the loss, updates the Q-network,
    and synchronizes the target network.

    Args:
        experiences (tuple): A batch of (states, actions, rewards, next_states, done_flags).
        gamma (float): Discount factor for future rewards.
        q_network (tf.keras.Model): The primary Q-network (to be updated).
        target_q_network (tf.keras.Model): The target Q-network (fixed for stability).
        optimizer (tf.keras.optimizers.Optimizer): The optimizer to update the network weights.
    """

    with tf.GradientTape() as tape:
        # Compute the loss based on the experiences
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Calculate gradients of the loss with respect to the Q-network parameters
    gradients = tape.gradient(loss, q_network.trainable_variables)

    # Apply the gradients to update the Q-network weights
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # Periodically update the target network to stabilize training
    update_target_network(q_network, target_q_network)

def run_episode(env, q_network, target_q_network, optimizer, memory_buffer, epsilon, gamma):
    """
    Runs a single episode using an epsilon-greedy policy.

    The agent takes actions, collects rewards, stores experiences,
    and periodically updates the Q-network.

    Args:
        env: The environment instance.
        q_network (tf.keras.Model): The main Q-network.
        target_q_network (tf.keras.Model): The target Q-network.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for training.
        memory_buffer (deque): Stores past experiences for replay.
        epsilon (float): Exploration rate (controls random actions).
        gamma (float): Discount factor for future rewards.

    Returns:
        float: Total reward accumulated during the episode.
    """

    state = env.reset()  # Start with a fresh environment
    total_reward = 0.0  # Track how well the agent performs

    for t in range(MAX_TIMESTEPS):  # Run the episode for a max number of steps
        # Format the state correctly for the Q-network (adds batch dimension)
        state_input = np.expand_dims(state, axis=0)

        # Get Q-values for the current state and choose an action
        q_vals = q_network(state_input)
        action = get_action(q_vals, epsilon)

        # Take the chosen action in the environment and observe the result
        next_state, reward, done, _ = env.step(action)

        # Store the experience in memory (for future training)
        memory_buffer.append(Experience(state, action, reward, next_state, done))

        # Periodically update the Q-network using stored experiences
        if check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer):
            experiences = get_experiences(memory_buffer)
            agent_learn(experiences, gamma, q_network, target_q_network, optimizer)

        # Move to the next state
        state = next_state.copy()
        total_reward += reward  # Accumulate the reward

        # End the episode if the environment signals "done"
        if done:
            break

    return total_reward  # Return the total points scored in this episode

def train_agent(env, q_network, target_q_network, optimizer):
    """
    Main training loop that runs multiple episodes and updates the Q-networks.
    """
    start_time = time.time()
    total_rewards_history = []
    epsilon = INITIAL_EPSILON
    memory_buffer = deque(maxlen=MEMORY_SIZE)

    # Initialize target network
    update_target_network(q_network, target_q_network)

    for episode in range(NUM_EPISODES):
        episode_reward = run_episode(env, q_network, target_q_network, optimizer, memory_buffer, epsilon, GAMMA)
        total_rewards_history.append(episode_reward)
        avg_reward = np.mean(total_rewards_history[-AVG_POINTS_WINDOW:])

        # Decay epsilon
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        print(f"Episode {episode + 1} | Avg Reward (last {AVG_POINTS_WINDOW} eps): {avg_reward:.2f} | Epsilon: {epsilon:.3f}", end="\r")
        if (episode + 1) % AVG_POINTS_WINDOW == 0:
            print(f"\nEpisode {episode + 1} | Avg Reward (last {AVG_POINTS_WINDOW} eps): {avg_reward:.2f}")

        # Check if environment is solved
        if avg_reward >= 200.0:
            print(f"\nEnvironment solved in {episode + 1} episodes!")
            q_network.save('cartpole_q_model.h5')
            break

    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f} s ({total_time/60:.2f} min)")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize the CartPole-v1 environment
    env = gym.make("CartPole-v1")
    num_actions = env.action_space.n
    input_shape = env.observation_space.shape

    def build_q_network(input_shape, num_actions):
        """
        Build a simple feed-forward Q-network.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_actions)
        ])
        return model

    # Build Q-networks
    q_network = build_q_network(input_shape, num_actions)
    target_q_network = build_q_network(input_shape, num_actions)

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Start training
    train_agent(env, q_network, target_q_network, optimizer)
