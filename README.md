**Tic-Tac-Toe Reinforcement Learning with Q-Learning**

This project implements a Q-learning algorithm to train two agents, X and O, to play Tic-Tac-Toe against each other. Each agent maintains its own Q-table to estimate the expected rewards of state-action pairs and learns through self-play.

Overview

    Q-Learning Algorithm: Agents update their Q-values using the Bellman equation to improve their policy over time.
    Epsilon-Greedy Strategy: Controls the exploration-exploitation trade-off during training.
    Draw Game Tracking: Tracks draw percentages over time, showing the agents' progression toward optimal play.
    Custom Tic-Tac-Toe Environment: Uses a TicTacToe class to simulate the game environment.
    Performance Monitoring: Prints draw rates at regular intervals during training.

**How It Works**

Q-Table Initialization:

    Each agent (X and O) maintains a separate Q-table.
    State-action pairs are initialized with a default Q-value of 0.0.

Gameplay:

    The game starts in a reset state, and players alternate turns.
    On each turn, the current player selects an action using an epsilon-greedy policy:
        Explore: Choose a random action with probability ϵϵ.
        Exploit: Choose the action with the highest Q-value with probability 1−ϵ1−ϵ.

Q-Table Updates:

    After each action, the Q-tables are updated using the Q-learning formula:
    
    Q(s,a)←Q(s,a)+α[r+γmax⁡aQ(s′,a)−Q(s,a)]
    
    Both agents update their Q-tables from their respective perspectives.

Termination:

    The game ends when one player wins or the board is full.
    Rewards:
        
    +1+1 for a win, −1−1 for a loss, and 00 for a draw.
    
    The draw status is logged for performance evaluation.

Exploration Decay:

    The epsilon value decays over time, reducing exploration and favoring exploitation as the agents improve.

Performance Tracking:

    Draw rates over the last 1,000 games are printed every 1,000 games.
    Results are saved to draw_games.csv.

**Cartpole Reinforcement Learning with Polynomial Features**

This project demonstrates a reinforcement learning (RL) approach to solving the CartPole-v1 environment from OpenAI's gymnasium library. It uses linear models with stochastic gradient descent (SGD) and polynomial feature expansion to approximate the Q-value function for each action.

**Overview**

    The agent learns to balance a pole on a cart by choosing one of two actions (move left or move right) based on the state of the environment. Key features include:  
    Q-learning with Experience Replay: The agent learns by storing experiences in a replay buffer and sampling mini-batches for training.
    Polynomial Feature Expansion: The state features are expanded to a higher-dimensional space using polynomial transformations, allowing the model to capture non-linear relationships.
    Epsilon-Greedy Policy: Exploration is controlled using an epsilon-greedy approach with exponential decay.
    SGDRegressor Models: Separate SGD models are trained for each action.

**How It Works**

State Representation

       Each state is a 4-dimensional vector representing:

       Cart position
       Cart velocity
       Pole angle
       Pole angular velocity

       These features are transformed into a higher-dimensional space using polynomial feature expansion.

Experience Replay

        A replay buffer stores experiences of the form (state, action, reward, next_state, done) for training.
        A random batch of experiences is sampled during each step to train the models, breaking temporal correlations.

Q-value Approximation
   
        Separate SGDRegressor models are trained to approximate the Q-value function for each action.
        The target for Q-value is computed as:

        Target=reward+γ×max(Q(next state,all actions))

Epsilon-Greedy Policy

        The agent selects actions using an epsilon-greedy policy:
        With probability , the agent explores by choosing a random action.
        Otherwise, it exploits by choosing the action with the highest Q-value.

