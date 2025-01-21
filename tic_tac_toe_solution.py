import random
import pandas as pd
from typing import Dict
from tic_tac_toe_env import TicTacToe

N_GAMES = 200000
ALPHA = 1.0  
GAMMA = 1.0 
INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.9999
MIN_EPSILON = 0.0001
EPSILON = INITIAL_EPSILON

# Initialize the tic-tac-toe environment
env = TicTacToe()

# list of booleans indicating draw games
draw_games = []

# Initialize Q-tables for both X and O.
q_x: Dict[str, Dict[int, float]] = {}  
q_o: Dict[str, Dict[int, float]] = {}  

"""
Get Q-value with a default value of 0.0 
"""
def get_q_value(q_table, state, action):
    return q_table.get(state, {}).get(action, 0.0) # 0.0 is a neutral initialization

"""
Set Q-value in the Q-table
"""
def set_q_value(q_table, state, action, value):
    if state not in q_table:
        q_table[state] = {}
    q_table[state][action] = value

"""
Select an action using epsilon-greedy strategy
"""
def epsilon_greedy_selection(q_table, state, epsilon):
    actions = env.get_available_actions()
    if random.random() < epsilon:  # Explore
        return random.choice(actions)
    else:  # Exploit
        q_values = [get_q_value(q_table, state, a) for a in actions]
        return actions[q_values.index(max(q_values))]

"""
Updating both Q-tables from each perspective
"""
def update_qTable(q_table_1, q_table_2, state, action, reward, next_state, alpha=ALPHA, gamma=GAMMA):

    #Update Q-table 1 from X perspective
    max_future_q = max([get_q_value(q_table_2, next_state, a) for a in env.get_available_actions()],
        default=0.0
    )
    current_q = get_q_value(q_table_1, state, action)
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q) # Apply the Q-learning update formula for Q-table 1
    set_q_value(q_table_1, state, action, new_q)

    #Update Q-table 2 from O perspective
    max_future_q = max([get_q_value(q_table_1, next_state, a) for a in env.get_available_actions()],  
        default=0.0
    )
    current_q = get_q_value(q_table_2, state, action)
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q) # Apply the Q-learning update formula for Q-table 2
    set_q_value(q_table_2, state, action, new_q)


"""
Play a single game and return whether it was a draw
"""
def play_game(epsilon):
    state, _ = env.reset()  # Reset returns the observation and an empty dict
    terminated = False
    total_reward = 0

    while not terminated:
        # Get player's turn
        player_turn = env.get_player_turn()

        if player_turn == 1:  # X player's turn
            action = epsilon_greedy_selection(q_x, state, epsilon)
        else:  # O player's turn
            action = epsilon_greedy_selection(q_o, state, epsilon)

        # Execute the chosen action
        new_state, reward, terminated, _, _ = env.step(action)
        total_reward += reward

        if reward == 1:  # X wins
            x_reward = 1
            o_reward = -1
        elif reward == -1:  # O wins
            x_reward = -1
            o_reward = 1
        else:
            x_reward = 0
            o_reward = 0

        update_qTable(q_x, q_o, state, action, x_reward, new_state, alpha=ALPHA, gamma=GAMMA)
        update_qTable(q_o, q_x, state, action, o_reward, new_state, alpha=ALPHA, gamma=GAMMA)

        state = new_state

    return total_reward == 0  # Whether the game was a draw

recent_draws = []  # Sliding window to track the last 1000 games

for game in range(N_GAMES):
    draw = play_game(EPSILON)
    draw_games.append(draw)
    recent_draws.append(draw)

    # Keep the recent_draws list to the last 1000 entries
    if len(recent_draws) > 1000:
        recent_draws.pop(0)

    # Decay
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    if game % 1000 == 0:
        draw_percentage = sum(recent_draws) / len(recent_draws) * 100
        print(f"Game {game}: Draw rate in last 1000 games: {draw_percentage:.2f}%, Epsilon: {EPSILON:.4f}")

# Save record of draw games
pd.Series(data=draw_games, name='draw_games').to_csv('draw_games.csv', index=False)
