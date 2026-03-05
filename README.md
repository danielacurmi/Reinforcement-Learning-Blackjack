# Reinforcement-Learning-Blackjack
This project implements a Reinforcement Learning (RL) approach to the game of Blackjack. It explores various RL algorithms to train agents to play the game effectively and visualizes the results of their performance.

## Project Structure

### Scripts
- **main.py**: The main script to run the project. It initializes the environment, trains the RL agents, and evaluates their performance.
- **blackjack.py**: Contains the implementation of the Blackjack game environment.
- **deck.py**: Handles the creation and management of the deck of cards.
- **hand.py**: Manages the player's and dealer's hands.
- **rl_algorithms.py**: Implements various RL algorithms such as Monte Carlo (MC), Q-Learning (QL), SARSA, and Deep Q-Learning (DQL).
- **rl_data.py**: Handles data storage and retrieval for training and evaluation.
- **data_vis.py**: Generates visualizations for the results of the RL training and evaluation.

### Plots
- **DealerAdvantage/**: Contains visualizations showing the dealer's advantage.
- **StateActionCounts/**: Visualizes state-action counts for different RL algorithms (MC, QL, SARSA, DQL).
- **StrategyTable/**: Displays strategy tables for the trained agents.
- **TotalUniqueSACounts/**: Shows the total unique state-action counts for each algorithm.
- **WinRate/**: Visualizes the win rates of the agents trained using different RL algorithms.
