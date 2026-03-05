import random
from blackjack import Blackjack
from rl_data import RLData
from collections import defaultdict, Counter, namedtuple

RLResult = namedtuple("RLResult", ["Q1", "Q2", "results", "sa_pairs", "num_unique_sa"])

class RLAlgorithms:
    def __init__(self, env, gamma=1.0):
        random.seed(71)
        self.env:Blackjack = env
        self.gamma = gamma
        self.data = RLData()

    def reset(self):
        self.data.reset()

    def _epsilon_greedy(self, state, epsfunc, mode) -> str:
        player_total, _, _ = state
        if player_total < 12:
            return "hit"
        if player_total == 21:
            return "stand"    

        actions = ["hit", "stand"]
        epsilon = epsfunc(self.data.k)

        if random.random() < epsilon:
            return random.choice(actions)

        match mode:
            case "montecarlo":
                q_vals = self.data._Q[state]
                return max(q_vals, key=lambda action: q_vals[action])

            case "sarsa" | "qlearning":
                q_vals = {a: self.data._Q[state][a] for a in actions}
                return max(q_vals, key=lambda action: q_vals[action])

            case "doubleq":
                q_vals = {a: self.data._Q[state][a] + self.data._Q2[state][a] for a in actions}
                return max(q_vals, key=lambda action: q_vals[action])

            case _:
                raise ValueError(f"Unknown mode '{mode}' in epsilon-greedy. Expected one of: montecarlo, sarsa, qlearning, doubleq.")
    
    def first_visit_MC(self, episode):
        g = 0
        visited = set()
        for t in reversed(range(len(episode))):
            state, action, reward, result = episode[t]
            player_total, _, _ = state

            g = self.gamma * g + reward
            if (state, action) not in visited and self.data.should_update_Value(player_total):
                self.data.set_N_value(state, action, self.data._N[state][action]+1)
                alpha = 1 / self.data._N[state][action]
                new_q_value = alpha * (g - self.data._Q[state][action])
                self.data.set_Q_value(state,action,new_q_value)
                visited.add((state, action))

        # Access the last state in episode and get the win state
        result = episode[-1][3]
        return result
    
    # Used to generate episodes from start to terminal state for Monte Carlo only
    def _generate_episode(self, epsfunc, is_exploring_start = False):
        """
        Returns: episode - list[state, action, reward, result]
        """
        episode = []
        self.env.reset()
        state = self.env.game_setup()
        player_total, _, _ = state

        while True:
            # Exploring start
            if is_exploring_start and (12 <= player_total <= 20):
                action = random.choice(['hit', 'stand'])
                is_exploring_start = False
            else:
                action = self._epsilon_greedy(state, epsfunc, mode="montecarlo")

            next_state, reward, done, result = self.env.step(action)
            episode.append((state, action, reward, result))
            state = next_state
            if done:
                break

        return episode
    
    def sarsa(self, epsfunc, sa_pairs:Counter):
        self.env.reset()
        state = self.env.game_setup()
        action = self._epsilon_greedy(state, epsfunc, mode="sarsa")
        
        while True:
            next_state, reward, done, result = self.env.step(action)
            # Get next_action immediately after getting next_state
            next_action = self._epsilon_greedy(next_state, epsfunc, mode="sarsa") if not done else 'stand'
            
            sa_pairs[state,action] += 1
            
            player_total, _, _ = state
            
            # Only process non-trivial states for updates
            if self.data.should_update_Value(player_total):
                current_n = self.data._N[state][action]
                self.data.set_N_value(state, action, current_n + 1)
                
                # Only retrieve the value if it was actually updated
                n_value = self.data._N[state][action]
                alpha = 1 / n_value
                
                target = reward
                if not done:
                    target += self.gamma * self.data._Q[next_state][next_action]
                    
                new_q_value = alpha * (target - self.data._Q[state][action])
                self.data.set_Q_value(state, action, new_q_value)
            
            if done:
                break
                
            # S = S', A = A'
            state = next_state
            action = next_action
            
        return result

    def q_learning(self, epsfunc, sa_pairs:Counter):
        self.env.reset()
        state = self.env.game_setup()

        while True:
            action = self._epsilon_greedy(state, epsfunc, mode="qlearning")
            next_state, reward, done, result = self.env.step(action)

            sa_pairs[state,action] += 1
            
            player_total, _, _ = state
            
            # Only process non-trivial states for updates
            if self.data.should_update_Value(player_total):
                current_n = self.data._N[state][action]
                self.data.set_N_value(state, action, current_n + 1)
                
                # Only retrieve the value if it was actually updated
                n_value = self.data._N[state][action]
                alpha = 1 / max(1, n_value)

                target = reward
                if not done:
                    max_q = max(self.data._Q[next_state][a] for a in ['hit', 'stand'])
                    target += self.gamma * max_q

                new_q_value = alpha * (target - self.data._Q[state][action])
                self.data.set_Q_value(state, action, new_q_value)

            if done:
                break
            # S = S'
            state = next_state

        return result
    
    # Sutton & Barto section 6.7
    def double_q_learning(self, epsfunc, sa_pairs:Counter):
        self.env.reset()
        state = self.env.game_setup()
        
        while True:
            action = self._epsilon_greedy(state, epsfunc, mode="doubleq")
            next_state, reward, done, result = self.env.step(action) 

            sa_pairs[state, action] += 1  
            
            player_total, _, _ = state
            
            # Only process non-trivial states for updates
            if self.data.should_update_Value(player_total):
                current_n = self.data._N[state][action]
                self.data.set_N_value(state, action, current_n + 1)
                
                # Only retrieve the value if it was actually updated
                n_value = self.data._N[state][action]
                alpha = 1 / max(1, n_value)
                
                # With 0.5 probability
                if random.random() < 0.5:

                    if done:
                        target = reward
                    else:
                        best_action = max(['hit', 'stand'], key=lambda a: self.data._Q[next_state][a])
                        target = reward + self.gamma * self.data._Q2[next_state][best_action]
                    new_q_value = alpha * (target - self.data._Q[state][action])
                    self.data.set_Q_value(state, action, new_q_value)

                else:

                    if done:
                        target = reward
                    else:
                        best_action = max(['hit', 'stand'], key=lambda a: self.data._Q2[next_state][a])
                        target = reward + self.gamma * self.data._Q[next_state][best_action]

                    new_q_value = alpha * (target - self.data._Q2[state][action])
                    self.data.set_Q2_value(state, action, new_q_value)
                    
            if done:
                break
            state = next_state # S = S'
        
        return result
    
    # Merge the two Q-value dictionaries into a single Q-table 
    def _merge_double_q_values(self) -> defaultdict:
        merged_Q = defaultdict(dict)
        
        all_states = set(self.data._Q.keys()) | set(self.data._Q2.keys())
        for state in all_states:
            actions = set(self.data._Q[state].keys()) | set(self.data._Q2[state].keys())
            for action in actions:
                q1_val = self.data._Q[state].get(action, 0.0)
                q2_val = self.data._Q2[state].get(action, 0.0)
                merged_Q[state][action] = 0.5 * (q1_val + q2_val)
        return merged_Q

    def run_rl_agent(self, mode: str, epsfunc, num_episodes = 100000, is_exploring_start = False) -> RLResult:
        """
        Returns - Q table/s, win map every 1000, sa_pair counts, num of unique sa pairs
        """
        self.reset()
        current_thousand_stats = defaultdict(int)
        all_results = []
        sa_pairs:Counter = Counter()
        
        for _ in range(num_episodes):

            match mode:
                case 'montecarlo':
                    episode = self._generate_episode(epsfunc, is_exploring_start)
                    for state, action, _, _ in episode:
                        sa_pairs[state, action] += 1
                    result = self.first_visit_MC(episode)
                case 'sarsa':
                    result = self.sarsa(epsfunc, sa_pairs)
                case 'qlearning':
                    result = self.q_learning(epsfunc, sa_pairs)
                case 'doubleq':
                    result = self.double_q_learning(epsfunc, sa_pairs)
                case _:
                    raise ValueError(f"Unknown mode '{mode}'. Expected one of: montecarlo, doubleq, sarsa, qlearning.")
            
            current_thousand_stats[result] += 1
            
            if self.data.k % 1000 == 0:
                all_results.append(dict(current_thousand_stats))
                all_results[-1]['episode'] = self.data.k
                current_thousand_stats.clear()

            self.data.k += 1

        # Return Q estimates depending on algorithm type
        if mode == 'doubleq':
            merged_Q = self._merge_double_q_values()
            return RLResult(merged_Q, self.data._Q2, all_results, sa_pairs, self.data.count_Q_entries())
        else:
            return RLResult(self.data._Q, None, all_results, sa_pairs, self.data.count_Q_entries())