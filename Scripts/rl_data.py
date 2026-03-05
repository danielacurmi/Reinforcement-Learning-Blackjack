from collections import defaultdict

class RLData:
    def __init__(self):
        self.TRIVIAL_PLAYER_TOTAL = 11
        self.reset()

    def reset(self):
        self._Q = defaultdict(lambda: {'hit': 0.0, 'stand': 0.0})
        self._Q2 = defaultdict(lambda: {'hit': 0.0, 'stand': 0.0})
        self._N = defaultdict(lambda: {'hit': 0, 'stand': 0})
        self.k = 1  # episode counter

    def should_update_Value(self, player_total) -> bool:
        if player_total <= self.TRIVIAL_PLAYER_TOTAL: return False
        if player_total >= 21: return False
        
        return True
    
    def set_Q_value(self, state, action, value) -> None:
        player_total, _, _ = state
        if not self.should_update_Value(player_total): return

        self._Q[state][action] += value

    def set_Q2_value(self, state, action, value) -> None:
        player_total, _, _ = state
        if not self.should_update_Value(player_total): return

        self._Q2[state][action] += value

    def set_N_value(self, state, action, value) -> None:
        player_total, _, _ = state
        if not self.should_update_Value(player_total): return

        self._N[state][action] = value

    def count_Q_entries(self) -> int:
        count = 0
        for state, actions in self._Q.items():
            player_total, _, _ = state
            if not self.should_update_Value(player_total):continue
            
            for action, value in actions.items():
                count += 1  # Count every state-action pair, regardless of value
        return count
