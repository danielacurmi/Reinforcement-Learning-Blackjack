from deck import Deck 
from hand import Hand

class Blackjack:
    def __init__(self):
        self.deck = Deck()
        self.player = Hand()
        self.dealer = Hand()
        self.terminal = False
        self.result = None  # 'win', 'loss', 'draw'

    def reset(self):
        self.deck.reset()
        self.player.reset()
        self.dealer.reset()
        self.terminal = False
        self.result = None

    def game_setup(self):
        self.player.add_card(self.deck.draw_card())
        self.player.add_card(self.deck.draw_card())

        self.dealer.add_card(self.deck.draw_card())
        self.dealer.add_card(self.deck.draw_card())

        return self._get_state()

    # return: (next_state, reward, done, result)
    def step(self, action):
        if action == 'hit':
            self.player.add_card(self.deck.draw_card())
            if self.player.is_bust():
                self.terminal = True
                self.result = 'loss'
                return self._get_state(), -1, True, self.result
            return self._get_state(), 0, False, 'Game In Progress'
        
        elif action == 'stand':
            while True:
                dealer_total, _ = self.dealer.calc_total()
                if dealer_total >= 17: break
                self.dealer.add_card(self.deck.draw_card())

            self.terminal = True
            return self._evaluate_game()
        
        raise ValueError(f"Invalid action '{action}'. Expected 'hit' or 'stand'.")
        
    def _get_state(self):
        player_total, usable_ace = self.player.calc_total()
        dealer_card = self.dealer.cards[0]
        if dealer_card == 'A':
            dealer_card = '11'  # Map Ace to 11 for dealer's visible card
        return player_total, dealer_card, usable_ace
    
    def _evaluate_game(self):
        player_total, _ = self.player.calc_total()
        dealer_total, _ = self.dealer.calc_total()

        if dealer_total > 21 or player_total > dealer_total:
            self.result = 'win'
            reward = 1
        elif dealer_total < player_total:
            self.result = 'win'
            reward = 1
        elif dealer_total > player_total:
            self.result = 'loss'
            reward = -1
        else:
            self.result = 'draw'
            reward = 0

        return self._get_state(), reward, True, self.result

    def get_dealer_firstCard(self) -> str:
        return self.dealer.cards[0]
