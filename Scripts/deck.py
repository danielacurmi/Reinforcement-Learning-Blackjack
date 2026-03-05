import random

class Deck:
    def __init__(self):
        self.reset()
        
    def _gen_deck(self) -> list[str]:
        #J Q K are all 10
        cards = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', '10', '10', '10']
        return cards * 4
    
    def shuffle(self) -> None:
        random.shuffle(self.deck)
    
    def draw_card(self) -> str:
        return self.deck.pop(0)
    
    def reset(self):
        self.deck = self._gen_deck()
        self.shuffle()

    def __repr__(self):
        return f'{self.deck}'