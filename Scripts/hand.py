class Hand:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.cards = []

    def add_card(self, card:str):
        self.cards.append(card)
    
    def is_bust(self) -> bool:
        total, _ = self.calc_total()
        return total > 21

    def calc_total(self) -> tuple[int, bool]:
        """Returns (total_value, has_usable_ace)"""
        total = 0
        ace_count = 0
        for card in self.cards:
            if card == 'A':
                ace_count += 1
                total += 11
            else:
                total += int(card)

        while total > 21 and ace_count:
            total -= 10
            ace_count -= 1
        
        # A usable ace exists if at least one ace remains valued at 11
        usable_ace = ace_count > 0
        return total, usable_ace
        
    def __repr__(self):
        return f"{self.cards} - {self.calc_total()}"
