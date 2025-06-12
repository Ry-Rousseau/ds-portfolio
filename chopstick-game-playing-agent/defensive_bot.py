import random

# check if a move is legal
def is_legal(tapping_hand, tapped_hand, state):
    if tapping_hand == tapped_hand: return False # must tap a different hand
    if state[tapping_hand] == 0: return False # tapping hand cannot be busted
    if state[tapped_hand] == 0 and state[tapping_hand] < 4: return False # split only allowed for hands with >=4 fingers
    return True
    
# the defensive_bot makes splits whenever possible, and otherwise play randomly
def generate_move(state_str):
    state = {'A': int(state_str[1]), 'B': int(state_str[3]), 'C': int(state_str[5]), 'D': int(state_str[7])}
    
    # when split is possible, do the split
    if state['A'] == 0 and state['B'] >= 4: return 'BA'
    if state['B'] == 0 and state['A'] >= 4: return 'AB'
 
    # no split possible, just pick a random legal move
    while True:
        tapping_hand = random.choice(['A', 'B'])
        tapped_hand = random.choice(['A', 'B', 'C', 'D'])
        if is_legal(tapping_hand, tapped_hand, state): break
    move = tapping_hand + tapped_hand
    return move
    