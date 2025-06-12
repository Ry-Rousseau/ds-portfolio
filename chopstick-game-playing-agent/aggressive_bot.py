# check if a move is attacing the opponent
def is_attacking(tapping_hand, tapped_hand, state):
    if state[tapping_hand] == 0: return False # tapping hand needs to be unbusted
    if state[tapped_hand] not in ['C', 'D']: return False # an attacking move needs to tap an opponent hand
    return True
    
# the aggressive_bot generates a move that makes opponent's two hands as close to both busted as possible
def generate_move(state_str):
    state = {'A': int(state_str[1]), 'B': int(state_str[3]), 'C': int(state_str[5]), 'D': int(state_str[7])}
    possible_moves = ['AB', 'AC', 'AD', 'BA', 'BC', 'BD']
    h = {} # h measures the desirability of each move (smaller better)
    
    for move in possible_moves:
        tapping_hand = move[0]
        tapped_hand = move[1]
        if is_attacking(tapping_hand, tapped_hand, state):
            # what would be the state if this move is played
            state_next = state.copy()
            state_next[tapped_hand] += state_next[tapping_hand]
            if state_next[tapped_hand] > 5: state_next[tapped_hand] = 0
            # the h function measures how close the opponent is to having both hands busted
            h[move] = (6 - state_next['C']) * (state_next['C'] != 0) + (6 - state_next['D']) * (state_next['D'] != 0)
        else:
            h[move] = 12 # low priority to non-attacking moves
    # return the move associated with minimum h value
    best_move = min(h, key=h.get)
    return best_move