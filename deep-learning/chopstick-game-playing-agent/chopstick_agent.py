# This file implements an AI player for a modified Chopsticks game using minimax search with alpha-beta pruning. The Chopsticks class handles game logic including legal move generation, state transitions, and terminal/utility calculations for a 4-hand variant (players A,B vs C,D). The core algorithm uses depth-limited search (default depth 11) with a heuristic evaluation function that considers hand advantages, dead hands, and tactical positions. The generate_move function serves as the main interface, taking a board state string and returning the AI's optimal move, achieving 70th percentile performance in competition.

from games4e import * # pip install games4e
import math

class Chopsticks(Game):
    def __init__(self, board):
        if type(board) == str: #converts state to dictionary if not already
            self.board = self.convert_state_to_dictionary(board)
        else:
            self.board = board
        moves = ['AB', 'AC', 'AD', 'BA', 'BC', 'BD', 'CD', 'CB', 'CA', 'DC', 'DB', 'DA']
        #assumes is is our turn
        self.initial = GameState(to_move=True, utility=0, board=self.board, moves=moves)

    def convert_state_to_dictionary(self, board_str):
        board = {'A': int(board_str[1]), 'B': int(board_str[3]), 'C': int(board_str[5]), 'D': int(board_str[7])}
        return board
    
    def actions(self, state):
        board = state.board
        to_move = state.to_move
        l_moves = []
        if to_move:
            for move in ['AB', 'AC', 'AD', 'BA', 'BC', 'BD']:
                tapping_hand = move[0]
                tapped_hand = move[1]
                if board[tapping_hand] == 0:
                    continue
                elif board[tapped_hand] == 0 and board[tapping_hand] >= 4 and move in ['AB', 'BA']:
                    l_moves.append(move)
                elif board[tapped_hand] == 0:
                    continue
                else:
                    if move not in ['AB', 'BA']:
                        l_moves.append(move)
        else:
            for move in ['CD', 'CB', 'CA', 'DC', 'DB', 'DA']:
                tapping_hand = move[0]
                tapped_hand = move[1]
                if board[tapping_hand] == 0:
                    continue
                elif board[tapped_hand] == 0 and board[tapping_hand] >= 4 and move in ['CD', 'DC']:
                    l_moves.append(move)
                elif board[tapped_hand] == 0:
                    continue
                else:
                    if move not in ['CD', 'DC']:
                        l_moves.append(move)
        return l_moves

    def result(self, state, move):
        ''' Returns the state that results after making a move'''
        if move not in self.actions(state):
            return state
        tapping_hand = move[0]
        tapped_hand = move[1]
        new_board = state.board.copy()
        if state.board[tapped_hand] == 0: #checks if move is split
            new_board[tapped_hand] = math.floor(state.board[tapping_hand]/2)
            new_board[tapping_hand] = math.ceil(state.board[tapping_hand]/2)
        else: 
            if state.board[tapped_hand] + state.board[tapping_hand] > 5:
                new_board[tapped_hand] = 0
            else:
                new_board[tapped_hand] = state.board[tapping_hand] + state.board[tapped_hand]
        moves = state.moves
        return GameState(to_move=not state.to_move, 
                         utility=self.compute_utility(new_board, move, state.to_move),
                           board=new_board, moves=moves)

    def utility(self, state, to_move):
        ''' Returns the utility of the current state'''
        return state.utility if to_move else -state.utility
    
    def terminal_test(self, state):
        """A state is terminal if it is won
        Returns a boolean value
        """
        if (state.board['C'] == 0 and state.board['D'] == 0) or (state.board['A'] == 0 and state.board['B'] == 0):
            return True
        return len(state.moves) == 0

    def next_board(self, board, move):
        ''' Returns the board that results after making a move'''
        tapping_hand = move[0]
        tapped_hand = move[1]
        new_board = board.copy()
        if board[tapped_hand] == 0: #checks if move is split
            new_board[tapped_hand] = math.floor(board[tapping_hand]/2)
            new_board[tapping_hand] = math.ceil(board[tapping_hand]/2)
        else: 
            if board[tapped_hand] + board[tapping_hand] > 5:
                new_board[tapped_hand] = 0
            else:
                new_board[tapped_hand] = board[tapping_hand] + board[tapped_hand]        
        return new_board

    def compute_utility(self,board,move,to_move):
        """If 'X' wins with this move, return INF; if 'O' wins return -INF; 
        else return 0."""
        next_board = self.next_board(board,move)
        to_move = not to_move
        if to_move and (next_board['C'] == 0 and next_board['D'] == 0):
            return math.inf
            #else if terminal and winning for opponent, return negative Inf
        elif not to_move and (next_board['A'] == 0 and next_board['B'] == 0):
            return -math.inf
        else:
            return 0  
        #possibly add route for if only one option in game state    

    def estimate_utility(self, state):
        ''' Returns the estimated utility of the state which has no children and is not in terminal state 
        using a heuristic function
        '''
        h = 0
        if self.terminal_test(state):
            if (state.board['C'] == 0 and state.board['D'] == 0):
                return math.inf
            else: 
                return -math.inf 
        if state.board['C'] >= 2 and state.board['D'] >= 2 and (state.board['A'] == 0 or state.board['B'] == 0): #situation where opponent has 2 hands versus our 1 hand
            h -= 10
        elif (state.board['C'] == 0 or state.board['D'] == 0) and state.board['A'] >= 2 and state.board['B'] >= 2: #situation where opponent has 1 hands versus our 2 hands
            h += 10
        if state.board['C'] == 0 or state.board['D'] == 0: #positive value for opponents hands being dead
            h += 3
        if state.board['A'] == 0 or state.board['B'] == 0: #negative value for our hands being dead 
            h -= 3
        #situation where at least 2 alive hands, with one hand with 5 fingers and our turn - positive weight    
        if (state.board['A'] == 5 or state.board['B'] == 5) and state.to_move and state.board['A'] >= 2 and state.board['B'] >= 2: 
            h += 10
        #situation where at least 2 alive hands, with one hand with 5 fingers and opponents turn - negative weight
        elif (state.board['C'] == 5 or state.board['D'] == 5) and not state.to_move and state.board['C'] >= 2 and state.board['D'] >= 2:
            h -= 10
        return h
 
def alpha_beta_cutoff_search(state, game, d=12, cutoff_test=None, eval_fn= Chopsticks.estimate_utility):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -math.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = math.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # depth 1 search to tell if next state is winning for each legal move
    for a in game.actions(state):
        next_state = game.result(state,a)
        if next_state.board['C'] == 0 and next_state.board['D'] == 0:
            return a
    #Body of alpha_beta_cutoff_search starts here:
    #The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or (lambda state, depth: depth > d or game.terminal_test(state)))
    eval_fn = eval_fn
    best_score = -math.inf
    beta = math.inf
    best_action = game.actions(state)[0] #default to only available action if in losing state
    for a in game.actions(state):
        v = max_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action

def generate_move(str_board):
    f_game = Chopsticks(str_board)
    return str(alpha_beta_cutoff_search(f_game.initial, f_game, d=11, cutoff_test=None, eval_fn=f_game.estimate_utility))