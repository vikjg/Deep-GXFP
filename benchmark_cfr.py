""" 
P0 thresholds
x1 = 0.059
x2 = 0.511
x3 = 0.526
x4 = x5 - 0.029
x5 = arbitrary
x6 = 0.85
x7 = x8 - 0.061
x8 = arbitrary
{bet-fold < check-fold < check-raise < check-call <
bet-fold < check-call < bet-call < check-raise < bet-call}


P1 thresholds:
facing check: 
y1 = 0.14
y2 = 0.719
y3 = 0.79
{bet-fold < check <
bet-fold < bet-call}

facing bet:
y1 = 0.5
y2 = 0.526
y3 = 0.894
{fold < raise < fold < call}

"""

import numpy as np
import random
from collections import defaultdict
from natsort import natsorted
import matplotlib.pylab as plt

class Node:
    def __init__(self, bet_options):
        self.num_actions = len(bet_options)
        self.regret_sum = defaultdict(int)
        self.strategy = defaultdict(int)
        self.strategy_sum = defaultdict(int)
        self.bet_options = bet_options

    def get_strategy(self):
        normalizing_sum = 0
        for a in self.bet_options:
            if self.regret_sum[a] > 0:
                self.strategy[a] = self.regret_sum[a]
            else:
                self.strategy[a] = 0
            normalizing_sum += self.strategy[a]

        for a in self.bet_options:
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0/self.num_actions

        return self.strategy
    
    def get_uniform_strategy(self):
        for a in self.bet_options:
            self.strategy[a] = 1.0/self.num_actions
        return self.strategy

    def get_average_strategy(self):
        avg_strategy = defaultdict(int)
        normalizing_sum = 0
        
        for a in self.bet_options:
            normalizing_sum += self.strategy_sum[a]
        for a in self.bet_options:
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / self.num_actions
        
        return avg_strategy

class benchmarkCFR:
    def __init__(self, iterations, decksize, starting_stack):
        self.iterations = iterations
        self.decksize = decksize
        self.bet_options = 2
        self.cards = np.arange(1, decksize + 1)
        self.nodes = {}

    def cfr_iterations_external(self):
        util = np.zeros(2)
        recent_util = np.zeros(2)
        final_strategy = {}
        for t in range(1, self.iterations + 1): 
            for i in range(2):
                random.shuffle(self.cards)
                utility = self.external_cfr(self.cards[:2], [], 1, 0, i, t)
                util[i] += utility
                if t > (self.iterations * 0.9):
                    recent_util[i] += utility
            
        
        print('Average game value: {}'.format(util[0]/(self.iterations)))
        print(recent_util)
        print(util)
        for i in natsorted(self.nodes):
            print(i, self.nodes[i].get_average_strategy())
            final_strategy[i] =  self.nodes[i].get_average_strategy()
        return final_strategy

    def valid_bets(self, history, acting_player):
        
        #print('VALID BETS CHECK HISTORY', history)
        #print('VALID BETS CHECK ROUND', rd)
        #print('VALID BETS CHECK ACTING STACK', acting_stack)
        curr_history = history
        
        
        if len(history) == 0:
            return [0, 1]
        
        elif len(history) == 1:
            if history[0] == 0:
                return [0, 1]
            elif history[0] == 1: 
                return [0, 1, 2]
        
        elif len(history) == 2:
            call_amount = curr_history[1] - curr_history[0]
            return [0, call_amount]
        
    def winning_hand(self, cards):
        if cards[0] > cards[1]:
            return 0 
        else:
            return 1 

    def external_cfr(self, cards, history, pot, nodes_touched, traversing_player, t):
        if t % 10000 == 0 and t>0:
            print('THIS IS ITERATION', t)
        plays = len(history)
        acting_player = plays % 2
        # print('*************')
        # print('HISTORY RD', history[rd])
        # print('PLAYS', plays)

        if plays >= 2:
            p0total = np.sum(history[0::2])
            p1total = np.sum(history[1::2])
            # print('P0 TOTAL', p0total)
            # print('P1 TOTAL', p1total)
            # print('ROUND BEG', rd)
                
            if p0total == p1total:
                winner = self.winning_hand(cards)
                if traversing_player == winner:
                    return pot/2
                elif traversing_player != winner:
                    return -pot/2

            elif history[-1] == 0: #previous player folded
                # print('FOLD RETURN')
                if acting_player == 0 and acting_player == traversing_player:
                    return p1total+0.5
                elif acting_player == 0 and acting_player != traversing_player:
                    return -(p1total +0.5)
                elif acting_player == 1 and acting_player == traversing_player:
                    return p0total+0.5
                elif acting_player == 1 and acting_player != traversing_player:
                    return -(p0total +0.5)
        # print('ROUND AFTER', rd)
        infoset = str(cards[acting_player]) + str(history)

        if acting_player == 0:
            infoset_bets = self.valid_bets(history, 0)
        elif acting_player == 1:
            infoset_bets = self.valid_bets(history, 1)
        #print('INFOSET BETS', infoset_bets)
        if infoset not in self.nodes:
            self.nodes[infoset] = Node(infoset_bets)

        #print(self.nodes[infoset])
        #print(infoset)

        nodes_touched += 1

        if acting_player == traversing_player:
            util = defaultdict(int)
            node_util = 0
            strategy = self.nodes[infoset].get_strategy()
            for a in infoset_bets:
                next_history = history + [a]
                pot += a
                util[a] = self.external_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
                node_util += strategy[a] * util[a]

            for a in infoset_bets:
                regret = util[a] - node_util
                self.nodes[infoset].regret_sum[a] += regret
            return node_util

        else: #acting_player != traversing_player
            strategy = self.nodes[infoset].get_strategy()
            #strategy = self.nodes[infoset].get_uniform_strategy()
            #print('STRATEGY', strategy)
            dart = random.random()
            #print('DART', dart)
            strat_sum = 0
            for a in strategy:
                strat_sum += strategy[a]
                if dart < strat_sum:
                    action = a
                    break
            #print('ACTION', action)
            next_history = history + [action]
            pot += action
            # if acting_player == 0:
            #     p0stack -= action
            # elif acting_player == 1:
            #     p1stack -= action
            # print('NEXT HISTORY2', next_history)
            util = self.external_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
            for a in infoset_bets:
                self.nodes[infoset].strategy_sum[a] += strategy[a]
            return util

if __name__ == "__main__":
    k = benchmarkCFR(1000000, 100, 20)
    final_strategy = k.cfr_iterations_external()
    
    p0_init = {}
    p1_face_bet = {}
    p1_face_check = {}
    p0_face_raise = {}
    
    for keys in final_strategy.keys():
        if '[1]' in keys:
            p1_face_bet[keys.replace('[1]', '')] = final_strategy[keys]
        elif '[]' in keys:
            p0_init[keys.replace('[]', '')] = final_strategy[keys]
        elif '[0]' in keys:
            p1_face_check[keys.replace('[0]', '')] = final_strategy[keys]
        else:
            p0_face_raise[keys.replace('[0, 1]', '')] = final_strategy[keys]
    
# =============================================================================
#     lists = natsorted(p0_init.items()) # sorted by key, return a list of tuples
#     
#     x, y = zip(*lists) # unpack a list of pairs into two tuples
#     plt.xticks([])
#     plt.plot(x, y)
#     plt.show()
#     
#     lists1 = natsorted(p1_face_bet.items()) # sorted by key, return a list of tuples
# 
#     x1, y1 = zip(*lists1) # unpack a list of pairs into two tuples
#     plt.xticks([])
#     plt.plot(x1, y1)
#     plt.show()
# =============================================================================
