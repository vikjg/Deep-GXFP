"""

x1 = 0.11 bluff
x2 = 0.77 value bet
y1 = 0.55 call/fold bound

"""


import numpy as np
import random
from natsort import natsorted
import matplotlib.pylab as plt

class Node:
    def __init__(self, num_actions):
        self.regret_sum = np.zeros(num_actions)
        self.strategy = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.num_actions = num_actions

    def get_strategy(self):
        normalizing_sum = 0
        for a in range(self.num_actions):
            if self.regret_sum[a] > 0:
                self.strategy[a] = self.regret_sum[a]
            else:
                self.strategy[a] = 0
            normalizing_sum += self.strategy[a]

        for a in range(self.num_actions):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0/self.num_actions

        return self.strategy

    def get_average_strategy(self):
        avg_strategy = np.zeros(self.num_actions)
        normalizing_sum = 0
        
        for a in range(self.num_actions):
            normalizing_sum += self.strategy_sum[a]
        for a in range(self.num_actions):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / self.num_actions
        
        return avg_strategy
    
    def get_uniform_strategy(self):
        for a in self.bet_options:
            self.strategy[a] = 1.0/self.num_actions
        return self.strategy

class vNMCFR:
    def __init__(self, iterations, decksize):
        self.iterations = iterations
        self.decksize = decksize
        self.cards = np.arange(1, decksize + 1)
        self.bet_options = 2
        self.nodes = {}

    def cfr_iterations_external(self):
        util = np.zeros(2)
        final_strategy = {}
        for t in range(1, self.iterations + 1): 
            for i in range(2):
                random.shuffle(self.cards)
                util[i] += self.external_cfr(self.cards[:2], [], 1, 0, i, t)
                print(i, util[i])
        print('Average game value: {}'.format(util[0]/(self.iterations)))
        for i in natsorted(self.nodes):
            print(i, self.nodes[i].get_average_strategy())
            final_strategy[i] =  self.nodes[i].get_average_strategy()
        return final_strategy

    def external_cfr(self, cards, history, pot, nodes_touched, traversing_player, t):
        print('THIS IS ITERATION', t)
        print(cards, history, pot)
        plays = len(history)
        acting_player = plays % 2
        opponent_player = 1 - acting_player
        if plays == 1:
            if (history[-1] == 0): #check
                if acting_player == traversing_player:
                    if cards[acting_player] > cards[opponent_player]:
                        return pot/2 #profit
                    else:
                        return -pot/2
                else:
                    if cards[acting_player] > cards[opponent_player]:
                        return -pot/2
                    else:
                        return pot/2
        if plays == 2:
            if history[-1] == 0: #bet fold
                if acting_player == traversing_player:
                    return 0.5
                else:
                    return -0.5
            if (history[-1] == 1 and history[-2] == 1): #bet call
                if acting_player == traversing_player:
                    if cards[acting_player] > cards[opponent_player]:
                        return pot/2 #profit
                    else:
                        return -pot/2
                else:
                    if cards[acting_player] > cards[opponent_player]:
                        return -pot/2
                    else:
                        return pot/2

        infoset = str(cards[acting_player]) + str(history)
        if infoset not in self.nodes:
            self.nodes[infoset] = Node(self.bet_options)

        nodes_touched += 1

        if acting_player == traversing_player:
            util = np.zeros(self.bet_options) #2 actions
            node_util = 0
            strategy = self.nodes[infoset].get_strategy()
            for a in range(self.bet_options):
                next_history = history + [a]
                pot += a
                util[a] = self.external_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
                node_util += strategy[a] * util[a]

            for a in range(self.bet_options):
                regret = util[a] - node_util
                self.nodes[infoset].regret_sum[a] += regret
            return node_util

        else: #acting_player != traversing_player
            strategy = self.nodes[infoset].get_strategy()
            util = 0
            if random.random() < strategy[0]:
                next_history = history + [0]
            else: 
                next_history = history + [1]
                pot += 1
            util = self.external_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
            for a in range(self.bet_options):
                self.nodes[infoset].strategy_sum[a] += strategy[a]
            return util

if __name__ == "__main__":
    k = vNMCFR(5000000, 100)
    final_strategy = k.cfr_iterations_external()
    
    p0_strategy = {}
    p1_strategy = {}
    
    for keys in final_strategy.keys():
        if '[1]' in keys:
            p1_strategy[keys.replace('[1]', '')] = final_strategy[keys]
        else:
            p0_strategy[keys.replace('[]', '')] = final_strategy[keys]
    
    lists = natsorted(p0_strategy.items()) # sorted by key, return a list of tuples
    
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.xticks([])
    plt.plot(x, y)
    plt.show()
    
    lists1 = natsorted(p1_strategy.items()) # sorted by key, return a list of tuples

    x1, y1 = zip(*lists1) # unpack a list of pairs into two tuples
    plt.xticks([])
    plt.plot(x1, y1)
    plt.show()
    
    
    