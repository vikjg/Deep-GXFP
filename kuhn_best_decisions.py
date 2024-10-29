import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import doctest


class KuhnTrainer:
  def __init__(self, train_iterations=10**4):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = 2
    self.PASS = 0
    self.BET = 1
    self.NUM_ACTIONS = 2
    self.avg_strategy = {}
    self.rank = {"J":1, "Q":2, "K":3}


  def Return_payoff_for_terminal_states(self, history, target_player_i):
    plays = len(history)
    player = plays % 2
    opponent = 1 - player
    terminal_utility = 0

    if plays > 3:
      terminalPass = (history[plays-1] == "p")
      doubleBet = (history[plays-2 : plays] == "bb")
      isPlayerCardHigher = (self.rank[history[player]] > self.rank[history[opponent]])

      if terminalPass:
        if history[-2:] == "pp":
          if isPlayerCardHigher:
            terminal_utility = 1
          else:
            terminal_utility = -1
        else:
            terminal_utility = 1
      elif doubleBet: 
          if isPlayerCardHigher:
            terminal_utility = 2
          else:
            terminal_utility = -2

    if player == target_player_i:
      return terminal_utility
    else:
      return -terminal_utility



  def whether_terminal_states(self, history):
    plays = len(history)
    if plays > 3:
      terminalPass = (history[plays-1] == "p")
      doubleBet = (history[plays-2 : plays] == "bb")

      return terminalPass or doubleBet
    else:
      return False


  def whether_chance_node(self, history):
    if history == "":
      return True
    else:
      return False



  def if_nonexistant(self, infoSet):
    if infoSet not in self.avg_strategy:
      self.avg_strategy[infoSet] = np.array([1/self.NUM_ACTIONS for _ in range(self.NUM_ACTIONS)], dtype=float)


  def show_plot(self, method):
    plt.scatter(list(self.exploitability_list.keys()), list(self.exploitability_list.values()), label=method)
    plt.plot(list(self.exploitability_list.keys()), list(self.exploitability_list.values()))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("iterations")
    plt.ylabel("exploitability")
    plt.legend()


  def calc_best_response_value(self, avg_strategy, best_response_strategy, best_response_player, history, prob):
      plays = len(history)
      player = plays % 2
      opponent = 1 - player

      if self.whether_terminal_states(history):
        return self.Return_payoff_for_terminal_states(history, best_response_player)

      elif self.whether_chance_node(history):
        cards = np.array(["J", "Q", "K"])
        cards_candidates = [cards_candidate for cards_candidate in itertools.permutations(cards)]
        utility_sum = 0
        for cards_i in cards_candidates:
          nextHistory = cards_i[0] + cards_i[1]
          utility_sum +=  (1/len(cards_candidates))* self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, prob)
        return utility_sum


      infoSet = history[player] + history[2:]
      self.if_nonexistant(infoSet)


      if player == best_response_player:
        if infoSet not in best_response_strategy:
          action_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          br_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)


          for assume_history, po_ in self.infoSets_dict[infoSet]:
            for ai in range(self.NUM_ACTIONS):
              nextHistory =  assume_history + ("p" if ai == 0 else "b")
              br_value[ai] = self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, po_)
              action_value[ai] += br_value[ai] * po_


          br_action = 0
          for ai in range(self.NUM_ACTIONS):
            if action_value[ai] > action_value[br_action]:
              br_action = ai
          best_response_strategy[infoSet] = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          best_response_strategy[infoSet][br_action] = 1.0


        node_util = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in range(self.NUM_ACTIONS):
          nextHistory =  history + ("p" if ai == 0 else "b")
          node_util[ai] = self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, prob)
        best_response_util = 0
        best_decision = np.argmax(node_util)
        print('best decision', best_decision)
        for ai in range(self.NUM_ACTIONS):
          best_response_util += node_util[ai] * best_response_strategy[infoSet][ai]

        return best_response_util

      else:
        nodeUtil = 0
        action_value_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in range(self.NUM_ACTIONS):
          nextHistory =  history + ("p" if ai == 0 else "b")
          action_value_list[ai] = self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, prob*avg_strategy[infoSet][ai])
          nodeUtil += avg_strategy[infoSet][ai] * action_value_list[ai]
        return nodeUtil


  def create_infoSets(self, history, target_player, po):
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    if self.whether_terminal_states(history):
      return

    elif self.whether_chance_node(history):
      cards = np.array(["J", "Q", "K"])
      cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards)]
      for cards_i in cards_candicates:
        nextHistory = cards_i[0] + cards_i[1]
        self.create_infoSets(nextHistory, target_player, po)
      return

    infoSet = history[player] + history[2:]
    if player == target_player:
      if self.infoSets_dict.get(infoSet) is None:
        self.infoSets_dict[infoSet] = []
      self.infoSets_dict[infoSet].append((history, po))

    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")
      if player == target_player:
        self.create_infoSets(nextHistory, target_player, po)
      else:
        self.if_nonexistant(infoSet)
        actionProb = self.avg_strategy[infoSet][ai]
        self.create_infoSets(nextHistory, target_player, po*actionProb)


  def update_avg_strategy(self, pi_strategy, br_strategy, iteration_t, lambda_num):
    alpha_t1 = 1 / (iteration_t+2)

    if lambda_num == 1:
        for information_set_u in self.infoSets_dict.keys():
          pi_strategy[information_set_u] +=  alpha_t1*(br_strategy[information_set_u] - pi_strategy[information_set_u])

    elif lambda_num == 2:
      for information_set_u in self.infoSets_dict.keys():
        x_br = self.calculate_realization_plan(information_set_u, br_strategy, 0)
        x_pi = self.calculate_realization_plan(information_set_u, pi_strategy, 1)
        pi_strategy[information_set_u] +=  (alpha_t1*x_br*(br_strategy[information_set_u] - pi_strategy[information_set_u])) / ( (1-alpha_t1)*x_pi + alpha_t1*x_br)



  def calculate_realization_plan(self, information_set_u, target_strategy, bit):
    if len(information_set_u) <= 2:
      return 1
    else:
      hi_a = information_set_u[0]
      if bit == 0:
        return target_strategy[hi_a][0]
      else:
        return target_strategy[hi_a][0]


  def get_exploitability_dfs(self):


    self.infoSets_dict = {}
    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)

    exploitability = 0
    best_response_strategy = {}
    for best_response_player_i in range(self.NUM_PLAYERS):
        exploitability += self.calc_best_response_value(self.avg_strategy, best_response_strategy, best_response_player_i, "", 1)


    return exploitability


  def eval_vanilla_CFR(self, history, target_player_i, iteration_t, p0, p1):
      plays = len(history)
      player = plays % 2
      opponent = 1 - player

      if self.whether_terminal_states(history):
        return self.Return_payoff_for_terminal_states(history, target_player_i)

      elif self.whether_chance_node(history):
        cards = np.array(["J", "Q", "K"])
        cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards)]
        utility_sum = 0
        for cards_i in cards_candicates:
          nextHistory = cards_i[0] + cards_i[1]
          utility_sum += (1/len(cards_candicates))* self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p0, p1)
        return utility_sum

      infoSet = history[player] + history[2:]
      self.if_nonexistant(infoSet)

      strategy = self.avg_strategy[infoSet]

      util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
      nodeUtil = 0

      for ai in range(self.NUM_ACTIONS):
        nextHistory = history + ("p" if ai == 0 else "b")
        if player == 0:
          util_list[ai] = self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p0 * strategy[ai], p1)
        else:
          util_list[ai] = self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p0, p1 * strategy[ai])
        nodeUtil += strategy[ai] * util_list[ai]

      return nodeUtil



  def train(self, lambda_num):
    self.exploitability_list = {}

    for iteration_t in tqdm(range(int(self.train_iterations))):
      best_response_strategy = {}

      self.infoSets_dict = {}
      for target_player in range(self.NUM_PLAYERS):
        self.create_infoSets("", target_player, 1.0)

      for best_response_player_i in range(self.NUM_PLAYERS):
        self.calc_best_response_value(self.avg_strategy, best_response_strategy, best_response_player_i, "", 1)

    
      self.update_avg_strategy(self.avg_strategy, best_response_strategy, iteration_t, lambda_num)


      if iteration_t in [int(j)-1 for j in np.logspace(1, len(str(self.train_iterations))-1, (len(str(self.train_iterations))-1)*3)] :
        self.exploitability_list[iteration_t] = self.get_exploitability_dfs()

    self.show_plot("XFP_{}".format(lambda_num))

    
#config
iterations = 1
lambda_num = 1


#train
kuhn_trainer = KuhnTrainer(train_iterations=iterations)
kuhn_trainer.train(lambda_num = lambda_num)


#result
print("avg util:", kuhn_trainer.eval_vanilla_CFR("", 0, 0, 1, 1))

result_dict = {}

for key, value in sorted(kuhn_trainer.avg_strategy.items()):
  result_dict[key] = value

df = pd.DataFrame(result_dict.values(), index=result_dict.keys(), columns=['Pass', "Bet"])
df = df.reindex(["J", "Jp", "Jb", "Jpb", "Q", "Qp", "Qb", "Qpb", "K", "Kp", "Kb", "Kpb"], axis="index")
df.index.name = "Node"

print(df)