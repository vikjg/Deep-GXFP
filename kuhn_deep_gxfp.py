import datetime
import time
from pathlib import Path

import numpy as np
import torch
from KuhnNetwork import AdvantageNet
import copy
from kuhn_poker_eng import PokerEnv
from collections import deque
from tqdm import tqdm
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

import os
import pandas as pd
from torchvision.io import read_image

class ActionDataset(Dataset):

    def __init__(self, actions, states, ts):
        self.actions = actions
        self.states = states
        self.ts = ts


    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.actions[idx], get_state_array_from_state(self.states[idx]), self.ts[idx]


class History:
    def __init__(self):
        self.env = PokerEnv([.5,.5],1.5)
        self.env.game.reset_game([.5,.5],1.5)

    def copy(self):
        new_env = PokerEnv([.5,.5], 1.5)
        new_env.game.deck = copy.deepcopy(self.env.game.deck)
        new_env.game.action_finished = copy.deepcopy(self.env.game.action_finished)
        new_env.game.blinds = copy.deepcopy(self.env.game.blinds)
        new_env.game.sb_player = copy.deepcopy(self.env.game.sb_player)
        new_env.game.action_player = copy.deepcopy(self.env.game.action_player)
        new_env.game.phase = copy.deepcopy(self.env.game.phase)
        new_env.game.started_new_phase = copy.deepcopy(self.env.game.started_new_phase)
        new_env.game.all_in = copy.deepcopy(self.env.game.all_in)

        new_env.game.game_over = copy.deepcopy(self.env.game.game_over)
        new_env.game.game_num = copy.deepcopy(self.env.game.game_num)

        new_env.game.p1.pnum = 0
        new_env.game.p2.pnum = 1
        new_env.game.p1.card_1 = copy.deepcopy(self.env.game.p1.card_1)
        new_env.game.p2.card_1 = copy.deepcopy(self.env.game.p2.card_1)

        if isinstance(self.env.game.p1.stack_size, torch.Tensor):
            new_env.game.p1.stack_size = self.env.game.p1.stack_size.clone()
        else:
            new_env.game.p1.stack_size = copy.deepcopy(self.env.game.p1.stack_size)
        if isinstance(self.env.game.p2.stack_size, torch.Tensor):
            new_env.game.p2.stack_size = self.env.game.p2.stack_size.clone()
        else:
            new_env.game.p2.stack_size = copy.deepcopy(self.env.game.p2.stack_size)

        if isinstance(self.env.game.p1.starting_stack_size, torch.Tensor):
            new_env.game.p1.starting_stack_size = self.env.game.p1.starting_stack_size.clone()
        else:
            new_env.game.p1.starting_stack_size = copy.deepcopy(self.env.game.p1.starting_stack_size)
        if isinstance(self.env.game.p2.starting_stack_size, torch.Tensor):
            new_env.game.p2.starting_stack_size = self.env.game.p2.starting_stack_size.clone()
        else:
            new_env.game.p2.starting_stack_size = copy.deepcopy(self.env.game.p2.starting_stack_size)

        if isinstance(self.env.game.p1.last_win, torch.Tensor):
            new_env.game.p1.last_win = self.env.game.p1.last_win.clone()
        else:
            new_env.game.p1.last_win = copy.deepcopy(self.env.game.p1.last_win)
        if isinstance(self.env.game.p2.last_win, torch.Tensor):
            new_env.game.p2.last_win = self.env.game.p2.last_win.clone()
        else:
            new_env.game.p2.last_win = copy.deepcopy(self.env.game.p2.last_win)

        if isinstance(self.env.game.pot_size, torch.Tensor):
            new_env.game.pot_size = self.env.game.pot_size.clone()
        else:
            new_env.game.pot_size = copy.deepcopy(self.env.game.pot_size)

        if isinstance(self.env.game.to_call, torch.Tensor):
            new_env.game.to_call = self.env.game.to_call.clone()
        else:
            new_env.game.to_call = copy.deepcopy(self.env.game.to_call)

        history = History()
        history.env = new_env
        return history

def get_state_array_from_state(state):
    state_array = np.zeros(17*7+ 5)#np.zeros((13 + 4) * 7 + 5)


    # state_array[state[0][0]] = 1
    # state_array[state[0][1] + 13] = 1
    # state_array[state[1][0]] = 1
    # state_array[state[1][1] + 13] = 1
    #
    #
    # for i in range(len(state[2])):
    #     state_array[state[2][i][0] + 17] = 1
    #     state_array[state[2][i][1] + 17 + 13] = 1

    state_array[state[0][0]] = 1
    state_array[state[0][1] + 13] = 1


    state_array[-3] = state[-3] / (state[-4] + state[-6] + state[-5])
    state_array[-2] = state[-2]
    state_array[-1] = state[-1] / 5
    state_array[-4] = state[-4] / (state[-6])
    state_array[-5] = state[-5] / (state[-6])



    state = torch.tensor(state_array)
    return state

def get_state_array(env, action_player, inaction_player):
    state_array = np.zeros(17*7 + 5)

    state = [action_player.card_1, env.game.pot_size,
     action_player.stack_size, inaction_player.stack_size, env.game.to_call, env.action_player.was_last_raiser,
             env.last_action]

    state_array[state[0][0]] = 1
    state_array[state[0][1] + 13] = 1


    # state_array[state[0][0]] = 1
    # state_array[state[0][1] + 13] = 1
    # state_array[state[1][0]] = 1
    # state_array[state[1][1] + 13] = 1
    #
    # for i in range(len(state[2])):
    #     state_array[state[2][i][0] + 17] = 1
    #     state_array[state[2][i][1] + 17 + 13] = 1

    # state_array[-3] = state[-3] / (state[-4] + state[-6] + state[-5])
    # state_array[-2] = state[-2]
    # state_array[-1] = state[-1] / 5
    # state_array[-4] = state[-4] / (state[-6])
    # state_array[-5] = state[-5] / (state[-6])

    state_array[-3] = state[-3] / (state[-6])
    state_array[-2] = state[-2]
    state_array[-1] = state[-1] / 5
    state_array[-4] = state[-4] / (state[-6])
    state_array[-5] = state[-5] / (state[-6])



    state = torch.tensor(state_array)
    return state


def last_hand_test(network, env):
    state_array = get_state_array(env, env.game.p1, env.game.p2)
    #print(state_array)
    if torch.cuda.is_available():
        state_array = state_array.to(device="cuda")
    action = network(state_array, False)
    action = network.sm(action)
    print(f"Last Hand: {action} {env.game.p1.card_1} {env.game.p2.card_1}")


def deep_gxfp(T, N):
    #torch.autograd.set_detect_anomaly(True)
    save_dir1 = Path("strat") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir2 = Path("adv") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir1.mkdir(parents=True)
    save_dir2.mkdir(parents=True)
    adv0 = deque(maxlen=10000000)
    adv1 = deque(maxlen=10000000)

    strat = deque(maxlen=100000000)
    m_bd0 = deque(maxlen=100000000)
    m_bd1 = deque(maxlen=100000000)

    strat_net = AdvantageNet(17*7 + 5, 512).float()#AdvantageNet((13 + 4) * 7 + 5, 128).float()

    t0 = AdvantageNet(17*7 + 5, 128).float()

    t1 = AdvantageNet(17*7 + 5, 128).float()

    # t0.load_state_dict(torch.load("C:/Users/huffm/PycharmProjects/HeadsUp_deepCFR/adv/2022-03-28T10-24-27/177"))
    # t1.load_state_dict(torch.load("C:/Users/huffm/PycharmProjects/HeadsUp_deepCFR/adv/2022-03-28T10-24-27/177"))


    gamma = .999999999
    randomness = 0

    if torch.cuda.is_available():
        print("Using cuda")
        t0 = t0.to(device="cuda")
        t1 = t1.to(device="cuda")
        strat_net = strat_net.to(device="cuda")

    sum_util = 0
    util_list = []
    cfr_util_list = []
    avg_util_list = []
    for t in range(T):
        p = t % 2
        init_history = History()
        print(f"t={t}")
        np.random.choice(range(20), 10, replace=False)
        start = time.time()
        sum_mr = 0
        for n in tqdm(range(N)):

            randomness *= gamma
            init_history.env.done = False

            #init_history.env.reset([.5,.5], np.random.randint(10,100))
            init_history.env.reset([.5,.5], 1.5)
            if n == N-1:
                h_copy = init_history.copy()
            
            
            if p == 0:
                mr = traverse(init_history, p, t0, t1, adv0, strat, m_bd0, t + 1, 0, -1)
                cfr_util_list.append((mr-1.5))
            else:
                mr = traverse(init_history, p, t0,  t1, adv1, strat, m_bd1, t + 1, 0, -1)
                cfr_util_list.append((1.5-mr))
            sum_mr += mr
            
        if p == 0:
            print(len(adv0))
            # train_raise(rm, rn, 100)
            train_adv(adv0, t0, 0, 1)
            #train(m_bd0, t0, 0, 1)
            # train_raise(rm, rn, 100)

            copy_net = t0
        else:
            print(len(adv1))
            # train_raise(rm, rn, 100)
            train_adv(adv1, t1, 0, 1)
            #train(m_bd1, t1, 0, 1)

            copy_net = t1
        t1.update_frozen()
        t0.update_frozen()
        if p == 0:
            util = sum_mr / N - 1.5
        elif p == 1:
            util = 1.5 - sum_mr / N
        #print(util)
        
        sum_util += util
        avg_util = sum_util / (t+1)
        best_response = max(cfr_util_list)
        print('Average utility exploit:', best_response + avg_util)
        avg_util_list.append(abs(avg_util)/10)
        last_hand_test(copy_net, h_copy.env)
        print(f"Last reward: {util}")
        util_list.append(util)
        torch.save(copy_net.state_dict(), str(save_dir2) + "/" + str(t))
        end = time.time()
        print(f"Loop time: {end - start}")
 
 
    train_strat(strat, strat_net, 1,1)
    strat_net.update_frozen()
    torch.save(strat_net.state_dict(), str(save_dir1) + "/" + str(t))
    return strat_net, util_list, avg_util_list, cfr_util_list

def traverse(h, p, t1, t2, m_adv, m_strat, m_bd, t, depth, la):
    #('depth', depth)

    if (h.env.done) and h.env.action_player.pnum == p:

        fss = h.env.action_player.stack_size

        return fss
    elif h.env.done:
        fss = h.env.inaction_player.stack_size

        return fss


    if depth > 13 and la != 0:
        if h.env.game.to_call != 0:
            amnt = min(h.env.game.to_call, h.env.action_player.stack_size)
            h.env.action_player.stack_size -= amnt
            h.env.game.pot_size -= (h.env.game.to_call - amnt)
            h.env.inaction_player.stack_size += (h.env.game.to_call - amnt)
        while h.env.game.phase != "showdown":
            h.env.game.next_card()
        winner = h.env.game.determine_winner()
        if winner == "tie":
            h.env.game.p1.stack_size += h.env.game.pot_size / 2
            h.env.game.p2.stack_size += h.env.game.pot_size / 2
        elif winner == "p1":
            h.env.game.p1.stack_size += h.env.game.pot_size
        elif winner == "p2":
            h.env.game.p2.stack_size += h.env.game.pot_size
        h.env.game.pot_size = 0
        h.env.game.p1.last_win = h.env.game.p1.stack_size - h.env.game.p1.starting_stack_size
        h.env.game.p2.last_win = h.env.game.p2.stack_size - h.env.game.p2.starting_stack_size
        h.env.reward = h.env.action_player.last_win
        h.env.inact_reward = h.env.inaction_player.last_win
        h.env.done = True
        return traverse(h, p, t1, t2, m_adv, m_strat, m_bd, t, depth + 1, la= -1)



    state = get_state_array(h.env, h.env.action_player, h.env.inaction_player)
    if torch.cuda.is_available():
        state = state.to(device="cuda")



    if h.env.action_player.pnum == p:  # Player gets to choose
        if p == 0:
            player = t1
        else:
            player = t2


        # amnt = rn(state)
        # ch = torch.distributions.chi2.Chi2(amnt[0])
        # uamnt = ch.sample() + amnt[1]
        reward = 0
        strat =  player(state, False)
        rewards = []
        ruse = []
        iss = h.env.action_player.stack_size
        for i in range(3):
            # if depth < 2:
            #     print(f"Depth {depth}: action {i}")
            #     print(f"len adv: {len(m_adv)}")

            temp_history = h.copy()
            temp_history.env.step(i)

            fss = traverse(temp_history, p, t1, t2, m_adv, m_strat, m_bd, t, depth + 1, la=i)

            r = fss - iss

            if r is None:
                r = 0
            reward += r
            rewards.append(r)
            ruse.append(r)

        advantages = []
        if  np.random.random() < 1/(1 + depth - p):
            for i in range(3):

                if isinstance(rewards[i], torch.Tensor):
                    advantages.append(rewards[i])
                else:
                    advantages.append(rewards[i])
            advantages = torch.tensor(advantages)
            advantages = strat + advantages*player.sm(strat)
            
            #print('sm strat' , player.sm(strat))
            
            #print('adv', advantages)
# =============================================================================
#             best_decision_index = torch.argmax(advantages)
#             best_decision = torch.zeros_like(advantages)
#             best_decision[best_decision_index] += 1
# =============================================================================
            #print('best decision', best_decision)

            m_adv.append(np.array([h.env.get_state(), advantages, t, h.env.inaction_player.card_1, h.env.inaction_player.card_2], dtype=object))
           # m_bd.append(np.array([h.env.get_state(), best_decision, t, h.env.inaction_player.card_1, h.env.inaction_player.card_2], dtype=object))
        temp = torch.sum(torch.tensor(ruse) * player.sm(strat))
        return temp.item() + iss
    elif h.env.action_player.pnum == 1-p:
        if p == 1:
            player = t1
        else:
            player = t2

        action = player(state, False)
        best_decision_index = torch.argmax(action)
        best_decision = torch.zeros_like(action)
        best_decision[best_decision_index] += 1
        #print(player.sm(action))
        #amnt = rn(state)
        # ch = torch.distributions.chi2.Chi2(amnt[0])
        # uamnt = ch.sample() + amnt[1]
        if np.random.random() < 1/(depth + p):
            #m_strat.append(np.array([h.env.get_state(), player.sm(action), t], dtype=object))
            ''' Training Strategy on best_decision'''
            m_strat.append(np.array([h.env.get_state(), best_decision, t], dtype=object))
            
            
        sm = torch.softmax(action.cpu(), dim=0)

        r = np.random.choice([i for i in range(len(action))],
                             1,
                             p=sm.detach().numpy())

        temp_history = h.copy()
        temp_history.env.step(r)

        return traverse(temp_history, p, t1, t2, m_adv, m_strat, m_bd, t, depth + 1, la=r)



def lossf_adv(d1, d2, ts, maxt, indicator=False):
    # if isinstance(d1, list):
    #     if len(d1) > 1:
    #         d1 = torch.cat(d1[:-1])
    #     else:
    #         d1 = torch.cat(d1)
    #     d1 = d1.flatten()
    #
    # if not indicator:
    #     d2 = torch.stack(d2)

    # ts = torch.cat([torch.tensor([ts[i] for i in range(len(d1[0]))])for j in range(len(d1))])
    # ts = torch.reshape(ts, d1.shape)
    # ts = ts/maxt
    # ts = ts.to(device="cuda")
    try:
        t = [ts for i in range(d1.shape[1])]
    except IndexError:
        t = ts
    if isinstance(t, list):
        t = torch.squeeze(torch.stack(t)).t()
        
    '''Testing with different loss functions'''
    lf = torch.nn.MSELoss()
    #lf = torch.nn.CrossEntropyLoss()
    if d1.shape == d2.shape:
        ret = lf(torch.squeeze(d1) * t, torch.squeeze(d2) * t)
        return ret
    return None


def train_adv(data, network, indicator, train_iters):
    if len(data) < 2:
        return network
    data = np.array(data, dtype=object)

    maxt = np.max(data[:,2])

    train_dataset = ActionDataset(data[:, 1], data[:, 0], data[:, 2])
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              pin_memory=False)

    total_loss = 0
    mu = 1
    for i in range(train_iters):
        # ts = []
        # actions = []
        # data_actions = []


        for batch, (data_action, state, t) in enumerate(train_loader):
            #state, t,= state.to(device="cuda"), t.to(device="cuda")
            state, t, =  state.float(), t.float()
            for j in range(len(data_action)):
                data_action[j] = data_action[j].float()
            network.optimizer.zero_grad()
            action = network(state)


            # ts.append(t)
            # actions.append(action)
            # data_actions.append(data_action)
            loss = lossf_adv(action, data_action, t, maxt, indicator)
            if loss is not None:
                loss = loss.float()
                total_loss += loss.item()

                loss.backward()
                network.optimizer.step()

        # if indicator:
        #     aces_test(network,1)

    print(f"Average batch Loss: {total_loss / (i + 1) / (batch + 1)}")
    #return network

def lossf_strat(d1, d2, ts, maxt, indicator=False):
    # if isinstance(d1, list):
    #     if len(d1) > 1:
    #         d1 = torch.cat(d1[:-1])
    #     else:
    #         d1 = torch.cat(d1)
    #     d1 = d1.flatten()
    #
    # if not indicator:
    #     d2 = torch.stack(d2)

    # ts = torch.cat([torch.tensor([ts[i] for i in range(len(d1[0]))])for j in range(len(d1))])
    # ts = torch.reshape(ts, d1.shape)
    # ts = ts/maxt
    # ts = ts.to(device="cuda")
    try:
        t = [ts for i in range(d1.shape[1])]
    except IndexError:
        t = ts
    if isinstance(t, list):
        t = torch.squeeze(torch.stack(t)).t()
        
    '''Testing with different loss functions'''
    #lf = torch.nn.MSELoss()
    lf = torch.nn.CrossEntropyLoss()
    if d1.shape == d2.shape:
        ret = lf(torch.squeeze(d1) * t, torch.squeeze(d2) * t)
        return ret
    return None

def train_strat(data, network, indicator, train_iters):
    if len(data) < 2:
        return network
    data = np.array(data, dtype=object)

    maxt = np.max(data[:,2])

    train_dataset = ActionDataset(data[:, 1], data[:, 0], data[:, 2])
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              pin_memory=False)

    total_loss = 0
    mu = 1
    for i in range(train_iters):
        # ts = []
        # actions = []
        # data_actions = []


        for batch, (data_action, state, t) in enumerate(train_loader):
            #state, t,= state.to(device="cuda"), t.to(device="cuda")
            state, t, =  state.float(), t.float()
            for j in range(len(data_action)):
                data_action[j] = data_action[j].float()
            network.optimizer.zero_grad()
            action = network(state)


            # ts.append(t)
            # actions.append(action)
            # data_actions.append(data_action)
            loss = lossf_strat(action, data_action, t, maxt, indicator)
            if loss is not None:
                loss = loss.float()
                total_loss += loss.item()

                loss.backward()
                network.optimizer.step()

        # if indicator:
        #     aces_test(network,1)

    print(f"Average batch Loss: {total_loss / (i + 1) / (batch + 1)}")
    #return network


strat, util, avg_util, cfr_util_list = deep_gxfp(1000, 1000)

import numpy as np
util_array = np.array(cfr_util_list)
np.save('C:/Users/Vik/GameTheory/Deep-GXFP/deep_gxfp_exploit', util_array)
cfr_every_100 = []
for t in range(len(cfr_util_list)):
    if t % 100 == 0:
        cfr_every_100.append(sum(cfr_util_list[t-100:t])/(100))
        
for t in range(len(avg_util)):
    avg_util[t] = avg_util[t] * 10000
    
        
import matplotlib.pyplot as plt
import numpy as np
x = np.array(avg_util)
plt.plot(x)
plt.title('Exploitability of HUNLH with Deep GXFP')
plt.xlabel('CFR Iterations within Network (Hundreds)')
plt.ylabel('Exploitability')
plt.show() 
