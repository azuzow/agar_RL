from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple, deque
import random
import numpy as np
import time
import timeit
import math

from agar import env
import utils
from utils import ReplayMemory, Net

path_to_adblock = r'1.42.2_0'
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_argument('load-extension=' + path_to_adblock)
name='cs394r'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agar1 = env(chrome_options,name)

memory = ReplayMemory(100000)
target_DQN = Net(n_actions = len(agar1.action_space)).to(device)
policy_DQN = Net(n_actions = len(agar1.action_space)).to(device)
target_DQN.load_state_dict(policy_DQN.state_dict())

# policy_DQN.load_state_dict(torch.load('/home/alexzuzow/Desktop/saved_models/policy_DQN.pt'))
# target_DQN.load_state_dict(torch.load('/home/alexzuzow/Desktop/saved_models/target_DQN.pt'))
target_DQN.eval()
optimizer = optim.RMSprop(policy_DQN.parameters())

BATCH_SIZE = 64
GAMMA = 0.9
TARGET_UPDATE = 5
SAVE_UPDATE = 20
N_EPISODES = 500

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

def select_action(state,steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    print (eps_threshold)
    steps_done+=1
    if sample <= eps_threshold:
        return random.randrange(0,len(agar1.action_space)),steps_done
    else:

        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected re
            action =  torch.squeeze(policy_DQN(state.unsqueeze(0)).max(1)[1].view(1, 1))
            return action.item(),steps_done

def update_model():
    #For the sake of efficiency, a lot of this code was taken from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = utils.Transition(*zip(*transitions))

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    state_action_values = policy_DQN(state_batch).gather(1, action_batch.unsqueeze(0))


    #if next state is None mark it as 0
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_DQN(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(torch.squeeze(state_action_values), expected_state_action_values)
    print('====',loss,'====')
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_DQN.parameters():
        param.grad.data.clamp_(-1, 1)
        # print (param)
    optimizer.step()
    # print ("UPDATED MODEL, LOSS=",loss)
    # print ("\n")



episode = 0
episode_rewards = []
episode_timestamps=[]
episode_loss=[]
steps_done = 0

for episode in range(N_EPISODES):
    episode_return = 0
    prev_score = 10
    timestep = 0

    state = agar1.reset()
    time.sleep(.5)
    while True:

        action,steps_done = select_action(state,steps_done)
        s = timeit.default_timer()
        next_state,score,failed,restart,done = agar1.step(action,timestep,episode)
        e = timeit.default_timer()
        print ("step time: " + str(e-s))
        if not failed:
            reward = score - prev_score
            episode_return+=reward
            prev_score = score

            if not done and state is not None:
                memory.push(state.unsqueeze(0), torch.tensor([action]), next_state.unsqueeze(0), torch.tensor([reward]))
            elif state is not None:
                memory.push(state.unsqueeze(0), torch.tensor([action]), None, torch.tensor([reward]))


        update_model()

        timestep +=1
        if  restart:
            episode +=1
            if episode_return != 0:
                episode_rewards.append(episode_return)
                episode_timestamps.append(timestep)
                # episode_loss.append(loss)
            break
    if episode % TARGET_UPDATE == 0:
        target_DQN.load_state_dict(policy_DQN.state_dict())
    if episode % SAVE_UPDATE == 0:
        torch.save(target_DQN.state_dict(), "/home/alexzuzow/Desktop/saved_models/target_DQN.pt")
        torch.save(policy_DQN.state_dict(), "/home/alexzuzow/Desktop/saved_models/policy_DQN.pt")
        np.save("episode_rewards.npy",episode_rewards)
        # np.save("episode_losses.npy",episode_loss)
        np.save("episode_timestamps.npy",episode_timestamps)
