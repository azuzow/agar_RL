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
from torch.autograd import Variable
from collections import namedtuple, deque
import random
import numpy as np
import time
import timeit
import math
import wandb
from agar import env
import utils
from utils import ReplayMemory, Net

path_to_adblock = r'1.42.2_0'
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_argument('load-extension=' + path_to_adblock)
name='cs394r'



agar1 = env(chrome_options,name)

memory = ReplayMemory(1000)


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


target_DQN = Net(n_actions = len(agar1.action_space)).to(device)
policy_DQN = Net(n_actions = len(agar1.action_space)).to(device)
target_DQN.load_state_dict(policy_DQN.state_dict())


# policy_DQN.load_state_dict(torch.load('models/policy_DQN.pt'),strict=False)
# target_DQN.load_state_dict(torch.load('models/target_DQN.pt'),strict=False)

target_DQN.eval()
optimizer = optim.RMSprop(policy_DQN.parameters())

BATCH_SIZE = 2
GAMMA = 0.99
TARGET_UPDATE = 5
SAVE_UPDATE = 20
N_EPISODES = 500

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
steps_done = 0


def select_action(state):


    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        policy_DQN.eval()
        action = policy_DQN(Variable(state.unsqueeze(0), volatile=True)).data.max(1)[1].view(1, 1)
        policy_DQN.train()
        return action
    else:
        return torch.tensor([[random.randrange(2)]])

def update_model():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state)).to(device)
    batch_action = Variable(torch.cat(batch_action)).to(device)
    batch_reward = Variable(torch.cat(batch_reward)).to(device)


    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), dtype=torch.bool, device=device)
    non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])

    current_q_values = policy_DQN(batch_state).gather(1, batch_action.unsqueeze(0))

    max_next_q_values = torch.zeros(BATCH_SIZE, device=device).float()
    max_next_q_values[non_final_mask] = target_DQN(non_final_next_states).max(1)[0]

    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values.squeeze())
    criterion = nn.SmoothL1Loss()
    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print ("LOSS: " + str(loss))


episode = 0
episode_rewards = []
episode_rewards=np.load("episode_rewards.npy").tolist()
episode_timestamps=[]
episode_loss=[]
wandb.init(project="CS394R_AGARIO", entity="cs394ragario")
for episode in range(N_EPISODES):
    episode_return = 0
    prev_score = 10
    timestep = 0

    state = agar1.reset()
    # time.sleep(.5)
    while True:
        if state is not None:
            action = select_action(state)
        else:
            action = random.randrange(0,len(agar1.action_space))

        next_state,score,failed,restart,done = agar1.step(action.item(),timestep,episode)

        if not done:
            reward = 1
        else:
            reward = -10

        episode_return+=reward
        episode_rewards.append(episode_return)

        action = torch.tensor([action])
        reward = torch.tensor([reward])

        print ("(REWARD: " + str(reward) + ", DONE: " + str(done) + ")")

        # print (next_state)
        # print ("\n")
        if not done and state is not None and len(next_state)>0:
            state.to(device)
            # next_state.to(device)
            memory.push(state.unsqueeze(0), action, next_state.unsqueeze(0), reward)
        elif state is not None:
            memory.push(state.unsqueeze(0), action, None, reward)


        update_model()

        timestep +=1
        if  restart:
            episode +=1
            wandb.log({"timestep": timestep})

            episode_timestamps.append(timestep)
                # episode_loss.append(loss)
            break

    if episode % TARGET_UPDATE == 0:
        target_DQN.load_state_dict(policy_DQN.state_dict())
    if episode % SAVE_UPDATE == 0:
        torch.save(target_DQN.state_dict(), "models/target_DQN.pt")
        torch.save(policy_DQN.state_dict(), "models/policy_DQN.pt")
        np.save("episode_rewards.npy",episode_rewards)
        # np.save("episode_losses.npy",episode_loss)
        np.save("episode_timestamps.npy",episode_timestamps)
