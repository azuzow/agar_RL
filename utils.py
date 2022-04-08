import cv2
import pytesseract
import re
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self, n_actions, h=128, w=128,in_channels=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, n_actions)
        self.float()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def format_term_img(img):
    #mask out bottom bar
    cv2.rectangle(img,(0,1128),(1221,1154),(255,255,255),-1)
    #mask out leaderboard
    cv2.rectangle(img,(1020,7),(1212,250),(255,255,255),-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128))
    return img


def format_frame (img, username,prev_fail,get_score=False):
    h,w,c = img.shape

    if username == "steph":
        game_height = 1280
        game_width = 2400

        lb_x1 = 2025
        lb_x2 = 2500
        lb_y1 = 15
        lb_y2 = 460

        score_x1 = 18
        score_x2 = 153
        score_y1 = 1231
        score_y2 = 1265
    elif username == "alex":
        game_height = 1154
        game_width = 1221

        lb_x1 = 1020
        lb_x2 = 1211
        lb_y1 = 9
        lb_y2 = 245

        score_x1 = 21
        score_x2 = 85
        score_y1 = 1120
        score_y2 = 1135
        if prev_fail==2:
            score_x2+=10
    else:
        assert False

    if not (h == game_height and w == game_width):
        print (h,w)
        return None, None, True

    #mask out leader boards
    cv2.rectangle(img,(lb_x1,lb_y1),(lb_x2,lb_y2),(255,255,255),-1)


    if get_score:
        score = img[score_y1:score_y2, score_x1:score_x2]

        # score = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)
        lower = np.array([0,0,0], dtype = "uint16")
        upper = np.array([235,235,235], dtype = "uint16")
        score = cv2.inRange(score,lower,upper)

        score_str = pytesseract.image_to_string(score)

        if "score" in score_str.lower():
            try:
                score = int(re.findall(r'\d+',score_str)[0])
            except IndexError:
                return None, None, True
            failed = False
        else:
            #TODO: could trigger early if blob is in score label region
            score = None
            failed = True
    #mask out score
    cv2.rectangle(img,(score_x1,score_y1),(score_x2,score_y2),(255,255,255),-1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128))
    if get_score:
        return img,score,failed
    else:
        return img

def img2score(img, username,timestep,prev_fail):

    return format_frame (img, username,prev_fail,get_score=True)
# #
# img = cv2.imread("agent_observations/4.png")
# # #
# # # cv2.imshow("img",img)
# # # cv2.waitKey(0)
# img, score, done = img2score(img,"steph",1)
# print (score,done)
