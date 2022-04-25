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
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self, n_actions, h=128, w=128,in_channels=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(128)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))))
        linear_input_size = convw * convh * 128
        self.head = nn.Linear(linear_input_size, n_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x/255
        x = x.to(device)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
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


def is_noise(img):
    # img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    counts = (cv2.countNonZero(img))
    if counts <=100 or counts >=800:
        return True

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    # if len(contours) == 1 and cv2.contourArea(contours[0]) > 30 and cv2.arcLength(contours[0],True) < 200:
    if len(contours) == 1:
        return True

    return False

def format_term_img(img):
    #mask out bottom bar
    cv2.rectangle(img,(0,1128),(1221,1154),(255,255,255),-1)
    #mask out leaderboard
    cv2.rectangle(img,(1020,7),(1212,250),(255,255,255),-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128))
    return img


def format_frame (img, username,prev_fail,classifier,get_score=False):
    h,w,c = img.shape

    if username == "steph":
        game_height = 1280
        game_width = 2400

        lb_x1 = 2025
        lb_x2 = 2500
        lb_y1 = 15
        lb_y2 = 470

        score_x1 = 18
        score_x2 = 153
        score_y1 = 1231
        score_y2 = 1265
    elif username == "alex":
        game_height = 1029
        game_width = 1156


        lb_x1 = 965
        lb_x2 = 1150
        lb_y1 = 4
        lb_y2 = 230

        score_x1 = 13
        score_x2 = 100
        score_y1 = 990
        score_y2 = 1015
    else:
        assert False

    # if not (h == game_height and w == game_width):
    #     print('height',h,'width',w)
    #     return None, None, True

    #mask out leader boards

    cv2.rectangle(img,(lb_x1,lb_y1),(lb_x2,lb_y2),(255,255,255),-1)

    if get_score:

        score = img[score_y1:score_y2, score_x1:score_x2]


        score_=cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)
        score_ = score_[:,51:]

        print(score_.shape)
        score_ = cv2.resize(score_,(180,125),interpolation=cv2.INTER_CUBIC)

        score_ = np.where(score_>=210,0,255).astype(np.uint8)
        # cv2.imwrite('5.png',score_)


        digit_0_ = score_[20:-10, 0:40]
        digit_1_ = score_[20:-10, 40:80]
        digit_2_ = score_[20:-10, 80:120]
        digit_3_ = score_[20:-10, 120:160]

        # cv2.imwrite('1.png',digit_0_)
        # cv2.imwrite('2.png',digit_1_)
        # cv2.imwrite('3.png',digit_2_)
        # cv2.imwrite('4.png',digit_3_)
        img_0 = "digit_0_{}.png".format(time.time())
        img_1 = "digit_1_{}.png".format(time.time())
        img_2 = "digit_2_{}.png".format(time.time())
        img_3 = "digit_3_{}.png".format(time.time())
        # cv2.imwrite('/home/alexzuzow/Desktop/agar_multiagent/scores/0/'+img_0,digit_0_)
        # cv2.imwrite('/home/alexzuzow/Desktop/agar_multiagent/scores/1/'+img_1,digit_1_)
        # cv2.imwrite('/home/alexzuzow/Desktop/agar_multiagent/scores/2/'+img_2,digit_2_)
        # cv2.imwrite('/home/alexzuzow/Desktop/agar_multiagent/scores/3/'+img_3,digit_3_)
        # # print('TEST1')
        digit_0 = torch.tensor(digit_0_.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        digit_1 = torch.tensor(digit_1_.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        digit_2 = torch.tensor(digit_2_.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        digit_3 = torch.tensor(digit_3_.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        # print('TEST2')
        digits_=[digit_0_,digit_1_,digit_2_,digit_3_]
        digits=[digit_0,digit_1,digit_2,digit_3]
        outputs=[]

        for i in range(len(digits)):
            if classifier is not None:
                value =torch.nn.functional.softmax(classifier(digits[i]).squeeze(0))
            else:
                value = 0
            outputs.append(value)
            # print(outputs[i])
        for i in range(len(outputs)):

            # print(outputs[i].max())
            if outputs[i]==None or outputs[i].max()<.9 or outputs[i].argmax().item()==10 :
                outputs[i]=''
            else:
                if i >1:
                    pass
                outputs[i]=str(outputs[i].argmax().item())

        score = "".join(outputs)
        if len(score)>0:

            score = int(score)
        else:
            score=-100

        if  score!= -100:
            failed=False
        else:
            #TODO: could trigger early if blob is in score label region
            failed=True
    #mask out score
    # failed=False
    cv2.rectangle(img,(score_x1,score_y1),(score_x2,score_y2),(255,255,255),-1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128))
    img = cv2.GaussianBlur(img,(5,5),0)
    ret,img = cv2.threshold(img,200,255,cv2.THRESH_TOZERO)

    if get_score:
        return img,score,failed
    else:
        return img

def img2score(img, username,timestep,prev_fail,classifier):

    return format_frame (img, username,prev_fail,classifier,get_score=True)


# 
# img = cv2.imread("/Users/stephanehatgiskessell/Downloads/agent_observations/82.png")
#
#
# img = format_frame (img, "alex",False,None,get_score=False)
# cv2.imshow("img",img)
# cv2.waitKey(0)
# img, score, done = img2score(img,"steph",1)
# print (score,done)
