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
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(64)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))))
        linear_input_size = convw * convh * 64
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


def format_frame (img, username,prev_fail,classifier,get_score=False):
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
        game_height = 629
        game_width = 756


        lb_x1 = 633
        lb_x2 = 754
        lb_y1 = 4
        lb_y2 = 257

        score_x1 = 46
        score_x2 = 70
        score_y1 = 604
        score_y2 = 612
    else:
        assert False

    if not (h == game_height and w == game_width):
        print('height',h,'width',w)
        return None, None, True

    #mask out leader boards
    cv2.rectangle(img,(lb_x1,lb_y1),(lb_x2,lb_y2),(255,255,255),-1)
   
    if get_score:

        score = img[score_y1:score_y2, score_x1:score_x2]
 
        
        score_=cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)

        score_ = cv2.resize(score_,(100,40),interpolation=cv2.INTER_CUBIC)
        
        score_ = np.where(score_>=210,255,0).astype(np.uint8)
        score =  cv2.bitwise_not(score_)
        cv2.imwrite('1.png',score)
        cv2.imwrite('2.png',score)
        cv2.imwrite('3.png',score)
        cv2.imwrite('4.png',score)
        cv2.imwrite('5.png',score)
        cv2.imwrite('6.png',score)
        score = score.astype(np.float32)

        digit_0 = torch.tensor(score[:, 0:22]/255.0).unsqueeze(0).unsqueeze(0)
        digit_1 = torch.tensor(score[:, 22:44]/255.0).unsqueeze(0).unsqueeze(0)
        digit_2 = torch.tensor(score[:, 44:66]/255.0).unsqueeze(0).unsqueeze(0)
        digit_3 = torch.tensor(score[:, 66:88]/255.0).unsqueeze(0).unsqueeze(0)
        
        output_0 =torch.nn.functional.softmax(classifier(digit_0).squeeze(0))
        output_1 =torch.nn.functional.softmax(classifier(digit_1).squeeze(0))
        output_2 = torch.nn.functional.softmax(classifier(digit_2).squeeze(0))
        output_3 = torch.nn.functional.softmax(classifier(digit_3).squeeze(0))
      
        outputs=[output_0,output_1,output_2,output_3]
        for i in range(len(outputs)):

            if outputs[i].max()<=.999 or outputs[i].argmax().item()==10:
                outputs[i]=''
            else:
 
                outputs[i]=str(outputs[i].argmax().item())

        outputs = "".join(outputs)
        score = int(outputs)

        if  outputs!= 0:
            failed=False
        else:
            #TODO: could trigger early if blob is in score label region
            return None, None, True
    #mask out score
    cv2.rectangle(img,(score_x1,score_y1),(score_x2,score_y2),(255,255,255),-1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128))
    if get_score:
        return img,score,failed
    else:
        return img

def img2score(img, username,timestep,prev_fail,classifier):

    return format_frame (img, username,prev_fail,classifier,get_score=True)

# img = cv2.imread("agent_observations/4.png")

# cv2.imshow("img",img)
# cv2.waitKey(0)
# img, score, done = img2score(img,"steph",1)
# print (score,done)
