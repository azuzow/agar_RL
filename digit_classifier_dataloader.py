import os
from torchvision.io import read_image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
####################################################
#       Create Train, Valid and Test sets
####################################################
train_data_path = '/home/alexzuzow/Desktop/agar_multiagent/DATA/Train' 
test_data_path = '/home/alexzuzow/Desktop/agar_multiagent/DATA/Test'

train_image_paths = [] #to store image paths in list
classes = [] #to store class values

#1.
# get all the paths from train_data_path and append image paths and class to to respective lists
# eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
# eg. class -> 26.Pont_du_Gard
for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/')[-1]) 
    train_image_paths.append(glob.glob(data_path + '/*'))
train_image_paths = [item for sublist in train_image_paths for item in sublist]

random.shuffle(train_image_paths)
# random.shuffle(test_image_paths)
# print('train_image_path example: ', train_image_paths[0])
# print('class example: ', classes[0])

#2.
# split train valid from train paths (80,20)
train_image_paths, valid_image_paths = train_image_paths[:int(len(train_image_paths))], train_image_paths[int(len(train_image_paths)):] 

#3.
# create the test_image_paths
test_image_paths = []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))
test_image_paths = [item for sublist in test_image_paths for item in sublist]
random.shuffle(test_image_paths)
print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))


# train_transforms = A.Compose(
#     [
#         A.SmallestMaxSize(max_size=350),
#         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
#         A.RandomCrop(height=256, width=256),
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#         A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#         A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
#         ToTensorV2(),
#     ]
# )
class DigitDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image= (image/255.0).astype(np.float32)

        label = image_filepath.split('/')[7]
        label= label[0]
        
        return image, label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,kernel_size= 3)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels= 64, kernel_size=3)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels= 128, kernel_size=3)
        self.fc1 = nn.Linear(14336, 128)
        self.fc2 = nn.Linear(128, 11)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


def train(model, device, train_loader, optimizer, epoch,loss):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.shape[0] < 128:
            continue
        target_ = np.zeros((128,11))

        for batch_num, num in enumerate(target):
            num=int(num)
            target_[batch_num,num]=1
        target=torch.Tensor(target_)
        # print(target.shape)
        data= data.unsqueeze(1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = loss(output,target)
        output.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), output.item()))


def test(model, device, test_loader,loss):
    model.eval()
    test_loss = 0
    correct = 0
    incorrect=0
    with torch.no_grad():
        for data, target in test_loader:

            if data.shape[0] < 128:
                continue
            target_ = np.zeros((128,11))
            for batch_num, num in enumerate(target):
                num=int(num)
                target_[batch_num,num]=1

            target=torch.Tensor(target_)
            data= data.unsqueeze(1)
            # print(data.shape,target.shape)
            data, target = data.to(device), target.to(device)
        

       

            
            output = model(data)
            test_loss += loss(output,target).item()  # sum up batch loss
            # print(output.shape)
            pred = output.argmax(dim=1,)  # get the index of the max log-probability
            ground_truth = target.argmax(dim=1)
            for i in range(len(pred)):
                if pred[i]!= ground_truth[i]:
                    print(pred[i].item(),ground_truth[i].item())
                    incorrect+=1

            correct += torch.sum(pred==ground_truth)
            # print(100*(torch.sum(pred==ground_truth)/64).item())
            # print(ground_truth)
        test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('incorrect:' ,incorrect,'/',len(test_loader.dataset))
    correct = 0
    incorrect=0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = DigitDataset(train_image_paths)
valid_dataset = DigitDataset(valid_image_paths) #test transforms are applied
test_dataset = DigitDataset(test_image_paths)


train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True

)

test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False
)



# model = CNN().to(device)
# optimizer = optim.Adam(model.parameters(), lr = 0.01)   
# loss = nn.CrossEntropyLoss()

# for epoch in range(1, 101):
#     train(model, device, train_loader, optimizer, epoch,loss)
#     # test(model, device, test_loader,loss)


# torch.save(model.state_dict(), 'models/classifier.pt')