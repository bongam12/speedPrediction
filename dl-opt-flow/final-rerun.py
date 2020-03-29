import cv2 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets 
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
from torch.autograd import Variable

# to change the brightness
def change_brightness(image, bright_factor):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb

# to get the frames from the video
def get_frames(filename):
    vidcap = cv2.VideoCapture(filename)
    success = True
    data = []
    while success:
        success,image = vidcap.read()
        if success:
            data.append(image)
    return data


def get_speed_data(filename): # getting speed from txt
    speed=[]
    with open(filename) as f:
        for line in f:
            val = line.rstrip('\n')
            val = float(val)
            speed.append(val)
    return speed            


# speed_data and images_data are lists that have the speed(target variable) and frames from video
# please change the path if the train.mp4 and train.txt are in different folder

path_txt = "train.txt"
path_mp4 = "train.mp4"

speed_data = get_speed_data(path_txt)
images_data= get_frames(path_mp4)


def opticalFlowDense(frame1,frame2):
    frame1 = frame1[200:400]
    frame1 = cv2.resize(frame1, (0,0), fx = 0.4, fy=0.5)
    frame2 = frame2[200:400]
    frame2 = cv2.resize(frame2, (0,0), fx = 0.4, fy=0.5)
    flow = np.zeros_like(frame1)
    prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    flow_mat = None
    image_scale = 0.4
    nb_images = 1
    win_size = 12
    nb_iterations = 2
    deg_expansion = 8
    STD = 1.2
    extra = 0   
    
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,flow_mat,image_scale,nb_images, win_size, nb_iterations, deg_expansion, STD,0)
    
    mag, ang = cv2.cartToPolar(flow_data[...,0], flow_data[...,1])
    flow[...,1] = 255
    flow[...,0] = ang*180/np.pi/2
    flow[...,2] = (mag *15).astype(int)
    return flow

# optical flow function
def opticalFlowDense(frame1,frame2):
    frame1 = frame1[200:400]
    frame1 = cv2.resize(frame1, (0,0), fx = 0.4, fy=0.5)
    frame2 = frame2[200:400]
    frame2 = cv2.resize(frame2, (0,0), fx = 0.4, fy=0.5)
    flow = np.zeros_like(frame1)
    prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    flow_mat = None
    image_scale = 0.4
    nb_images = 1
    win_size = 12
    nb_iterations = 2
    deg_expansion = 8
    STD = 1.2
    extra = 0   
    
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,flow_mat,image_scale,nb_images, win_size, nb_iterations, deg_expansion, STD,0)
    
    mag, ang = cv2.cartToPolar(flow_data[...,0], flow_data[...,1])
    flow[...,1] = 255
    flow[...,0] = ang*180/np.pi/2
    flow[...,2] = (mag *15).astype(int)
    return flow

# optical_flow_images is a list that contains opticalFlow flow between every two consecutive images
# speed_Data_final ia average speed every two consecutive image

optical_flow_images = []
speed_Data_final = []
for i in range(0,len(images_data)-1):
    img = cv2.resize(opticalFlowDense(images_data[i],images_data[i+1]),(200,66))/255
    optical_flow_images.append(img)
    
    mean_speed = (speed_data[i] + speed_data[i+1])/2
    label = np.asarray(mean_speed,dtype= np.float32)
    speed_Data_final.append(label)
    

train_data = optical_flow_images[:int(0.8*len(optical_flow_images))]
train_labels = speed_Data_final[:int(0.8*len(speed_Data_final))]
val_data = optical_flow_images[int(0.8*len(optical_flow_images)):]
val_labels = speed_Data_final[int(0.8*len(speed_Data_final)):]

class imagedataset(Dataset):

    def __init__(self,video_file,speed_file,transforms):
        self.transforms = transforms
        self.data   = video_file
        self.labels = speed_file

    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self,index):
        img   = self.data[index]
        label = self.labels[index]
        transformed_image = self.transforms(img)  
        return transformed_image,label
    
transformations = transforms.Compose([transforms.ToTensor()])

train_data = imagedataset(train_data,
                          train_labels,
                          transforms = transformations)
val_data =   imagedataset(val_data,
                          val_labels,
                          transforms = transformations)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)


# CNN architecture - End-to-End Deep Learning for Self-Driving Cars Architecture with some modifications

# Need to put drop out 
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0)
        self.relu1 = nn.ReLU()

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2, padding=0)
        self.relu2 = nn.ReLU()

        # Convolution 3
        
        self.cnn3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2, padding=0)
        self.relu3 = nn.ReLU()
        self.dropout1=nn.Dropout(p=0.5)
        
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        
        # Convolution 5
        
        self.cnn5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu5 = nn.ReLU()
        
        # Fully connected 1 
        #self.fc1 = nn.Linear(in_features=1280, out_features=100)
        self.fc1 = nn.Linear(in_features=1152, out_features=100)
        self.relu_fc1 = nn.ReLU()
   
        
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.relu_fc2 = nn.ReLU()
        
        self.fc3 = nn.Linear(in_features=50, out_features=10)
        self.relu_fc3 = nn.ReLU()
        
        self.fc4 = nn.Linear(in_features=10, out_features=1)

  
    def forward(self, x):
        out = self.cnn1(x) #1
        out = self.relu1(out)
        out = self.cnn2(out) #2
        out = self.relu2(out)
        out = self.cnn3(out) #3
        out = self.relu3(out)
        out = self.dropout1(out) 
        out = self.cnn4(out) #4
        out = self.relu4(out)
        out = self.cnn5(out) #5
        out = out.reshape(out.size(0), -1)
        out = self.relu5(out)
        
        out = self.fc1(out) # lin function
        out = self.relu_fc1(out)
        out = self.fc2(out)
        out = self.relu_fc2(out)
        out = self.fc3(out)
        out = self.relu_fc3(out)
        out = self.fc4(out)
        
        return out
    
    
# Model, critieria, learning rate , optimizer
model = CNNModel()
criterion = nn.MSELoss()
learning_rate = .0001
optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)

import os
# train the model
num_epochs =100
path= "alpha" # provide path to save the model

# Imp - Please select a threshold of MSE you want to stop the training at , else train for high number of epochs and select the best ones
threshold = 5 
# select threshold of MSE for stopping the training


for epoch in range(num_epochs):

    total_train_loss =0
    for i, (images, labels) in enumerate(train_loader):
   
        images = images.float()
        labels = labels.float()
        
        if torch.cuda.is_available():
          images = Variable(images.cuda())
        else:
          images = Variable(images)
            
        if torch.cuda.is_available():
          labels = Variable(labels.cuda())
        else:
          labels = Variable(labels)

        
        # Clear gradients w.r.t. parameters
        optimiser.zero_grad()
        outputs = model(images) #forward pass
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimiser.step() # updating parameters
        total_train_loss += loss.item()
        
    print("Epoch number : {} Training Loss : {} ".format( epoch+1,total_train_loss/len(train_loader))) 
       
    # after every epoch, testing on validation set
    total_val_loss = 0
    for i, (images, labels) in enumerate(val_loader):
        images = images.float()
        labels = labels.float()
      
        if torch.cuda.is_available():
          images = Variable(images.cuda())
        else:
          images = Variable(images)
        
        if torch.cuda.is_available():
          labels = Variable(labels.cuda())
        else:
          labels = Variable(labels)
        
        val_outputs = model(images)
        val_loss_size = criterion(val_outputs, labels)
        total_val_loss += val_loss_size.item()
    Validation_Loss = total_val_loss/len(val_loader)

    torch.save(model.state_dict(), os.path.join(path, 'epoch-{}.pth'.format(epoch))) #save model for each epoch
    
    if Validation_Loss<threshold:
        print("Finished training")
        break
        
        
# Testing  on validation set again with batch size = 1

import os
criterion = nn.MSELoss()
val_data = optical_flow_images[int(0.8*len(optical_flow_images)):]
val_labels = speed_Data_final[int(0.8*len(speed_Data_final)):]


transformations = transforms.Compose([transforms.ToTensor()])

val_data =   imagedataset(val_data,
                          val_labels,
                          transforms = transformations)

val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)

def MSE_val(model,val_loader):
    total_val_loss = 0
    pred =[]
    for i, (images, labels) in enumerate(val_loader):
        images = images.float()
        labels = labels.float()

        if torch.cuda.is_available():
          images = Variable(images.cuda())
        else:
          images = Variable(images)
        if torch.cuda.is_available():
          labels = Variable(labels.cuda())
        else:
          labels = Variable(labels)
        val_outputs = model(images)
        item = val_outputs.detach().numpy()
        pred.append(item)
        val_loss_size = criterion(val_outputs, labels)
        total_val_loss += val_loss_size.item()

    Validation_Loss = total_val_loss/len(val_loader)
    return Validation_Loss

# loading the last epoch model
the_model = CNNModel()



epoch = 10 # select best epoch with min. MSE value
the_model.load_state_dict(torch.load(os.path.join(path, 'epoch-{}.pth'.format(epoch))))
the_model.eval()
MSE_val = MSE_val(the_model,val_loader)
