import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 205810963

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = 4
        self.n = n
        kernel_size = 5
        padding = int((kernel_size - 1) / 2)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.BN1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(in_channels=n,out_channels=2*n,kernel_size=kernel_size,padding=padding)
        self.BN2 = nn.BatchNorm2d(2*n)
        self.conv3 = nn.Conv2d(in_channels=2*n,out_channels=4*n,kernel_size=kernel_size,padding=padding)
        self.BN3 = nn.BatchNorm2d(4*n)
        self.conv4 = nn.Conv2d(in_channels=4*n,out_channels=8*n,kernel_size=kernel_size,padding=padding)
        self.BN4 = nn.BatchNorm2d(8*n)
        self.fc1 = nn.Linear(8*n * 28 * 14, 100)
        self.fc2 = nn.Linear(100, 2)


    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''

        inp = self.conv1(inp)
        inp = self.BN1(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)
   
        inp = self.conv2(inp)
        inp = self.BN2(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2) 

        inp = self.conv3(inp)
        inp = self.BN3(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)  
        
        inp = self.conv4(inp)
        inp = self.BN4(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)  
        

        inp = inp.reshape(-1, 8*self.n*28*14)
        inp = self.fc1(inp)
        inp = F.relu(inp)
        inp = self.fc2(inp)
        inp = F.log_softmax(inp, dim=1)
        return inp



class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        n = 8
        self.n = n
        kernel_size = 5
        padding = int((kernel_size - 1) / 2)
        self.conv1 = nn.Conv2d(in_channels=6,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.BN1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(in_channels=n,out_channels=2*n,kernel_size=kernel_size,padding=padding)
        self.BN2 = nn.BatchNorm2d(2*n)
        self.conv3 = nn.Conv2d(in_channels=2*n,out_channels=4*n,kernel_size=kernel_size,padding=padding)
        self.BN3 = nn.BatchNorm2d(4*n)
        self.conv4 = nn.Conv2d(in_channels=4*n,out_channels=8*n,kernel_size=kernel_size,padding=padding)
        self.BN4 = nn.BatchNorm2d(8*n)
        self.fc1 = nn.Linear(8*n * 14 * 14, 100)
        self.fc2 = nn.Linear(100, 2)
        # TODO: complete this method

    # TODO: complete this class
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        inp = torch.cat((inp[:,:,:224,:], inp[:,:,224:,:]), dim=1)
        # TODO start by changing the shape of the input to (N,6,224,224)
        inp = self.conv1(inp)
        inp = self.BN1(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)
   
        inp = self.conv2(inp)
        inp = self.BN2(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2) 

        inp = self.conv3(inp)
        inp = self.BN3(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)  
        
        inp = self.conv4(inp)
        inp = self.BN4(inp)
        inp = F.relu(inp)
        inp = F.max_pool2d(inp, kernel_size=2)  
        

        inp = inp.reshape(-1, 8*self.n*14*14)
        inp = self.fc1(inp)
        inp = F.relu(inp)
        inp = self.fc2(inp)
        inp = F.log_softmax(inp, dim=1)
        return inp
