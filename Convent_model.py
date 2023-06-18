import torch.nn as nn
import torch.nn.functional as F

class Convnet(nn.Module):
    
    def __init__(self):
        super(Convnet, self).__init__()
        
        
        self.conv1=nn.Conv2d(1,50,kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
   
        
        
        self.conv2 = nn.Conv2d(50,60, kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
   
        
        
        self.conv3 = nn.Conv2d(60, 80,  kernel_size = 3)

       
        
        
        self.batch_norm1 = nn.BatchNorm2d(50)
        self.batch_norm2 = nn.BatchNorm2d(60)
  
        self.fc1 = nn.Linear(80*2*2, 250) 
        self.fc2 = nn.Linear(250, 25)
        
        
    def forward(self,x):
        x=self.conv1(x)
        x = self.batch_norm1(x)
        x=F.relu(x)
        x=self.pool1(x)
        
        x=self.conv2(x)
        x = self.batch_norm2(x)
        x=F.relu(x)
        x=self.pool2(x)
        
        x=self.conv3(x)
        x=F.relu(x)
        
        x = x.view(-1,80*2*2)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = F.log_softmax(x, dim=1)
        
        return x  