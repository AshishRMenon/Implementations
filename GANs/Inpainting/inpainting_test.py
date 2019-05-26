#!/usr/bin/env python
# coding: utf-8

# In[ ]:



#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pathlib
from skimage import io
import scipy.io
from torch.autograd import Variable



#----------------------------------------------------------------------------------------#
batchSize = 64
loadSize = 350         
fineSize = 128         
Bottleneck = 4000      
nef = 64               
ngf = 64               
ndf = 64               
nc = 3                 
wtl2 = 0.999            
overlapPred = 4       
nThreads = 4           
niter = 500             
lr = 0.0002            
beta1 = 0.5            
display_id = 10        
display_iter = 50      
gpu = 1                
name = 'train1'        
manualSeed = 0         

conditionAdv = 0       
noiseGen = 0           
noisetype = 'normal'   
nz = 100     
xval=1          
#----------------------------------------------------------------------------------------#



random.seed(999)

# Root directory for dataset
dataroot = "/ssd_scratch/cvit/ashish/Imagenet2012/top25classes"

# Number of workers for dataloader
workers = 2

# Batch size during training
#batchSize=128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100


fineSize=128

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 4

wtl2=1
gamma=Bottleneck
nz_size=Bottleneck
noisegen=0
conditionAdv=0


dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.imsave('Input_sample.jpg',np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


g='/home/ashishmenon/labpc/inpainting/gmodel.pth'
fg=pathlib.Path(g)



class Generator(nn.Module):
    
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        
     #NET E   
        self.hidden0 = nn.Sequential(
        nn.Conv2d( nc, nef, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2,True) )
                                    
        self.hidden01=nn.Sequential(
        nn.Conv2d( nef, nef, 4, 2, 1, bias=False),
        nn.BatchNorm2d(nef),
        nn.LeakyReLU(0.2,True))

        # state size. (ngf*8) x 4 x 4
        self.hidden1=nn.Sequential(
        nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(nef * 2),
        nn.LeakyReLU(0.2,True)            )

        # state size. (ngf*4) x 8 x 8
        self.hidden2= nn.Sequential(
        nn.Conv2d( nef * 2,nef * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.LeakyReLU(0.2,True)      )

        # state size. (ngf*2) x 16 x 16
        self.hidden3=nn.Sequential(
        nn.Conv2d( nef * 4, nef * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(nef * 8),
        nn.LeakyReLU(0.2,True)     )

        # state size. (ngf) x 32 x 32
        self.hidden4=nn.Sequential(
        nn.Conv2d( nef * 8, Bottleneck, 4, bias=False))
        # state size. (nc) x 64 x 64
        
        
  #--------------------------------------------------#      
        #Net G_noise
        self.hidden_noise=nn.Sequential(
        nn.Conv2d( nz, nz, 1, 1, 0, bias=False))
  #---------------------------------------------------------------------------#      
       
        self.hiddenGP=nn.Sequential( nn.BatchNorm2d(gamma), nn.LeakyReLU(0.2, True))
    
        self.hiddenG0=nn.Sequential(
        nn.ConvTranspose2d( nz_size, ngf * 8, 4,bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True)              )

        self.hiddenG1=nn.Sequential(
        nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True)              )

        self.hiddenG2=nn.Sequential(
        nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True)              )
        
        self.hiddenG3=nn.Sequential(
        nn.ConvTranspose2d( ngf * 2, ngf , 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True)              )

        self.hiddenG301=nn.Sequential(
        nn.ConvTranspose2d( ngf , ngf , 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True)              )

        self.hiddenG4=nn.Sequential(
        nn.ConvTranspose2d( ngf , nc , 4, 2, 1, bias=False),
        nn.Tanh()      )



    def forward(self,x):
        x1=self.hidden0(x[0])
        if fineSize==128:
            x1=self.hidden01(x1)
        x1=self.hidden1(x1)
        x1=self.hidden2(x1)
        x1=self.hidden3(x1)
        x1=self.hidden4(x1)
        if noiseGen:
            xnoise=self.hidden_noise(x[1])
            y= [ x1 , xnoise]
            z=torch.cat([y[0],y[1]])
            gamma=Bottleneck+nz
            z=self.hiddenGP(z)
            nz_size = Bottleneck+nz
        else:
            gamma=Bottleneck
            z=self.hiddenGP(x1)
            nz_size= Bottleneck
        z=self.hiddenG0(z)
        z=self.hiddenG1(z)
        z=self.hiddenG2(z)
        z=self.hiddenG3(z)
        #if fineSize == 128:
        #    z=self.hiddenG301(z)
        z=self.hiddenG4(z)
        
        return z



netG = Generator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.load_state_dict(torch.load(g))
netG.eval()




if noisetype == 'uniform':
    noise_vis=2*torch.rand(batchSize, nz, 1, 1, device=device)-1
if noisetype == 'normal':
    noise_vis=torch.randn(batchSize, nz, 1, 1, device=device)




image_ctx=real_batch[0].to(device)

real_center = image_ctx[:,:,int(fineSize/4):int(fineSize/2) + int(fineSize/4),int(fineSize/4):int(fineSize/2) + int(fineSize/4)].clone()

image_ctx[:,0,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 2*117.0/255.0 - 1.0

image_ctx[:,1,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 2*104.0/255.0 - 1.0

image_ctx[:,2,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 2*117.0/255.0 - 1.0

input_image_ctx=image_ctx.clone()
input_image_ctx=input_image_ctx.to(device)



if noiseGen:
    noise=noise
    pred_center = netG([input_image_ctx,noise])
else:
    pred_center = netG([input_image_ctx])



image_ctx[:,:,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4)+overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]=pred_center[:,:,overlapPred:int(fineSize/2) - overlapPred,overlapPred:int(fineSize/2) - overlapPred].clone()




input_image_ctx[:,0,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 1

input_image_ctx[:,1,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 1

input_image_ctx[:,2,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 1

plt.imsave('output_sample1.jpg',np.transpose(vutils.make_grid(input_image_ctx, padding=2, normalize=True).cpu(),(1,2,0)))

plt.imsave('output_sample2.jpg',np.transpose(vutils.make_grid(image_ctx.detach(), padding=2, normalize=True).cpu(),(1,2,0)))


