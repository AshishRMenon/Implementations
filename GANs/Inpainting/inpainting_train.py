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
dataroot = "/ssd_scratch/cvit/ashish/Imagenet2012/train"

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
ngpu = 2


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



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




g='/home/ashishmenon/labpc/GAN_KAN/DCGAN/gmodel.pth'
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

if not(fg.exists ()):
    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
if (fg.exists ()):
    state_dict=torch.load(g)
    print('-----------------------------------------------------------------')
    print(state_dict.keys())
    print('-----------------------------------------------------------------')
    Generator(ngpu).load_state_dict(state_dict)
    netG = Generator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

print(netG)



d='/home/ashishmenon/labpc/GAN_KAN/DCGAN/dmodel.pth'
fd=pathlib.Path(d)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.D_ctx = nn.Sequential(nn.Conv2d(nc,ndf,5,2,2,bias=False))
        self.D_pred = nn.Sequential(nn.Conv2d(nc,ndf,5,2,2+32,bias=False))
        self.residual=nn.Sequential(nn.LeakyReLU(0.2,True))
        self.D1 =nn.Sequential(nn.Conv2d(ndf*2, ndf,4,2,1,bias=False),
                               nn.BatchNorm2d(ndf),
                               nn.LeakyReLU(0.2,True))

        self.D2=nn.Sequential(nn.Conv2d(nc, ndf,4,2,1,bias=False),nn.LeakyReLU(0.2,True))
 
        self.D3=nn.Sequential(nn.Conv2d(ndf, ndf,4,2,1,bias=False),nn.LeakyReLU(0.2,True))
        self.D4 = nn.Sequential(
        # input is (nc) x 64 x 64
        nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf *2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 32
        nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 16 x 16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        nn.Conv2d(ndf * 8, 1, 4, bias=False),
        nn.Sigmoid()
    )
    

    def forward(self, input):
        if conditionAdv:
            x1=self.D_ctx(input[0])
            x2=self.D_pred(input[1])
            y=[x1,x2]
            z=torch.cat([y[0],y[1]],0)
            z=self.residual(z)
            z=self.D1(z)
        else:
            z=self.D2(input[0])
            
        #if fineSize==128:
        #    z=self.D3(z)
        z=self.D4(z)
        return z
    
            
if not (fd.exists()):
    netD = Discriminator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)


if fd.exists():
    state_dict=torch.load(d)
    print('-----------------------------------------------------------------')
    print(state_dict.keys())
    print('-----------------------------------------------------------------')
    Discriminator(ngpu).load_state_dict(state_dict)
    netD = Discriminator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

print(netD)



criterion = nn.BCELoss()

if wtl2!=0:
    criterionMSE=nn.MSELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



mask_global = torch.ByteTensor(batchSize,fineSize,fineSize)
#input_ctx_vis = torch.Tensor(batchSize, nc, fineSize,fineSize,device=device)
#input_ctx = torch.Tensor(batchSize, nc, fineSize,fineSize,device=device)
#input_center = torch.Tensor(batchSize, nc, fineSize,fineSize,device=device)


img_list = []
G_losses = []
D_losses = []
iters = 0

#noise = torch.Tensor(batchSize, nz, 1, 1,device=device)
#label = torch.Tensor(batchSize,device=device)



#----------------------------Random Pattern--------------------------------------------------#

# res = 0.06 
# density = 0.25
# MAX_SIZE = 10000

# low_pattern = torch.rand(int(res*MAX_SIZE), int(res*MAX_SIZE))
# low_pattern=low_pattern*255
# pattern1=transforms.ToPILImage()
# pattern2 = transforms.Scale(MAX_SIZE,interpolation=3)
# pattern=pattern2(pattern1(low_pattern))
# low_pattern = torch.zeros(int(res*MAX_SIZE), int(res*MAX_SIZE))
# fineSize=128

# op3=transforms.ToTensor()
# pattern=op3(pattern)
# pattern=torch.reshape(pattern,(MAX_SIZE,MAX_SIZE))
# pattern = torch.lt(pattern,density)

#----------------------------Random Pattern--------------------------------------------------#


if noisetype == 'uniform':
    noise_vis=2*torch.rand(batchSize, nz, 1, 1, device=device)-1
if noisetype == 'normal':
    noise_vis=torch.randn(batchSize, nz, 1, 1, device=device)



print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    j=0
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_ctx = data[0].to(device)
        
        b_size = real_ctx.size(0)
        
 #--------------------------------------------------------------------------------------------------------#       

#Random Patch 
    
#         real_center = real_ctx
#         input_center=real_center.clone()
#         if wtl2!=0:
#             input_real_center=real_center.clone()
#         while(True):
#             x = np.random.randint(1,MAX_SIZE-fineSize)
#             y = np.random.randint(1,MAX_SIZE-fineSize)
#             mask = pattern[y:y+fineSize-1,x:x+fineSize-1]   
#             area = torch.sum(mask)*100/(fineSize*fineSize)
#             #area = torch.sum(mask)
#             if (area>20 and area<30):
#                 break

#         mask_global=mask.repeat(batchSize,1,1)
        
        
#         real_ctx[{{},{1},{},{}}][mask_global] = 2*117.0/255.0 - 1.0
#         real_ctx[{{},{2},{},{}}][mask_global] = 2*104.0/255.0 - 1.0
#         real_ctx[{{},{3},{},{}}][mask_global] = 2*123.0/255.0 - 1.0

#         Random patch
#         real_ctx(1,0,1).masked_fill_(mask_global, 2*117.0/255.0 - 1.0)
#         real_ctx(1,1,1).masked_fill_(mask_global, 2*104.0/255.0 - 1.0)
#         real_ctx(1,2,1).masked_fill_(mask_global, 2*117.0/255.0 - 1.0)
        
#--------------------------------------------------------------------------------------------#        
        
        #Square Patch
        real_center = real_ctx[:,:,int(fineSize/4) : int(fineSize/2) + int(fineSize/4),int(fineSize/4) : int(fineSize/2) + int(fineSize/4) ].clone()
        
#         real_ctx(1,0,1 + fineSize/4 + overlapPred, fineSize/2 + fineSize/4 - overlapPred)= 2*117.0/255.0 - 1.0
#         real_ctx(1,1,1 + fineSize/4 + overlapPred, fineSize/2 + fineSize/4 - overlapPred)= 2*104.0/255.0 - 1.0
#         real_ctx(1,2,1 + fineSize/4 + overlapPred, fineSize/2 + fineSize/4 - overlapPred)= 2*117.0/255.0 - 1.0
        
        real_ctx[:,0,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 2*117.0/255.0 - 1.0
        real_ctx[:,1,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 2*104.0/255.0 - 1.0
        real_ctx[:,2,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 2*123.0/255.0 - 1.0
    
#--------------------------------------------------------------------------------------------#        
        print('CP1')
        input_center=real_center.clone()
        input_ctx=real_ctx.clone()
        realclone=real_ctx.clone()
        input_ctx=Variable(input_ctx.to(device))
        input_center=Variable(input_center.to(device))
        if wtl2!=0:
            input_real_center=real_center.clone()
   
        label = torch.full((b_size,), real_label, device=device)
        if conditionAdv:
            output = netD([input_ctx,input_center]).view(-1)
        else:
            output = netD([input_center])
        errD_real = criterion(output, label)
        print('gradient1_cal')
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        if noisetype == 'uniform':
            noise=2*torch.rand(b_size, nz, 1, 1, device=device)-1
            
        if noisetype == 'normal':
            noise=torch.randn(b_size, nz, 1, 1, device=device)
   
        if noiseGen:
            noise=Variable(noise)
            fake = netG([input_ctx,noise])
        else:
            fake = netG([input_ctx])
        # Generate fake image batch with G
        input_center=fake.clone()
        
        label.fill_(fake_label)
        
        # Classify all fake batch with D
        
        if conditionAdv:
            output = netD([input_ctx,input_center])
        else:
            output = netD([input_center])
        
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        print('gradient2_cal')
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
#         noise = torch.randn(b_size, nz, 1, 1, device=device)
#         fake = netG(noise)
        
        if noisetype == 'uniform':
            noise=2*torch.rand(b_size, nz, 1, 1, device=device)-1
            
        if noisetype == 'normal':
            noise=torch.randn(b_size, nz, 1, 1, device=device)
   
        if noiseGen:
            noise=Variable(noise)
            fake = netG([input_ctx,noise])
        else:
            fake = netG([input_ctx])
        # Generate fake image batch with G
        input_center=fake.clone()
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD([input_center]).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        errG_total = errG
        if wtl2!=0:
            errG_l2 = criterionMSE(input_center, input_real_center)
            if overlapPred==0:
                if (wtl2>0 and wtl2<1):    
                    errG_total = (1-wtl2)*errG + wtl2*errG_l2
                else:
                    errG_total = errG + wtl2*errG_l2
            
            else:
                
                if (wtl2>0 and wtl2<1):
                    errG_total = (1-wtl2)*errG + wtl2*errG_l2
                else:
                    errG_total = errG + wtl2*errG_l2
            
#         if noiseGen:
#             netG:backward([input_ctx,noise], df_dg)
#         else:
#             netG:backward([input_ctx], df_dg)
        
        
        
        # Calculate gradients for G
        print('gradient3_cal')
        errG_total.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        input_ctx_vis=real_ctx.clone()
        input_ctx_vis=Variable(input_ctx_vis.to(device))

        with torch.no_grad():
            if noiseGen:
                fake = netG([input_ctx_vis,noise_vis]).detach().cpu()
                    
            else:
                fake = netG([input_ctx_vis]).detach().cpu()
        real_ctx[:,:,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]=fake[:,:,overlapPred:int(fineSize/2) - overlapPred,overlapPred:int(fineSize/2) - overlapPred].clone()
#         disp.image(fake, {win=display_id, title=name})
#         disp.image(real_center, {win=display_id * 3, title=name})
#         disp.image(real_ctx, {win=display_id * 6, title=name})

        
        print('CHeck print status of error values')
        if xval==1:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG_total.item(), D_x, D_G_z1, D_G_z2))
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        realclone[:,0,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 1

        realclone[:,1,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 1

        realclone[:,2,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred,int(fineSize/4) + overlapPred:int(fineSize/2) + int(fineSize/4) - overlapPred]= 1
 
        
        if (errG.item()<8 and errD.item()>0.2):

            j+=1   
#             if os.path.exists('/ssd_scratch/cvit/ashish/normal_output/epoch{}'.format(epoch)):
#                 io.imsave(('/ssd_scratch/cvit/ashish/normal_output/epoch{}/out{}.png').format(epoch,j),np.transpose(vutils.make_grid(fake[:64],padding=2,normalize=True),(1,2,0)))
#                 io.imsave(('/ssd_scratch/cvit/ashish/normal_output/epoch{}/in{}.png').format(epoch,j),np.transpose(vutils.make_grid(data[0].to(device)[:64],padding=5,normalize=True).cpu(),(1,2,0)))

#             else:
#                 os.makedirs('/ssd_scratch/cvit/ashish/normal_output/epoch{}'.format(epoch))
#                 io.imsave(('/ssd_scratch/cvit/ashish/normal_output/epoch{}/out{}.png').format(epoch,j),np.transpose(vutils.make_grid(fake[:64],padding=2,normalize=True),(1,2,0))) 
#                 io.imsave(('/ssd_scratch/cvit/ashish/normal_output/epoch{}/in{}.png').format(epoch,j),np.transpose(vutils.make_grid(data[0].to(device)[:64], padding=5,normalize=True).cpu(),(1,2,0))) 
            
            
            torch.save(netG.state_dict(),'gmodel.pth')
            torch.save(netD.state_dict(),'dmodel.pth')
            plt.imsave('train_output_sample1.jpg',np.transpose(vutils.make_grid(real_ctx, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.imsave('train_input_sample1.jpg',np.transpose(vutils.make_grid(realclone, padding=2, normalize=True).cpu(),(1,2,0)))

             
            
        iters += 1
        




plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Outputcurve.png')





real_batch = next(iter(dataloader))


