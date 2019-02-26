The Data Set can be found using the following Link http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

# FACE DATA SET

**GAN's** use the convolutional Neural Network for this task in each of the Generator and the Discriminator.

**Discriminator** has a

*   First layer made of (nc, ndf, 4, 2, 1) here nc=3 channels, ndf=Size of feature maps in discriminator(64), 4X4 filter, stride 2 and padding=1)

*   Second layer made of (ndf, ndf * 2, 4, 2, 1)

*   Third layer made of (ndf * 2, ndf * 4, 4, 2, 1)

*   Fourth layer made of (ndf * 4, ndf * 8, 4, 2, 1)

*   Final layer made of (ndf * 8, 1, 4, 1, 0)

*   OptimizerD = optim.Adam(Discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

**Generator** has a

*   First layer made of (nz,ngf * 8, 4, 1, 0) here nz=Size of latent vector (100),ngf=Size of feature maps in generator(64), 4X4 filter, stride 2 and padding=1)

*   Second layer made of (ngf * 8, ngf * 4, 4, 2, 1)

*   Third layer made of (ngf * 4, ngf * 2, 4, 2, 1)

*   Fourth layer made of (ngf * 2, ngf, 4, 2, 1)

*   Final layer made of (ngf, nc, 4, 2, 1)

*   OptimizerG = optim.Adam(Generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

**Loss Function**: Binary Cross Entropy Loss
