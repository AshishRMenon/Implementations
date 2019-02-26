**GAN's** use the convolutional Neural Network for this task in each of the Generator and the Discriminator.

**Discriminator** has a

*   First layer made of (nc, ndf, 5, 1, 0) here nc=3 channels, ndf=Size of feature maps in discriminator(64), 5X5 filter, stride 2 and padding=0)

*   Second layer made of (ndf, ndf * 2, 5, 2, 0)

*   Third layer made of (ndf * 2, ndf * 4, 5, 2, 0)

*   Fourth layer made of (ndf * 4, ndf * 8, 5, 2, 0)

*   Final layer made of (ndf * 8, 1, 5, 2, 0)

*   OptimizerD = optim.Adam(Discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

**Generator** has a

*   First layer made of (nz,ngf * 8, 5, 2, 0) here nz=Size of latent vector (100),ngf=Size of feature maps in generator(64), 5X5 filter, stride 2 and padding=0)

*   Second layer made of (ngf * 8, ngf * 4, 5, 2, 0)

*   Third layer made of (ngf * 4, ngf * 2, 5, 2, 0)

*   Fourth layer made of (ngf * 2, ngf, 5, 2, 0)

*   Final layer made of (ngf, nc, 5, 1, 0)

*   OptimizerG = optim.Adam(Generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

**Number of epochs** = 50
