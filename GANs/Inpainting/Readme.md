## PAPER_TITLE: 
### Context Encoders: Feature Learning by Inpainting

## Authors:
Deepak Pathak, Philipp Krähenbühl, Jeff Donahue, Trevor Darrell

University of California, Berkeley Alexei A. Efros


## Objective:
This file shows the codes and results of GAN's used for Context based Region filling, where we make use of an encoder for getting a vector representation of image with missing region.


## Summary:

Context Encoders:Is a convolutional neural network trained to generate the contents of an arbitrary image region conditioned on its surroundings. In order to succeed at this task, context encoders need to both understand the content of the entire image, as well as produce a plausible hypothesis for the missing part(s). When training context encoders, the authors have experimented with both a standard pixel-wise reconstruction loss, as well as a reconstruction plus an adversarial loss. The latter produces much sharper results because it can better handle multiple modes in the output. We found that a context encoder learns a representation that captures not just
appearance but also the semantics of visual structure

The overall architecture is a simple encoder-decoder pipeline. The encoder takes an input image with missing regions and produces a latent feature representation of that image. The decoder takes this feature representation and produces the missing image content 

