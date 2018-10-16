## Initial 

This readme is mainly explaining exp001 Variational AutoEncoder structure.
To have a general idea on how to setup this auto-encoder visit ~/src/galmed/cxr-vae/README.md


##Title

VAE-32 with only four pakcages of data
## Structure

We are Using a Variational AutoEncoder.
The structure is defined in "VAE Model" section:

(input/output format example: 128x256x1 means a width of 128, height of 256, and 1 channel)
(Filter format example: 5x5x1x64 means a 5x5 filter with 1 input channel and 64 output channels)

Input: images of size 256x256 

Encoder:
- conv1: input=image, output=128x128x64 , filter=5x5x1x64 , stride=2, activation=elu, zero-padding
- conv2: input=128x128x64, output=64x64x128, filter=5x5x64x128, stride=2, activation=elu, zero-padding
- conv3: input=64x64x128, output=32x32x256, filter=5x5x128x256, stride=2, activation=elu, zero-padding
- fc1: input=32x32x256, output=64
- gaussian sample: input=64, output=32

Decoder:
- fc1: input=32, output=32x32x256, filter=5x5x256x128, stride=2, activation=elu, zero-padding
- deconv1 : input=32x32x256, output=64x64x128, filter=5x5x256x128, stride=2, activation=elu, zero-padding, (includes dropout)
- deconv2 : input=64x64x128, output=128x128x64, filter=5x5x128x64, stride=2, activation=elu, zero-padding, (includes dropout)
- deconv3 : input=128x128x64, output=256x256x32, filter=5x5x64x32, stride=2, activation=elu, zero-padding, (includes dropout)
- deconv4 (logits): input=256x256x32, output=256x256x1, filter=5x5x32x1, stride=1, activation=elu, zero-padding, (includes dropout)

Output: images of size 256x256 

Loss function: KL Divergance + Regularization (L1 Norm)


## Visualization

In order to visualize our model TensorBoard can be used. 
Run tensorboard.sh file with the "port" and "event root directory" arguments and then tunnel through this port to access Tensorboard.


