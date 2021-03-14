NeuralStyleTransfer
======================
A Neural Style Transfer based on VGG19 model

:star: Star this project on GitHub — it helps!

[Neural Style Transfer](https://arxiv.org/abs/1609.04802) is a task of transferring style
of one image to another. It does it by using features of some pretrained model. In this
case as such **Base Model** the **VGG19** pretrained on **ImageNet** was used. 
Firstly we create our own model from certain layers of the **VGG19** network.
And then by adding gradients from the network to the input image we obtain our result image
with transferred style.


## Table of content

- [Compiling model](#compile)
- [Training](#train)
- [Results](#res)
- [License](#license)
- [Links](#links)

## Compiling model

As mentioned above, first of all we should compile our model from pretrained one.
In this particular case the **VGG19** was used. We should define between which of
the layers the `Content loss` and `Style loss` are going to be calculated.
As model's input is going to be the copy of *content_image* we do not need so much
*nodes* to calculate `Content loss` as we need for `Style loss`(In this case **1 node**
was used for `Content loss` and **5 nodes** for `Style loss`.
* Model compiler is under `model/__init__.py`.

## Training

### Database

The Super-Resolution GAN was trained on **STL10** dataset from `torchvision.dataset`.

### WarmUp of Generator

Before to train both **Generator** and **Discriminator** we should pretrain our **Ganarator** on
dataset with **Pixel-wise Loss** function.

![](images/g_loss_warmup.png)

See [Super-Resolution [GAN WarmUp]](https://github.com/akanametov/SuperResolution/blob/main/demo/SuperResolution%5BGeneratorWarmUp%5D.ipynb) for **Generator**'s warmup.

### Training with Discriminator

After **Generator** warmup we train booth **Generator** and **Discriminator** with their loss functions.
The **Generator loss** consists of **Adverserial loss**(BCE loss between *fake prediction and target*),
**Model Based loss**(feature based MSE loss between *fake and real images*) and **Pixel-wise loss**(MSE loss between *fake and real images*).

![Generator loss](images/g_loss.png)

![Discriminator loss](images/d_loss.png)

**After 100 epochs of training:**

<a>
    <img src="images/train.png" align="center" height="400px" width="400px"/>
</a>


See [Super-Resolution](https://github.com/akanametov/SuperResolution/blob/main/demo/SuperResolution.ipynb) for **SR-GAN**'s training.

## License

This project is licensed under MIT.

## Links

* [Super-Resolution GAN (arXiv article)](https://arxiv.org/abs/1609.04802)
