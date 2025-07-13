#SEMI-SL 

Website is found [here](https://amitness.com/posts/fixmatch)
`It summarizes the FixMatch paper & offers code implementations.

The answer lies in a field called semi-supervised learning. FixMatch is a recent semi-supervised approach by _Sohn et al._ from Google Brain that improved the state of the art in semi-supervised learning(SSL). It is a simpler combination of previous methods such as UDA and ReMixMatch.

In this post, we will understand FixMatch and also see how it got 78% median accuracy and 84% maximum accuracy on CIFAR-10 with just 10 labeled images.


For weak augmentation, the paper uses a standard flip-and-shift strategy. It includes two simple augmentations:

**Random Horizontal Flip**
This augmentation is applied with a probability of 50%. This is skipped for the SVHN dataset since those images contain digits for which horizontal flip is not relevant. In PyTorch, this can be performed using [transforms](https://pytorch.org/vision/stable/transforms.html) as:
```bash
from PIL import Image
import torchvision.transforms as transforms

im = Image.open('dog.png')
weak_im = transforms.RandomHorizontalFlip(p=0.5)(im)
```

**Random Vertical and Horizontal Translation**
This augmentation is applied up to 12.5%. In PyTorch, this can be implemented using the following code where 32 is the size of the image needed:
``` bash
import torchvision.transforms as transforms
from PIL import Image

im = Image.open('dog.png')
resized_im = transforms.Resize(32)(im)
translated = transforms.RandomCrop(size=32, 
                                   padding=int(32*0.125), 
                                   padding_mode='reflect')(resized_im)
```

#### Strong Augmentation
These include augmentations that output heavily distorted versions of the input images. FixMatch applies either RandAugment or CTAugment and then applies CutOut augmentation.

**1. Cutout**

This augmentation randomly removes a square part of the image and fills it with gray or black color.

PyTorch doesn’t have a built-in implementation of Cutout but we can reuse its `RandomErasing` transformation to apply the CutOut effect.

```bash
import torch
import torchvision.transforms as transforms

# Image of 520*520
im = torch.rand(3, 520, 520)

# Fill cutout with gray color
gray_code = 127

# ratio=(1, 1) to set aspect ratio of square
# p=1 means probability is 1, so always apply cutout
# scale=(0.01, 0.01) means we want to get cutout of 1% of image area
# Hence: Cuts out gray square of 52*52
cutout_im = transforms.RandomErasing(p=1, 
                                     ratio=(1, 1), 
                                     scale=(0.01, 0.01), 
                                     value=gray_code)(im)
```

FixMatch uses one among two variants of AutoAugment:
**RandAugment**  
The idea of Random Augmentation(RandAugment) is very simple.

- First, you have a list of 14 possible augmentations with a range of their possible magnitudes.
- You select random N augmentations from this list. Here, we are selecting any two from the list.
- Then you select a random magnitude M ranging from 1 to 10. We can select a magnitude of 5. This means a magnitude of 50% in terms of percentage as maximum possible M is 10 and so percentage = 5/10 = 50%.
- Now, the selected augmentations are applied to an image in the sequence. Each augmentation has a 50% probability of being applied.

![[The Illustrated FixMatch for Semi-Supervised Learning.png]]

**CTAugment**  
CTAugment was an augmentation technique introduced in the ReMixMatch paper and uses ideas from control theory to remove the need for Reinforcement Learning in AutoAugment. Here’s how it works:

- We have a set of 18 possible transformations similar to RandAugment
- Magnitude values for transformations are divided into bins and each bin is assigned a weight. Initially, all bins weigh 1.
- Now two transformations are selected at random with equal chances from this set and their sequence forms a pipeline. This is similar to RandAugment.
- For each transformation, a magnitude bin is selected randomly with a probability according to the normalized bin weights
- Now, a labeled example is augmented with these two transformations and passed to the model to get a prediction
- Based on how close the model predictions were to the actual label, the magnitude bins weights for these transformations are updated.
- Thus, it learns to choose augmentations that the model has a high chance to predict a correct label and thus augmentation that fall within the network tolerance.


The paper uses wider and shallower variants of ResNet called [Wide Residual Networks](https://arxiv.org/abs/1605.07146) as the base architecture.

The exact variant used is Wide-Resnet-28-2 with a depth of 28 and a widening factor of 2. This model is two times wider than the ResNet. It has a total of 1.5 million parameters.

**Model training is also explained on the site**

For the 1 image per class tests, they did this test 4 times, those are the four datasets.
