---
title: "Working with custom datasets in PyTorch"
excerpt: "Pytorch dataloader tutorial for custom datasets where both inputs and labels are images"
categories:
    - Blog
    - Python
    - Pytorch
    - Deep Learning
   
tags:
    - Pytorch
    - Dataset
    - DataLoader
    - Deep Learning
    - Training 


toc: true
toc_label: "Contents of this post"
toc_icon: "cog"
---
Most PyTorch courses and tutorials show how to train a model using the pre-loaded datasets (such as MNIST) that subclass the `torch.utils.data.Dataset`. But in realistic scenarios, we have to train models on our own datasets and implement functions specific to them. Therefore, many of us don't know the underlying operations and functions to implement to get PyTorch to work with our own dataset.

PyTorch provides two data primitives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset` that allow you to use pre-loaded datasets as well as your own data. `Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the Dataset to enable easy access to the samples.

In this post, we will learn how to use the `Dataset` and `DataLoader` class. make subclasses from them to use for our custom dataset where samples and labels both are images. for classification training, we have samples as images and their class annotations in a CSV file (0 or 1). Pytorch official documentation provides a [tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
 to work with that sort of dataset

**Note:** In this particular example, i used [DRIVE](https://paperswithcode.com/dataset/drive#:~:text=The%20Digital%20Retinal%20Images%20for,including%207%20abnormal%20pathology%20cases.&text=to%2045%20degrees.-,Each%20image%20resolution%20is%20584*565%20pixels%20with%20eight%20bits,color%20channel%20(3%20channels).) dataset. DRIVE is a fundus images datset where samples are the retinal images and the labels are the corresponding segmentation map of retinal blood vessels (samples and labels are both images). With slight changes, this example can be used to load any type of dataset for training in pytorch
{: .notice--info}

# Subclassing `torch.utils.data.Dataset` to generate samples and labels

The whole code for making a dataset generator using `torch.utils.data.Dataset` that will be explained line by line:



## `Dataset` subclass:

```python

from torch.utils.data import Dataset
import os
import natsort
from PIL import Image
import numpy as np
import cv2

class CustomDataset(Dataset):
    """the class for loading the dataset from the directories
    Arguments:

        img_dir: directory for the dataset images

        label_dir: labels of the images directory

        transform: transforms applied to inputs

        transform_label: transforms applied to the labels
    """

    def __init__(self, img_dir, label_dir, 
                transform_image,transform_label,
                image_scale = None):

        self.image_scale = image_scale
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform_image = transform_image
        self.transform_label = transform_label
        all_images = os.listdir(self.img_dir)
        all_lables = os.listdir(self.label_dir)

        self.total_imgs = natsort.natsorted(all_images)
        self.total_labels = natsort.natsorted(all_lables)
    

    def __len__(self):
        return len(self.total_imgs)

    @classmethod
    def preprocessing(cls, image, label, scale=None):
        """class method for preprocessing of the image 
        and label
        usage: preprocessing the images before feeding
        to the network for training as well as before
        making predictions dataset class dishes out 
        pre-processed bathes of images and labels """
        return(image, label)


    def __getitem__(self, idx):
        """ Generator to yield a tuple of image and
        label
        idx: the index to iterate over the dataset in
        directories of both images and labels
        ---------------------

        :return:  image, label
        :rtype: torch tensor
        """

        img_loc = os.path.join(self.img_dir,
                         self.total_imgs[idx])
        label_loc = os.path.join(self.label_dir,
                         self.total_labels[idx])
        # opening image using cv2 function
        image = cv2.imread(img_loc)
        # opening image with PIL package
        label = Image.open(label_loc)  


        image, label = self.preprocessing(image, label,
                         mask, scale=self.image_scale)

        label = np.asarray(label).astype(np.uint8)


        '''=====applying transformations ======='''

        label = self.transform_label(label)
        image = self.transform_label(image)

        return image, label

```

## Writing the `CustomDataset` class:

import `torch.utils.data.Dataset` and all the other necessary packages according to  your data.

```python
from torch.utils.data import Dataset
import os
import natsort
from PIL import Image
import numpy as np
import cv2
```

# make a subclass from `Dataset` and initializing it

`__init__` function is a class constructor that is run once to initializing the class instance
 ```python

    def __init__(self, img_dir, label_dir, transform_image, transform_label, image_scale = None):

        self.image_scale = image_scale
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform_image = transform_image
        self.transform_label = transform_label
        all_images = os.listdir(self.img_dir)
        all_lables = os.listdir(self.label_dir)

        self.total_imgs = natsort.natsorted(all_images)
        self.total_labels = natsort.natsorted(all_lables)

 ```
 we are providing the image and labels directories along with separate transforms for images and labels as parameters to the class `__ini__` function.
 we list samples and labels from their respective directories and then sort them according to their names with `natsort`:
 ```python
        all_images = os.listdir(self.img_dir)
        all_lables = os.listdir(self.label_dir)

        self.total_imgs = natsort.natsorted(all_images)
        self.total_labels = natsort.natsorted(all_lables)
 ```
 these sorted lists will be used to get each individual image and its corresponding label (labels are also images for segmentation tasks).


## Breaking down `__getitem__`:
 `__getitem__` loads the images and labels and iterated through them using the `idx` index. in this function we can apply necessary transforms from `torchvision.transforms` or other necessary pre-processing steps with a withing this function (or using a separate class method)

```python
    def __getitem__(self, idx):

        img_loc = os.path.join(self.img_dir, self.total_imgs[idx])
        label_loc = os.path.join(self.label_dir, self.total_labels[idx])
        # opening image using cv2 function
        image = cv2.imread(img_loc)
        # opening image with PIL package
        label = Image.open(label_loc)  

        image, label = self.preprocessing(image, label, scale=self.image_scale)

        '''===============applying transformations ===================='''

        label = self.transform_label(label)
        image = self.transform_label(image)

        return image, label
```
i used the `cv2` as well as `PIL.Image` to load images from the label or image directory just to demonstrate the flexibility and how you can use either of the two you are most comfortable with.

```python
        img_loc = os.path.join(self.img_dir, self.total_imgs[idx])
        label_loc = os.path.join(self.label_dir, self.total_labels[idx])
```
we join paths of the image/label directory with the image/label list we sorted earlier.
for example: `total_images` is the list of all the sorted images in the `img_dir` and `idx` provides an index for each individual image so that we have a complete path to the image/label to be loaded.  

we can invoke the `preprocessing()` class method to perform some additional processing steps and then finally apply the transforms to the image and label and then return an image with its corresponding label.
## `__len__` method:
 returns the len of the sample size so that the `Dataset` class knows how the number of iterations to be performed to load the entire dataset.

## `preprocessing` method:
 a class method implemented which is exclusively not a part of the `torch.utils.data.Dataset` class but additionally added to the `CustomDataset` class as a class method. can be excluded if not needed.
 I intentionally left it empty however it can be used to perform some preprocessing using cv2 for example: resizing, grayscale conversion, applying CLAHE, gemma correction and other similar cv2 operations.

# Perparing  data for training with DataLoader:

## DataLoader to load `CustomDataset`:

```python
import CustomDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transforms

transform_label = transforms.Compose([transforms.ToTensor() ])

transform_image = transforms.Compose([transforms.ToTensor()])

dataset = CustomDataset(img_dir, label_dir, 
                        transform_image= transform_image,
                        transform_label=transform_label,
                        image_scale=.5)

n_val = int(len(dataset) * 0.2)
n_train = int(len(dataset) - n_val)
train, val = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train, batch_size=15, shuffle=True,
num_workers=0, pin_memory=False)

val_loader = DataLoader(val, batch_size=10, shuffle=False,
num_workers=0, pin_memory=False)

```

importing all the necessary packages
```python
import CustomDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transforms
```
transforms like random rotation, resize, random crop and other PyTorch transforms can be applied with `torchvision. transform. Here I only applied the tensor conversion that converts the data to a PyTorch tensor.  

```python
transform_label = transforms.Compose([transforms.ToTensor() ])

transform_image = transforms.Compose([transforms.ToTensor()])
```
Now the next step is to make an instance of our `CustomDataset` class with all the necessary parameters.
```python
dataset = CustomDataset(img_dir, label_dir, 
                        transform_image= transform_image,
                        transform_label=transform_label,
                        image_scale=.5)
```

we can split our data set into `train` and `validation` set using random split
```python
n_val = int(len(dataset) * 0.2)
n_train = int(len(dataset) - n_val)
train, val = random_split(dataset, [n_train, n_val])

```
make an iteratable generator from dataset for train and validation for training.
```python
train_loader = DataLoader(train, batch_size=15, shuffle=True,
num_workers=0, pin_memory=False)

val_loader = DataLoader(val, batch_size=10, shuffle=False,
num_workers=0, pin_memory=False)

```
# Iterate through the dataloader:

dataloader can be iterated and returns an image, label pair with the specified batch size.

for example, we defined our batch size to be 15 for `train_loader`. Considering our images are all grey-scaled (1 channelled) with a size of `512x512`, This will yield a PyTorch tensor of (15,1,512,512) 
with 15 samples each sample of 1 channel and 512 height and width

```python
for img, label in (train_loader):
    
    n,c,h,w =(img.shape)  #shape of tensor = (15,1,512,512)
    for i in range(n):
        im = np.squeeze(img[i, :, :, :].numpy())
        la = np.squeeze(label[i, :, :, :].numpy())
        visualize(im, la)


```
we can convert the tensors to NumPy arrays and plot them or use them straight for training just like pre-loaded datasets.


