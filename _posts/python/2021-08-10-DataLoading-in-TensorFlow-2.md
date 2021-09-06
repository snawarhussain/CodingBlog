---
title: "Different ways to load custom dataset in TensorFlow 2 for classification"
excerpt: "Tensorflow2: preparing and loading custom datasets"
categories:
    - Blog
    - TensorFlow
   
tags:
    - TensorFlow
    - Keras
    - ImageDataGenerator
    - Deep Learning
    - Training 

toc: true
toc_label: "Contents of this post"
toc_icon: "cog"
---



---
With the release of TensorFlow 2 and Keras being the default frontend for it. there is mass confusion on which tutorial to follow to work with custom datasets in TensorFlow 2 since Keras provides their documentation while TensorFlow official website has its own guide to follow and to be honest, none of them is user friendly and just adds to the confusion, particularly, if you are switching from another framework like PyTorch or just have been out of touch with TensorFlow for a long time.


There is no unified way of creating a custom dataset for training in TensorFlow, rather it depends on the type of dataset you are working with. TensorFlow is quite flexible in this regard and you can feed data in a number of ways to the model for training and evaluation.

# Model.fit()  method
first we need to understand what kind of data can be fed to the `tf.keras.Model.fit()` 

According to the official documentation, the fit() method can work with several data types.

`fit()` method accepts inputs `x` and targets `y`. input `x` could be: 
* A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
* A TensorFlow `tensor`, or a list of tensors (in case the model has multiple inputs).
* A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
* A `tf.data dataset`. Should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights).
* A generator or `keras.utils.Sequence` returning `(inputs, targets)` or `(inputs, targets, sample_weights)`.


and targets `y` :
	
* Like the input data `x`, it could be either `Numpy array(s)` or `TensorFlow tensor(s)`. **It should be consistent with `x`** (you cannot have Numpy inputs and tensor targets, or inversely).

* If `x` is a dataset, generator, or `keras.utils.Sequence` instance, `y` should not be specified (since targets will be obtained from `x`)

# Using Numpy and cv2 to load classification data

---


In this example, we will load image classification data for both training and validation using NumPy and cv2. you need to get comfortable using python operations like os.listdir, enumerate to loop through directories and search for files and load them iteratively and save them in an array or list.
for a binary classification task,  the image dataset should be structured in the following way:

```js
/
  training/
          -Class_A/
          -Class_B/

  testing/
        -Class_A/
        -Class_B/

```
if the image data directory is not in this structure we can make new directories and randomly split the data into training and testing sets using code this comes in handy especially if you are dealing with a large dataset.
let's go ahead and write code for this step.





## Creating new directories for the dataset 
in this example, I am using an image dataset of healthy and glaucoma infested fundus images. source directory has two folders namely `healthy` and `glaucoma` that have images. we need to create training and testing directories for both classes of  `healthy` and `glaucoma` images. and randomly split a portion of data between the training and testing set. for that first we need to make new directories for our dataset to reside for that we will execute the following code:


```python
import os

to_create = {
    'root': '/content/dataset',
    'train_dir': '/content/dataset/training',
    'test_dir': '/content/dataset/testing',
    'healthy_train_dir': '/content/dataset/training/healthy',
    'glaucoma_train_dir': '/content/dataset/training/glaucoma',
    'healthy_test_dir': '/content/dataset/testing/healthy',
    'glaucoma_test_dir': '/content/dataset/testing/glaucoma'
}

for directory in to_create.values():
    try:
        os.mkdir(directory)
        print(directory, 'created')         #iterating through dictionary to make new dirs
    except:
        print(directory, 'failed')
```

    /content/dataset created
    /content/dataset/training created
    /content/dataset/testing created
    /content/dataset/training/healthy created
    /content/dataset/training/glaucoma created
    /content/dataset/testing/healthy created
    /content/dataset/testing/glaucoma created


## splitting and randomly sampling the data into test and train sets
after creating directories, the next step is to split the data into portions for training and testing. from the whole data, we will randomly sample a portion for the train and the rest for the test.

for that, we will iterate through the files in each class namely `healthy` and `glaucoma` one by one and list all the file names and paths. after that, we will first shuffle and then slice them into portions for training and testing and copy these files  to the training and testing directories we created in previous step  

the function `split_data()` take a source path to files, and after making a list of all the files, shuffles them and copy a portion defined by `split_size` in  `Train_path` and `Test_path`  




```python
from shutil import copyfile
import random

def split_data(SOURCE,Train_path, Test_path, split_size):
  all_files =[]
  for image in os.listdir(SOURCE):            #os.listdir lists all the files/folders in a directory 
    image_path = os.path.join(image, SOURCE)      # joining the root dir with the image name to get the path to the image
    if os.path.getsize(image_path):           #os.path.getsize returns the size of a file in bytes, here it is being used to check if file has size greater than 0 or not 
      all_files.append(image)
    else:
      print('{} has zero size, skipping'.format(image))

  total_files = len(all_files)
  split_point = int(total_files * split_size)                         
  shuffled = random.sample(all_files, total_files)    #sample n number of files randomly from the given list of files 
  train = shuffled[:split_point]                  #slicing from start to split point 
  test = shuffled[split_point:]
  for image in train:                             #copy files from one path to another 
    copyfile(os.path.join(SOURCE, image ), os.path.join(Train_path, image))

  for image in test:
    copyfile(os.path.join(SOURCE, image ), os.path.join(Test_path, image))

split_data('/content/data/glaucoma',to_create.get('glaucoma_train_dir'), to_create.get('glaucoma_test_dir'), 0.8)
split_data('/content/data/healthy',to_create.get('healthy_train_dir'), to_create.get('healthy_test_dir'), 0.8)
```

# Loading data in TensorFlow 
once we have our data in desired structure and randomly split. now comes the important part where we need to load it in Tensorflow.
from here we can do it in two ways without `numpy` and probably `cv2` that are pretty straightforward and intuitive. 


1.   iterate through train and test dir and load the data into NumPy array and feed the model these array as `x` inputs and `y` targets
2.   after getting numpy arrays from 1. use them to create a tensor dataset with `from_tensor_slices()` method of TensorFlow. with this approach we can create batches of our data that are randomly shuffled on each epoch for the training.
3. and the 3rd and the easiest way, we can load tensors of data directory using the `flow_from_directory()` method. that automatically detects the classes of data for classification and generates batches.

here I will demonstrate the number 2 and 3 method for loading data.







# Cv2 and NumPy to load data in TensorFlow
for this method we will loop through the directories to list all images and their paths just like we did before while splitting the data.
and then create NumPy array from images and their labels.


```python
import cv2
import numpy as np
def data_load(root_path, scale=(256,256)):
  categories =  os.listdir(root_path) 
  x = []
  y =[]
  for i, cat in enumerate(categories):
    img_path = os.path.join(root_path, cat)
    images = os.listdir(img_path)
    for image in images:
      img = cv2.imread(os.path.join(img_path, image), 0)
      img = cv2.resize(img, scale)
      x.append(img)
      y.append(i)
  return np.array(x), np.array(y)
x_train, y_train = data_load(to_create.get('train_dir'))
x_test, y_test = data_load(to_create.get('test_dir'))
```


```python
print ( ' trainset has length of {}'.format(len(x_train)))
print ( ' testset has length of {}'.format(len(x_test)))
```

     trainset has length of 24
     testset has length of 6


each item in the `x`_train ndarray is an array of image while `y` is array of labels where each label is either 0 or 1 for glaucoma and healthy class respectively.

the shape of `x` is `(24, 256, 256)` which indicates that we have 24 images of size (256, 256) each.
to feed this input data to our TensorFlow CNN, we need to add an extra dimension for channels, since we are using gray images, we just need to add another dimension of size 1 to the data. we can do that using numpy's `expand_dims()` function. 

**NOTE:** By default, TensorFlow uses channel last data format, which means the data should be in the format of (B, H, W, C) with batch(B), height(H), width(W) and finally channels(C) at the last position for the input tensor so we will add the channel dimension at the last position with axis=3
{: .notice--info}



```python
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
```


```python
x_train.shape
```




    (24, 256, 256, 1)




```python
y_train.shape
```




    (24,)



we can use the  `x` and `y` to create a dataset using the tensorflow method `from_tensor_slices()`. it takes the `x` inputs and `y` targets numpy arrays and return tuples of tensors that can be shuffled and batched with biult-in methods.



```python
import tensorflow as tf
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
```

We can call methods like `shuffle()` and `batch()` on the TensorFlow dataset.

with `shuffle()` method called, the dataset fills a buffer with `buffer_size` elements, then randomly samples elements from this buffer, replacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.

while `batch()` method defines the batches to generate from the dataset according to the batch size given.


```python
train_data = train_data.shuffle(24).batch(6)
test_data = test_data.shuffle(24).batch(2)
```

## Creating a simple CNN for classification task to train the prepared data
this is just for demonstration purposes to give an idea of how we can use the prepared data to train a model.



```python

model =  tf.keras.Sequential([

            tf.keras.layers.Conv2D(64, 3, strides=(1, 1),
                                   activation='relu', padding='same',
                                   input_shape=[256, 256, 1]),

            tf.keras.layers.MaxPooling2D(pool_size=2),

            tf.keras.layers.Conv2D(128, 3, strides=(1, 1),
                                   activation='relu', padding='same',
                                   ),
            tf.keras.layers.MaxPooling2D(pool_size=2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 256, 256, 64)      640       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 128, 128, 64)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 128, 128, 128)     73856     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 64, 64, 128)       0         
    _________________________________________________________________
    flatten (Flatten)            (None, 524288)            0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                33554496  
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 33,629,057
    Trainable params: 33,629,057
    Non-trainable params: 0
    _________________________________________________________________


# Compiling and traing the model for 20 epochs
on each epoch the dataset will return a batch of 6 shuffled images along with their labels. for a total of 4 step. 
`4*6 = 24(total images in train_set)`


```python
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['binary_accuracy'], )
history = model.fit(train_data, epochs=20)
```

    Epoch 1/20


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py:375: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      "The `lr` argument is deprecated, use `learning_rate` instead.")


    4/4 [==============================] - 7s 1s/step - loss: 582.2830 - binary_accuracy: 0.5833
    Epoch 2/20
    4/4 [==============================] - 5s 1s/step - loss: 85.3040 - binary_accuracy: 0.3750
    Epoch 3/20
    4/4 [==============================] - 5s 1s/step - loss: 8.2543 - binary_accuracy: 0.5000
    Epoch 4/20
    4/4 [==============================] - 5s 1s/step - loss: 1.0707 - binary_accuracy: 0.4583
    Epoch 5/20
    4/4 [==============================] - 5s 1s/step - loss: 0.6798 - binary_accuracy: 0.5000
    Epoch 6/20
    4/4 [==============================] - 5s 1s/step - loss: 0.9097 - binary_accuracy: 0.5417
    Epoch 7/20
    4/4 [==============================] - 5s 1s/step - loss: 0.6698 - binary_accuracy: 0.5000
    Epoch 8/20
    4/4 [==============================] - 5s 1s/step - loss: 0.6643 - binary_accuracy: 0.5833
    Epoch 9/20
    4/4 [==============================] - 5s 1s/step - loss: 0.6501 - binary_accuracy: 0.7083
    Epoch 10/20
    4/4 [==============================] - 5s 1s/step - loss: 0.6487 - binary_accuracy: 0.6667
    Epoch 11/20
    4/4 [==============================] - 5s 1s/step - loss: 0.6339 - binary_accuracy: 0.7083
    Epoch 12/20
    4/4 [==============================] - 5s 1s/step - loss: 0.6202 - binary_accuracy: 0.6667
    Epoch 13/20
    4/4 [==============================] - 5s 1s/step - loss: 0.6093 - binary_accuracy: 0.7083
    Epoch 14/20
    4/4 [==============================] - 5s 1s/step - loss: 0.5785 - binary_accuracy: 0.6667
    Epoch 15/20
    4/4 [==============================] - 5s 1s/step - loss: 0.5627 - binary_accuracy: 0.7083
    Epoch 16/20
    4/4 [==============================] - 5s 1s/step - loss: 0.5541 - binary_accuracy: 0.6667
    Epoch 17/20
    4/4 [==============================] - 5s 1s/step - loss: 0.5215 - binary_accuracy: 0.7917
    Epoch 18/20
    4/4 [==============================] - 5s 1s/step - loss: 0.5031 - binary_accuracy: 0.7917
    Epoch 19/20
    4/4 [==============================] - 5s 1s/step - loss: 0.4845 - binary_accuracy: 0.7500
    Epoch 20/20
    4/4 [==============================] - 5s 1s/step - loss: 0.4575 - binary_accuracy: 0.7917


80% train accuracy is not that bad. next tutorial i will show how to get a 100% test and train accuracy on this dataset


# Using TensorFlow's built-in class `ImageDataGenerator` and `flow_from_directory()` class method  to load data
Although the numpy way of loading data is more intuitive and very *pythonic* however, it might be a little difficult for some poeple. for that, Tensorflow has  biult-in methods  like `flow_from_directory()` which can be used to load classification data.
we need to pass the root directory of the data set as an argument and tensorflow handles the rest. it recognizes each subdirectory in the root as a class and the files insides and inputs. for example if we pass a directory named `training` with two folders,  with  30 and  25 images respectively,  to this method then it will recgonize them as two classes `0` and `1` having 30 and 25 images. 
we can also set data shuffling to be true and define batch size, with arguments to this method and apply transforms like rotation, zoom, random_crop and random flips.
for more, go check out the official documentation for this [here](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory)

below is an example of using this method to load data from a directory


for the training dataset, first, we define the different transforms, operations that we want to apply to this dataset and pass it to the `ImageDataGenerator` class. then we call the method `flow_from_directory()` on the class constructor `train_datagen` and pass the train directory path along with other arguments like `color_mode` and `batch_size` the shuffle is `True` by default so I didn't bother specifying it.

as for the `test_data` we only apply rescale to the image and not the augmentations. because we need to test our model on the unchanged data. 

notice how TensorFlow recognizes two different classes and the images for each of them and generates labels in the backend without defining them manually. it certainly gives a lot of abstraction and gets the job done for us
{: .notice--info}


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
kwargs = dict(
    featurewise_center=False,
    featurewise_std_normalization=False,
    # rotation_range=90,
    rescale=1. / 255,
    # zca_whitening=True,
    #horizontal_flip=True,
    #vertical_flip=True,
    # preprocessing_function=preprocessing
)

train_datagen = ImageDataGenerator(**kwargs)

train_tfdata = train_datagen.flow_from_directory(directory=to_create.get('train_dir'),
                                                                        seed=24,
                                                                        color_mode='grayscale',
                                                                        batch_size=6,class_mode='binary'
                                                                        )
test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  )
test_tfdata = test_datagen.flow_from_directory(directory=to_create.get('test_dir'),
                                                                        seed=24,
                                                                        color_mode='grayscale',
                                                                        batch_size=2,class_mode='binary'
                                                                        )
```

    Found 24 images belonging to 2 classes.
    Found 6 images belonging to 2 classes.


Now we can use these datasets to train our already defined model. this time we will pass the train_tfdata for in order to train the model



```python

history = model.fit(train_tfdata, epochs=20)
```

    Epoch 1/20
    4/4 [==============================] - 8s 2s/step - loss: 1.1575 - binary_accuracy: 0.2500
    Epoch 2/20
    4/4 [==============================] - 8s 2s/step - loss: 1.0284 - binary_accuracy: 0.1667
    Epoch 3/20
    4/4 [==============================] - 8s 2s/step - loss: 0.8951 - binary_accuracy: 0.0833
    Epoch 4/20
    4/4 [==============================] - 8s 2s/step - loss: 0.8095 - binary_accuracy: 0.2500
    Epoch 5/20
    4/4 [==============================] - 8s 2s/step - loss: 0.7385 - binary_accuracy: 0.4167
    Epoch 6/20
    4/4 [==============================] - 8s 2s/step - loss: 0.7180 - binary_accuracy: 0.3333
    Epoch 7/20
    4/4 [==============================] - 8s 2s/step - loss: 0.7209 - binary_accuracy: 0.4583
    Epoch 8/20
    4/4 [==============================] - 8s 2s/step - loss: 0.6970 - binary_accuracy: 0.5000
    Epoch 9/20
    4/4 [==============================] - 8s 2s/step - loss: 0.6807 - binary_accuracy: 0.5417
    Epoch 10/20
    4/4 [==============================] - 8s 2s/step - loss: 0.6745 - binary_accuracy: 0.5833
    Epoch 11/20
    4/4 [==============================] - 8s 2s/step - loss: 0.6620 - binary_accuracy: 0.6667
    Epoch 12/20
    4/4 [==============================] - 8s 2s/step - loss: 0.6562 - binary_accuracy: 0.6667
    Epoch 13/20
    4/4 [==============================] - 8s 2s/step - loss: 0.6521 - binary_accuracy: 0.6250
    Epoch 14/20
    4/4 [==============================] - 8s 2s/step - loss: 0.6350 - binary_accuracy: 0.6667
    Epoch 15/20
    4/4 [==============================] - 8s 2s/step - loss: 0.6274 - binary_accuracy: 0.6667
    Epoch 16/20
    4/4 [==============================] - 8s 2s/step - loss: 0.6217 - binary_accuracy: 0.6667
    Epoch 17/20
    4/4 [==============================] - 8s 2s/step - loss: 0.6034 - binary_accuracy: 0.7083
    Epoch 18/20
    4/4 [==============================] - 8s 2s/step - loss: 0.6008 - binary_accuracy: 0.7083
    Epoch 19/20
    4/4 [==============================] - 8s 2s/step - loss: 0.5836 - binary_accuracy: 0.7500
    Epoch 20/20
    4/4 [==============================] - 8s 2s/step - loss: 0.5789 - binary_accuracy: 0.7083



```python
model.evaluate(test_tfdata)
```

    3/3 [==============================] - 1s 370ms/step - loss: 0.6590 - binary_accuracy: 0.3333





    [0.6589697599411011, 0.3333333432674408]



while we have a 70% train accuracy, we ended up getting a mere 33% accuracy on our test dataset. This is a clear sign of over-fitting

That's it ...


 easy, right?

