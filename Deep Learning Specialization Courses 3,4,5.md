# Deep Learning Specialization Courses 3,4,5

[TOC]

## Structuring Machine learning projects

### week 1

#### Why strategy

because there are a lot of parameters that we can determine, so we need a good strategy to find the best path to success.

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200423085007994.png" alt="image-20200423085007994" style="zoom:70%;" />

#### Orthogonalization

distribute controllers in angles (at 90 degree)
good fit on train (if not good on this use bigger net, Adam , ... ) -> dev (if we didn't do well on dev use regularization, bigger train) - > test (use bigger dev set if test set accuracy is bad) -> real world (dev set and cost function)
early stopping is not good

#### single number Evaluation Metric

f1 score = harmonic mean of precision and recall always use dev set
average is not bad to use and gain insights.

#### Satisficing and optimizing metric

how to combine accuracy and running time. using optimizing (maximize accuracy) and satisficing running time (time < 100 ms)

maximize accuracy and s.t. FP < 1

#### Train Dev Test Distributions

dev aka hold out is used as a metric to find a good classifier hyper parameters and then we go and test our classifier on test set

random shuffle data from all the distribution

importat: **choose a dev set and test set to reflect data you expect to get in the future and consider important to do well on**

#### Size of the dev and test sets

- old era with 100 examples 60/30/10.
- new era = 98/1/1 (1 millions of data)
- better use test set

#### When to change dev/test sets and metrics

When user choice is not consistent with your metric.

- amend your metric if you are not satisfy with it or it's not in correspondence with user choices

#### Why human-level performance

1. advances of dl is competitive with human
2. workflow of learning is similar

- Bayes optimal error in the optimal error. (aka Bayes Error).
- manual error analysis why did a person get this right
- better analysis of bias/variance

#### Avoidable bias

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200423093005098.png" alt="image-20200423093005098" style="zoom:80%;" />

when huge gap between human and train error, focus on bias

when gap between train and dev error, focus on variance (avoidable bias between human and train error)

#### Understanding human-level performance

It depends on you application, maybe surpassing one doctor is enough for your application. maybe surpassing a group of 5 experienced doctors are good enough for you.

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200423093717675.png" alt="image-20200423093717675" style="zoom:40%;" />

#### Surpassing human-level performance

When you surpass human level, it's harder to enhance it.

humans are good in natural tasks(vision, nlp, etc.)

for unnatural problems computers are really good and surpass human level performance.

#### Improving your model performance

how much better you think you can do better on training set.

-----

#### Andrej Karpathy interview

http://karpathy.github.io/2015/11/14/ai/

http://cs231n.stanford.edu/

http://karpathy.github.io/

____

### week2 

#### Carrying out error analysis

look at dev examples to evaluate ideas

**error analysis:**

- get 100 mislabeled dev set example
- count up how many are dogs

if a lot of dev images are mislabeled, we can improves it better. 

|   images   | blurry | dog  | cats |
| :--------: | ------ | ---- | ---- |
|     1      | x      |      |      |
|     2      |        |      | x    |
|     3      |        |      | x    |
| % of total | 8%     | 43%  | 61%  |

counting FP and FN

#### Cleaning up incorrectly labeled data

dl is robust to random errors but not robust to systematical errors. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200425094411151.png" alt="image-20200425094411151" style="zoom:40%;" />

1. overall dev set error 10%
2. error due incorrect labels 0.6%
3. errors due to other causes 9.4%

goal of dev set is to choose better algorithms. 

- apply same process to dev and test set
- consider examining right labeled too.
- training can be come from slightly different distributions 

#### Build your first system quickly, then iterate

how to pick best direction to go?

1. set up dev/test set and metric
2. build initial system
3. use Bias/Variance analysis and error analysis to prioritize next steps
4. (you can use academic literatures to get idea)

#### Training and testing on different distributions

hinger of el algo -> more data -> multiple distributions of data 

- [ ]  option 1: shuffle different distros of data and then split.
- [x] option2: train = web + app , dev = just app, test = just app

option 2 is better.

#### Bias and Variance with mismatched data distributions

we can not draw high bias/ variance conclusions when we don't have same distributions in train/dev.

use training-dev

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200426082445980.png" alt="image-20200426082445980" style="zoom:40%;" />

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200426083110270.png" alt="image-20200426083110270" style="zoom:40%;" />

#### Addressing data mismatch

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200426083329126.png" alt="image-20200426083329126" style="zoom:40%;" />

artificial data synthesis can work well but we should try bigger synthetization to don't over fit. 

#### transfer Learning

pre training = use another models weights

fine tuning= train some of weights

- type of input must be same
- when transferring from a -> b we have lot data in task a
- low level features from A could be helpful for learning B

#### Multi-task learning

having more than 1 label for data

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200426091759927.png" alt="image-20200426091759927" style="zoom:40%;" />

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200426092341758.png" alt="image-20200426092341758" style="zoom:40%;" />

#### What is end-to-end deep learning

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200427091455434.png" alt="image-20200427091455434" style="zoom:40%;" />

some times end to end works because of majority of data, but some times doesn't work.

if you have lot of data for subtasks, you need to break your problem to subtasks.

#### Whether to use end-to-end deep learning

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200427091906226.png" alt="image-20200427091906226" style="zoom:40%;" />

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200427092250271.png" alt="image-20200427092250271" style="zoom:40%;"/>

## Convolutional Neural Networks

### Week 1

#### Computer Vision

Rapid advances are happening so fast that can use in applications 

and ideas of cv can be used in other areas

having large images(1000\*1000 \* 3 pixel = 1 Mega pixel) will make learnable parameters very plentiful. it's hard to train a neural network in computation wise.

#### Edge detection Example

conv operation: basis of CNN.

vertical edges and horizontal image

filter aka kernel

convolution = *

element wise product

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509090606377.png" alt="image-20200509090606377" style="zoom:30%;" />

```python
def conv_forward #Python
tf.nn.conv2d #Tensorflow
keras.layers.Conv2D # Keras
```

the middle kernel is an vertical kernel

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509091110029.png" alt="image-20200509091110029" style="zoom:30%;" />

#### More Edge Detection

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509091800182.png" alt="image-20200509091800182" style="zoom:33%;" />

in deep learning we don't hand pick kernels and network will learn the kernel numbers as parameters.

#### Padding

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509092145554.png" alt="image-20200509092145554" style="zoom:30%;" />

6\*6 conv 3\*3  will make 4\*4 matrix where 4 is coming from 6-3+1 

padding = p = 1 => n+1\*n+1

- Valid Conv = No padding
- same Conv = we pad so that output becomes the same shape of input size => $p=\frac{f-1}{2}$ as f is kernel dim
- we usually use odd numbers for f (kernel dim)

#### Strided Conv

this is the step of moving kernel. stride is s in below formula

$output dim ={\frac{n + 2*p - f}{s}+1}$

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509094618368.png" alt="image-20200509094618368" style="zoom:33%;" />

if output dim is not integer we use floor of output dim

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509094344869.png" alt="image-20200509094344869" style="zoom:33%;" />

we don not compute that last operation if some of the kernel has gone out of the input.

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509094952695.png" alt="image-20200509094952695" style="zoom:33%;" />

in deep learning we call this conv operator. we don't flip on each dim.

this should not effect your implementations

#### Convolutions Over Volume

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509140258652.png" alt="image-20200509140258652" style="zoom:33%;" />

(height, width, channels aka depth) channels must be same.

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509141024469.png" alt="image-20200509141024469" style="zoom:33%;" />

#### One Layer of a Convolutional Network

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509144415322.png" alt="image-20200509144415322" style="zoom:33%;" />

#### Simple Convolutional Network Example

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509150745907.png" alt="image-20200509150745907" style="zoom:33%;" />

#### Pooling Layers

1. Max Pooling (returning max of the corresponding matrixes in dim of f \* f) it does not have any learnable parameters. it just have Hyperparameters
2. Average pooling = like max pooling but averaging on kernel

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509165148103.png" alt="image-20200509165148103" style="zoom:33%;" />

#### CNN Example

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200509172235078.png" alt="image-20200509172235078" style="zoom:33%;" />

#### Why Convolutions

1. parameter sharing: a good kernel can be used in many places of the picture
2. sparsity of connections: in each layer only a faction of  inputs numbers are connected to output.

### Week 2

#### Why look at case studies

some architecture that works good on sth can be good for ypu task too.

- Lenet-5
- AlexNet
- VGG
- ResNet (residual net)
- Inception

these ideas can b e applied in other areas of AI and DL.

#### Classic Networks

##### Lenet - 5

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200511083032374.png" alt="image-20200511083032374" style="zoom:33%;" />

focus on section 2 and 3 of paper. other sections are unfortunately obsolete right now.

they used tanh and sigmoid (not relu)

60 k parameters

##### Alex Net

- like Lenet-5 but much bigger (60 M parameter)

- relu
- Multiple gpu
- local response normalization (obsolete)

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200511083525950.png" alt="image-20200511083525950" style="zoom:33%;"/>

##### VGG-16

- all kernels are same : conv layers 3*3 filters  , s=1, same padding
- all max pools are same: max pool = 2*2 , s=2
- 16 in name refers to 16 learnable layers.
- 138 M parameters
- a simple principle to use in CNN is to double Conv layers every time you are going deeper ( from 64 Conv layers to 128 to 256 and etc. )

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200511084246170.png" alt="image-20200511084246170" style="zoom:33%;" />

Conv X (where X is a number) means we are stacking X Conv layers.

#### ResNets

short cut(skip) connection helps with exploding gradients and vanishing gradients

##### residual block

We copy a layer and add them to later layers.

we stack residual blocks together to make a deep network.

#### Why ResNets Work

it is easy for network to learn to skip some layers and can be shallow if the network finds out being deeper will hurt him.

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200512134943090.png" alt="image-20200512134943090" style="zoom:33%;" />

dashed lines happens when there is a pooling that reduce shape of image. so we need a Ws matrix for residual block to compensate for this dimension reduction. Ws parameter will be multiplied to the a[l+1] to fix it's dimension for addition of parameters at the skip connection.

#### Networks in Networks and 1x1 Convolutions

1*1 conv will multiply every pixel by a constant.

6 x 6 x 32 image * 1 x 1 x 32 conv will reduce the depth of image. the output will be 6 x 6 x #filters. this idea is named "Network in network".

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200512135922162.png" alt="image-20200512135922162" style="zoom:33%;" />

this technique is used to shrink/escalate the depth of image. 

#### Inception Network Motivation

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200512140448980.png" alt="image-20200512140448980" style="zoom:33%;" />

some times it is called "bottle neck layer". 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200513091935919.png" alt="image-20200513091935919" style="zoom:33%;" />

the below image is of a Inception network developed by google that is name "GooLenet" to cite Lenet name in its name. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200513092357133.png" alt="image-20200513092357133" style="zoom:33%;" />

Inception v2,3,4, etc. can be found in Internet

#### Using Open-Source Implementation

you can implement papers from scratch but if a lot of vision researchers did that before you, you can start your work based on their work. You can access their open source implementations from GitHub.

#### Transfer Learning

there are four ways of transfer learning in implementation

1. if we have few examples, freeze all of primitive layers and delete last layer's activations. then add you SoftMax layer and train the network
2. save the last layers of your images to disk (we call it features) then train a shallow neural net from those features. 
3. if we have a lot of training examples we can freeze less layers. 
4. if you have a massive amount of data, you can use transfer learning weights for initialization of your data and let it be trained on the massive data. 

#### Data Augmentation

computer vision is a complex task. we can get all of the data we need. Common augmentation are below: 

- mirroring 
- random cropping 
- rotation, shearing local warping
- Color shifting (+20, -20, +20). you can use little numbers as well. [ You can use PCA color augmentation. make color proportions even ]

you can augment online and in parallel of learning. 

we can also use augmentation on hard disk.

#### State of Computer Vision

uniqueness of CV. 

spectrum of little data and lots of data. 

- lot of data : simpler algorithms, hand engineering, 
- little data = more hand engineering (transfer learning will help a lot)

in CV we don't have enough data yet. so we try more network architectures

##### tips on benchmark/wining competitions

- Ensemble : train several nets independently and average their outputs (Y-hats). you can test 3-15 networks
- Multi-Crop at test time: Run classifier in multiple versions of test images and average the results; 10 crop will be look like this

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200513133909862.png" alt="image-20200513133909862" style="zoom:33%;" />

##### use open source code

- use Architectures of network published in the literature
- use open source implementation if possible
- use pretrained models and fine tune on you dataset

### Week 3

#### Object Localization

putting a bounding box around the object

- classification & localization : often 1 object in the middle of pic

- object detection : often more than 1 object

bounding box = bx , by = middle of object , bh , bw = height and width

y = (pc = is there an object? , bx,by,bh,bw, c1,c2,c3  = classes)

if there is no object pc = 0 and other outputs are don't care

##### loss

- if pc = 1 => square error for all losses

- if pc =0 => loss = (pc - y_hat)^2 

in practice we can use other errors. 

#### Landmark Detection

this is used in face detection. for eg 64 points of important parts (corner of eyes and lips and etc)

we can use it to pose detection (shoulder and legs  and etc.)

labeled should be consistent with a sequent. 

#### Object Detection

car detection = closely cropped images of imaged and centered. 

sliding image window detection. we go through the image window to windows. with strides and different window sized.

 <img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200518094159718.png" alt="image-20200518094159718" style="zoom:67%;" />

sliding window is not recommended with NN and Conv Nets. (it is too slow)

#### Convolutional Implementation of Sliding Windows

Turn FC to Conv Net: for every layer of FC, we replace with a conv of the shape of the input. we use N as the number of FC neurons. so we need N Conv filters. 

(400 , 5x5x16)  Conv = 400 FC

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200518095052763.png" alt="image-20200518095052763" style="zoom:67%;" />

we train our conv net with the first line of the picture above

in the middle line, we do not slide a window on 16x16 image (these images are the test images that are bigger than train images) . we convolve it with the same kernels. it turns out that every 1x1x4 matrix in the 2x2x4 matrix at last layer is the corresponding results for the sliding window. this way we share a lot of computational parameters between windows. 

#### Bounding Box Predictions

bounding box can be rectangle 

YOLO = you only look once

use a grid (19 x 19). for each cell in grid we use localization. 

for each of grid we specify a label. (and we run localization for every cell)

YOLO assign the mid point of the object to a grid cell. 

the output will be (19x19x8) 8 is the output vector of (p, bx, by, bh, bw, c1,c2,c3)

we demonstrate the problem of having two or more obj in grid.

YOLO is a one conv net. with shared computations. we can use it with Realtime object detection 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200521110009801.png" alt="image-20200521110009801" style="zoom:67%;" />

##### specify bounding box

top left corner of grid cell is 0,0 and bottom right corner of grid cell is 1,1. bx,by,bh,bw will be calculated with correspondence to these relative coordinates. 

bx,by, should be between 0 and 1 but bh,bw can be bigger than 1 . 

#### Intersection Over Union

evaluate object localization

IOU (inter section over union). if IOU > 0.5 or 0.6  the answer is correct. 

union of predicted area and target area divided by the intersection area of them. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200521110855408.png" alt="image-20200521110855408" style="zoom:50%;" />

#### Non-max Suppression

your algorithm may find an object twice or more, Non-max suppression will help us here. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200521111239913.png" alt="image-20200521111239913" style="zoom:50%;" />

we choose the most probable one (based on output of the algorithm and we delete those bounding boxes that have a big IOU with the high probable one.)

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200521112041606.png" alt="image-20200521112041606" style="zoom:50%;" />

Non-max Algorithm

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200521112250687.png" alt="image-20200521112250687" style="zoom:50%;" />

if we have multiple classes. we should run non-max on each class. 

#### Anchor Boxes

one grid cell, two or more objects, what should we do? Anchor boxes to the rescue. 

we make our output doubled with different pre defined boxes. our output will be 3x3x8x(Number of anchor boxes).

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200521113004988.png" alt="image-20200521113004988" style="zoom:63%;" />

a concrete example

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200521113512253.png" alt="image-20200521113512253" style="zoom:50%;" />

we can use k means algorithm to define anchor box shapes

#### YOLO Algorithm

Output is 3 x 3 (grid size) x 2 (anchor boxes) x (5 [pc , bx,by,bh,bw] + # of classes)

we determine anchor boxing with IOU

5 anchor boxes is an normal value for anchor boxes.

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200521125551710.png" alt="image-20200521125551710" style="zoom:50%;" />

#### Region Proposals

rich feature hierarchies = propose regions. classify proposed regions one at a time. output label + bounding boxes

fast r-cnn : propose regions. use conv implementation of sliding windows to classify all the proposed regions

faster r-cnn : CNN to propose regions etc. . 

### Week4

#### What is face recognition

liveness recognition is essential for a good face recognition software

##### face verification vs face recognition

- verification : f(image, ID) = T or F

- face recognition: f(image) = ID

#### One shot learning

learn from one example. 

2 way doing this : 

1. image -> cnn -> a softmax (this is not practical) 

2. d(img1,img2) = degree of difference between images (with a threshold and this is more practical. in this method, we compare the given image against all of the database)

#### Siamese network

-  image 1-> cnn -> enc1 vector
- image 2  -> the same cnn -> enc 2 vector 
- the result is distance of enc 1 and enc2

the loss is we want to minimize the distance of same faces and maximize the distances of different faces. 

#### Triplet loss

learning objective: anchor image, positive image, and negative images are our 3 images. 

we want to d(anchor, positive) to be small and want d(anchor, negative) to be large. 

this formula can output zero for all vectors so we add an alpha (a constant for margin between two distances)<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200824173012289.png" alt="image-20200824173012289" style="zoom:67%;" />

we need about 10 or more pictures for a person. we also should train a triplet that is hard to train. the effect of choosing selectively on the accuracy is a lot. choosing random triplets will be ineffective on accuracy, because it is easy to recognize.  

#### Face verification and binary classification

chi square distance 

the both models parameters are tied together. we can pre compute encoding for our database and save the encoding of pictures in another database. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200824173938957.png" alt="image-20200824173938957" style="zoom:67%;" />

#### Neural style transfer

content (C) + style (S) => generated (G) image 

we should understand that what every layer of the conv net is doing.

#### what are deep cnn are learning

what image maximize a conv kernel? and what is that conv layer representing. layer one convs are trying to visualize edges

later convs will be responsible for bigger portions of the image. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200824175825062.png" alt="image-20200824175825062" style="zoom:50%;" />

#### cost function

J (G) = Alpha * J content (C,G) + Beta * J style (S,G)

1. initiate G randomly
2. update every pixel 

#### Content cost function

- we should choose a hidden layer 
- use pretrained 
-  we try to make $a^{[l][c]}$ (activation of the layer L and Content) and $a^{[l][g]}$ similar by using their squared distance

#### Style cost function

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200824180914419.png" alt="image-20200824180914419" style="zoom:67%;" />

we use correlation between channels of a style image and measure it against the correlation of the generated image and try to make these correlation similar. this similarity will give us a measurement to define the similarity of their styles.

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200824181613897.png" alt="image-20200824181613897" style="zoom:50%;" />

we use sum of all layers for style image to capture all kind of high level and low level features from the style image. 

so our final J will be sum of J style over all layers (or a subset of the layers)

#### 1D and 3D data for conv

1d Data (eg. EKG data)

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200824182313745.png" alt="image-20200824182313745" style="zoom:67%;" />

3d Data (CT scan and movie data)

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200824182334912.png" alt="image-20200824182334912" style="zoom:67%;" />

## Sequence Models

### Week 1

#### Recurrent neural models

##### Why sequence models?

examples:

- speech recognition
- music generation
- sentiment classification
- DNA sequence analysis
- Machine translation
- Video activity recognition
- Name entity recognition

they can have sequence in input, output or both. 

##### Notation

named entity recognition:

x: input - a set of $x^{<t>}$

y: a vector of binary (just in this example) - a set of $y^{<t>}$

$T_x$ = the length of the input

$T_y$ = the length of the output

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826091615532.png" alt="image-20200826091615532" style="zoom:50%;" />

and we make x and y to a matrix and then we have X and Y. then we can have:

$X^{(i)<t>} = i_{th}$ example of the dataset and $t_{th}$ in it

we need a dictionary (vocabulary) to present each word with the index of it in the dictionary. 

we can use one hot encoding

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826092002294.png" alt="image-20200826092002294" style="zoom:50%;" />

###### recurrent neural net model

we can use a simple neural net

problems:

- inputs and outputs lengths are different (zero-pad is good but not that good)
- doesn't share features across the different position of the text.

**RNN** will gets activations from the previous time step and passes its activation to the next time step.

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826093029038.png" alt="image-20200826093029038" style="zoom:50%;" />

it uses the same $W_x$ and $W_y$ weights each time step. the forward propagation is the sum of the inputs that has been multiplied with their W

problems:

- it doesn't look at the next words for predicting its output, but only looks at previous inputs. (Bi-RNNs can alleviate this)

we use this simplified notation, this will be enable us to denote more complex architectures.

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826093233849.png" alt="image-20200826093233849" style="zoom:67%;" />

we stack up the $W_s$ together vertically and call it $W_a$. also, stack the $a^{<t-1>}, x^{<t>}$ together horizontally. 

##### Back propagation through time

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826094451822.png" alt="image-20200826094451822" style="zoom:67%;" />

the main part of this graph is the right to left from a<t> to a<0>

##### Different types of RNN

in some examples of the problems, the input and output length can be different. in other words $T_X != T_Y$

if Tx and Ty are both sequence (whether equal or not), we call it **many to many architecture**.

and if it is a sentiment classification we call it **many to one architecture**. in this architecture we only use the last output of the last RNN cell.

in music generation we have a **one to many architecture** in oppose to the previous architecture. in this example we feed the y of each time step as the input of the next time step.

in machine translation we use another architecture. an encoder-decoder one

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826095204002.png" alt="image-20200826095204002" style="zoom:50%;" />

we first read the input and then generate the output one by one.

**to summarize**

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826095323711.png" alt="image-20200826095323711" style="zoom:80%;" />

##### Language model and sequence generation 

how to build a language model!

language model will give a probability of the sentence(sequence of inputs) to happen

Tokenize = give every word a token (index of it in dictionary and also <EOS> to the end of sentence )

also and <UNK> token to unknown words.

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826100559828.png" alt="image-20200826100559828" style="zoom:50%;" />

##### Sampling novel sequences

like music generation we feed the output of the previous time to the input of the next time. 

and we can use word level vocab or character level vocab. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826101201029.png" alt="image-20200826101201029" style="zoom:83%;" />

##### Vanishing gradients(RNN)

languages can have very long dependencies. so it is hard to address the first inputs to generate the latter outputs (because of the length of the sequence). 

vanishing gradients = very small gradients in long sequence. decrease gradually

##### Gated recurrent units (GRU)

can capture long dependencies. we use memory cell $C$. 

it uses a gate to memorize some part of the input and carry it to the latter time steps. then it decides that to update the memory cell or not using a sigmoid func. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826102516766.png" alt="image-20200826102516766" style="zoom:67%;" />

each output is the sum of the using the memory cell or not. 

this was the simplified version, the full version does look like this. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826102639554.png" alt="image-20200826102639554" style="zoom:67%;" />

this have another gate to say that how much it should depend on the previous input of the time step. the added gate is denoted in blue color in above picture. 

##### Long shot-term memory (LSTM)

there are 3 gates

forget, memorize, output

we carry the info untouched from c<0> and to the latter time steps. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826124542080.png" alt="image-20200826124542080" style="zoom:67%;" />

this have an additional gate in contrast to the GRU. this is the most common version of the lstm and there are other versions available out there. 

there is not a superior architecture, but to try one of them first, you can use LSTM and then use GRU after that. GRU is more simple and faster than the LSTM.

##### Bi directional RNN

fixes the issue of left to right approach for inputting the text. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826125237594.png" alt="image-20200826125237594" style="zoom:67%;" />

BRRN is simply two unidirectional with tied weights (I think)

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826125213440.png" alt="image-20200826125213440" style="zoom:80%;" />

this have a problem of that you need the full sequence the input to start computing. so you cannot have real time speech recognition systems. 

##### Deep RNN

notation is $a^{[l]<t>}$ that L is the layer and t is the time stamp. 

<img src="Deep Learning Specialization Courses 3,4,5.assets/image-20200826130012671.png" alt="image-20200826130012671" style="zoom:80%;" />

