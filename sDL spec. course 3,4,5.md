# Deep Learning Specialization Courses 3,4,5

[TOC]



## Structuring Machine learning projects

### week 1

#### Why strategy?

because there are a lot of parameters that we can determine, so we need a good strategy to find the best path to success 

<img src="sDL spec. course 3,4,5.assets/image-20200423085007994.png" alt="image-20200423085007994" style="zoom:40%;" />

#### Orthogonalization:

distribute controllers in angles (at 90 degree) 
good fit on train (if not good on this use bigger net, Adam , ... ) -> dev (if we didn't do well on dev use regularization, bigger train) - > test (use bigger dev set if test set accuracy is bad) -> real world (dev set and cost function) 
early stopping is not good

#### single number Evaluation Metric:
 f1 score = harmonic mean of precision and recall always use dev set 
average is not bad to use and gain insights. 

#### Satisficing and optimizing metric

how to combine accuracy and running time. using optimizing (maximize accuracy) and satisficing running time (time < 100 ms)

maximize accuracy and s.t. FP < 1 

#### Train Dev Test Distributions

dev aka hold out is used as a metric to find a good classifier hyper parameters and then we go and test our classifier on test set

random shuffle data from all the distribution 

" choose a dev set and test set to reflect data you expect to get in the future and consider important to do well on "

#### Size of the dev and test sets 

- old era with 100 examples 60/30/10. 
- new era = 98/1/1 (1 millions of data)
- better use test set 

#### When to change dev/test sets and metrics

When user choice is not consistent with your metric. 

- amend your metric if you are not satisfy with it or it's not in correspondence with user choices 

#### Why human-level performance?

1. advances of dl is competitive with human
2. workflow of learning is similar 

- Bayes optimal error in the optimal error. (aka Bayes Error).
- manual error analysis why did a person get this right
- better analysis of bias/variance

#### Avoidable bias

<img src="sDL spec. course 3,4,5.assets/image-20200423093005098.png" alt="image-20200423093005098" style="zoom:40%;" />

when huge gap between human and train error, focus on bias

when gap between train and dev error, focus on variance (avoidable bias between human and train error)

#### Understanding human-level performance

It depends on you application, maybe surpassing one doctor is enough for your application. maybe surpassing a group of 5 experienced doctors are good enough for you. 

<img src="sDL spec. course 3,4,5.assets/image-20200423093717675.png" alt="image-20200423093717675" style="zoom:40%;" />

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

<img src="sDL spec. course 3,4,5.assets/image-20200425094411151.png" alt="image-20200425094411151" style="zoom:40%;" />

1. overall dev set error 10%
2. error due incorrect labels 0.6%
3. errors due to other causes 9.4%

goal of dev set is to choose better algorithms. 

- apply same process to dev and test set
- consider examining right labeled too.
- training can be come from slightly different distributions 

#### Build your first system quickly, then iterate

how to pick best direction to go?

1.  set up dev/test set and metric
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

<img src="sDL spec. course 3,4,5.assets/image-20200426082445980.png" alt="image-20200426082445980" style="zoom:40%;" />

<img src="sDL spec. course 3,4,5.assets/image-20200426083110270.png" alt="image-20200426083110270" style="zoom:40%;" />

#### Addressing data mismatch

<img src="sDL spec. course 3,4,5.assets/image-20200426083329126.png" alt="image-20200426083329126" style="zoom:40%;" />

artificial data synthesis can work well but we should try bigger synthetization to don't over fit. 

#### transfer Learning

pre training = use another models weights 

fine tuning= train some of weights 

- type of input must be same
- when transferring from a -> b we have lot data in task a 
- low level features from A could be helpful for learning B

#### Multi-task learning

having more than 1 label for data

<img src="sDL spec. course 3,4,5.assets/image-20200426091759927.png" alt="image-20200426091759927" style="zoom:40%;" />

<img src="sDL spec. course 3,4,5.assets/image-20200426092341758.png" alt="image-20200426092341758" style="zoom:40%;" />

#### What is end-to-end deep learning?

<img src="sDL spec. course 3,4,5.assets/image-20200427091455434.png" alt="image-20200427091455434" style="zoom:40%;" />

some times end to end works because of majority of data, but some times doesn't work. 

if you have lot of data for subtasks, you need to break your problem to subtasks. 

#### Whether to use end-to-end deep learning

<img src="sDL spec. course 3,4,5.assets/image-20200427091906226.png" alt="image-20200427091906226" style="zoom:40%;" />

<img src="sDL spec. course 3,4,5.assets/image-20200427092250271.png" alt="image-20200427092250271" style="zoom:40%;"/>



## Convolutional Neural Networks

### Week 1

#### Computer Vision

Rapid advances are happening so fast that can use in applications 

and ideas of cv can be used in other areas

having large images(1000*1000 * 3 pixel = 1 Mega pixel) will make learnable parameters very plentiful. it's hard to train a neural network in computation wise. 

#### Edge detection Example

conv operation: basis of CNN.

vertical edges and horizontal image

filter aka kernel 

convolution = * 

element wise product 

<img src="sDL spec. course 3,4,5.assets/image-20200509090606377.png" alt="image-20200509090606377" style="zoom:30%;" />

```python
def conv_forward #Python
tf.nn.conv2d #Tensorflow
keras.layers.Conv2D # Keras
```

the middle kernel is an vertical kernel 

<img src="sDL spec. course 3,4,5.assets/image-20200509091110029.png" alt="image-20200509091110029" style="zoom:30%;" />

#### More Edge Detection

<img src="sDL spec. course 3,4,5.assets/image-20200509091800182.png" alt="image-20200509091800182" style="zoom:33%;" />

in deep learning we don't hand pick kernels and network will learn the kernel numbers as parameters.

#### Padding

<img src="sDL spec. course 3,4,5.assets/image-20200509092145554.png" alt="image-20200509092145554" style="zoom:30%;" />

6\*6 conv 3\*3  will make 4\*4 matrix where 4 is coming from 6-3+1 

padding = p = 1 => n+1\*n+1 

- Valid Conv = No padding
- same Conv = we pad so that output becomes the same shape of input size => $p=\frac{f-1}{2}$ as f is kernel dim
- we usually use odd numbers for f (kernel dim)

#### Strided Conv

this is the step of moving kernel. stride is s in below formula

$output dim ={\frac{n + 2*p - f}{s}+1}$

<img src="sDL spec. course 3,4,5.assets/image-20200509094618368.png" alt="image-20200509094618368" style="zoom:33%;" />

if output dim is not integer we use floor of output dim 

<img src="sDL spec. course 3,4,5.assets/image-20200509094344869.png" alt="image-20200509094344869" style="zoom:33%;" />

we don not compute that last operation if some of the kernel has gone out of the input.



<img src="sDL spec. course 3,4,5.assets/image-20200509094952695.png" alt="image-20200509094952695" style="zoom:33%;" />

in deep learning we call this conv operator. we don't flip on each dim. 

this should not effect your implementations

#### Convolutions Over Volume

<img src="sDL spec. course 3,4,5.assets/image-20200509140258652.png" alt="image-20200509140258652" style="zoom:33%;" />

(height, width, channels aka depth) channels must be same. 

<img src="sDL spec. course 3,4,5.assets/image-20200509141024469.png" alt="image-20200509141024469" style="zoom:33%;" />

#### One Layer of a Convolutional Network

<img src="sDL spec. course 3,4,5.assets/image-20200509144415322.png" alt="image-20200509144415322" style="zoom:33%;" />

#### Simple Convolutional Network Example

<img src="sDL spec. course 3,4,5.assets/image-20200509150745907.png" alt="image-20200509150745907" style="zoom:33%;" />

#### Pooling Layers

1. Max Pooling (returning max of the corresponding matrixes in dim of f \* f) it does not have any learnable parameters. it just have Hyperparameters
2. Average pooling = like max pooling but averaging on kernel

<img src="sDL spec. course 3,4,5.assets/image-20200509165148103.png" alt="image-20200509165148103" style="zoom:33%;" />

#### CNN Example

<img src="sDL spec. course 3,4,5.assets/image-20200509172235078.png" alt="image-20200509172235078" style="zoom:33%;" />

#### Why Convolutions?

1. parameter sharing: a good kernel can be used in many places of the picture
2. sparsity of connections: in each layer only a faction of  inputs numbers are connected to output.

### Week 2

#### Why look at case studies?

some architecture that works good on sth can be good for ypu task too.

- Lenet-5
- AlexNet
- VGG
- ResNet (residual net)
- Inception

these ideas can b e applied in other areas of AI and DL.

#### Classic Networks

##### Lenet - 5

<img src="sDL spec. course 3,4,5.assets/image-20200511083032374.png" alt="image-20200511083032374" style="zoom:33%;" />

focus on section 2 and 3 of paper. other sections are unfortunately obsolete right now. 

they used tanh and sigmoid (not relu)

60 k parameters

##### Alex Net

- like Lenet-5 but much bigger (60 M parameter)

- relu 
- Multiple gpu
- local response normalization (obsolete)

<img src="sDL spec. course 3,4,5.assets/image-20200511083525950.png" alt="image-20200511083525950" style="zoom:33%;" />

##### VGG-16

- all kernels are same : conv layers 3*3 filters  , s=1, same padding
- all max pools are same: max pool = 2*2 , s=2
- 16 in name refers to 16 learnable layers. 
- 138 M parameters
- a simple principle to use in CNN is to double Conv layers every time you are going deeper ( from 64 Conv layers to 128 to 256 and etc. )

<img src="sDL spec. course 3,4,5.assets/image-20200511084246170.png" alt="image-20200511084246170" style="zoom:33%;" />

Conv X (where X is a number) means we are stacking X Conv layers. 

#### ResNets

short cut(skip) connection helps with exploding gradients and vanishing gradients 

##### residual block

We copy a layer and add them to later layers. 

we stack residual blocks together to make a deep network. 

#### Why ResNets Work

it is easy for network to learn to skip some layers and can be shallow if the network finds out being deeper will hurt him. 

<img src="sDL spec. course 3,4,5.assets/image-20200512134943090.png" alt="image-20200512134943090" style="zoom:33%;" />

dashed lines happens when there is a pooling that reduce shape of image. so we need a Ws matrix for residual block to compensate for this dimension reduction. Ws parameter will be multiplied to the a[l+1] to fix it's dimension for addition of parameters at the skip connection.

#### Networks in Networks and 1x1 Convolutions

1*1 conv will multiply every pixel by a constant.

6 x 6 x 32 image * 1 x 1 x 32 conv will reduce the depth of image. the output will be 6 x 6 x #filters. this idea is named "Network in network".

<img src="sDL spec. course 3,4,5.assets/image-20200512135922162.png" alt="image-20200512135922162" style="zoom:33%;" />

this technique is used to shrink/escalate the depth of image. 

#### Inception Network Motivation

<img src="sDL spec. course 3,4,5.assets/image-20200512140448980.png" alt="image-20200512140448980" style="zoom:33%;" />

some times it is called "bottle neck layer". 

<img src="sDL spec. course 3,4,5.assets/image-20200513091935919.png" alt="image-20200513091935919" style="zoom:33%;" />

the below image is of a Inception network developed by google that is name "GooLenet" to cite Lenet name in its name. 

<img src="sDL spec. course 3,4,5.assets/image-20200513092357133.png" alt="image-20200513092357133" style="zoom:33%;" />

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
- rotation, shearinglocal warping
- Color shifting (+20, -20, +20). you can use little numbers as well. [ You can use PCA color augmentation. make color proportions even ]

you can augment online and in parallel of learning. 

we can also use augmentation on hard disk.

#### State of Computer Vision

uniqueness of CV. 

spectrum of little data and lots of data. 

- lot of data : simpler algorithms, hand engineering, 
- little data = more hand engineering (transfer learning will help alot)

in CV we don't have enough data yet. so we try more network architectures

##### tips on benchmark/wining competitions

- Ensemble : train several nets independently and average their outputs (Y-hats). you can test 3-15 networks
- Multi-Crop at test time: Run classifier in multiple versions of test images and average the results; 10 crop will be look like this

<img src="sDL spec. course 3,4,5.assets/image-20200513133909862.png" alt="image-20200513133909862" style="zoom:33%;" />

##### use open source code

- use Architectures of network published in the literature
- use open source implementation if possible
- use pretrained models and fine tune on you dataset