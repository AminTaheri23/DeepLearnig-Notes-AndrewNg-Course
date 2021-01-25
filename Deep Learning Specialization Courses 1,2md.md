# Course 1 - Neural Networks and Deep Learning

**(the content was really similar to Andrew Ng Machine learning course in Coursera, so I didn't write too much about it)**

## Specialization Syllabus

- neural network and deep learning (cat recognizer)

- improving deep neural net

- structuring you ml project

- CNN
- NLP and sequence models

house price prediction (size - # of bedroom - family size - zip code - wealth)

### supervised learning application 

- house price prediction
- online advertisement
- photo tagging (CNN)
- speech recognition (RNN)
- machine translation (RNN)
- autonomous driving (hybrid)

### types of data

- structured data : table and data

- unstructured data : images and audio and text (deep learning)

**Geoffrey Hinton interview**: Boltzmann machines 

review chain rule

this the python implementation of a perceptron (without activation)

```python
z = np.dot(w.t , x) +b
```



------

# Course 2 - Improving Deep Neural Networks: Hyper-parameter tuning, Regularization and Optimization

## Week 1

- **train / dev / test**

- number of hidden layers and each unit in them 

- learning rates 

- activation function
- etc. 

### **Train / dev / test set:**

- small data = 70/30 or 60/20/20

- Large data (1M) : 98/1/1

dev and training = same distribution. 
not having a test set might be okay. (train/test is wrong and -> train /dev is correct one. some people use the wrong terms)

### **Bias / Variance**

- under fit = high bias
- over fit = high variance

metrics to observe this: compare to human error (or Optimal Bayes error) 
it is possible to have high bias and high variance at the same time. 

### **Basic recipe for ML**

<img src="Deep Learning Specialization Courses 1,2md.assets/image-20200801190159907.png" alt="image-20200801190159907" style="zoom:40%;" />

bias / variance trade off (back in the ML era)in deep we don't have this.

###  Regularization

L2 regularization in logistic regression = (lambda * || w ||2 2 )/ (2m) = Euclidian distance 
l1 in logistic regression = ||w|| don't have power 2 in L1 

the picture below is the Frobenius norm

<img src="Deep Learning Specialization Courses 1,2md.assets/Image.png" alt="Image" style="zoom:68%;" />

L2 = aka Weight decay because we make weights smaller 

### **Why Regularization prevent Overfitting**

it uses the linear part of tanh for example so that the model cannot perform all sorts of non linearity 

### **Dropout regularization**

inverted drop out make a matrix of 0 / 1 with sparsity of you probability with shape of Activation matrix
then multiply it element wise to activation function

```python
keep_prob = 0.8
A3 /= keep_prob
```

in making prediction in test time, we don't use dropout .

### **Understanding Dropout**

#### Other regularization

- data augmentation (flip horizontally - zoom - etc. )
- Early stopping (makes mid-size ||w||2,f ( it's easy to do, orthogonality problem)

#### **Normalizing inputs (speed up)**

2 step : 

1. subtract mean
2. normalize variance

why normalize? skewed gradients are harder to converge.

### **Vanish / explode gradients**

If our model is too deep and:

W>I (identity matrix) -> exploding gradients

W<I (identity matrix) -> vanishing gradients

### Weight initialization for deep

- partial solution : careful initialization

- set Variance of w to be ```sqr(2/n)``` in ReLu (Xavier initialization)
- if ```tanh => sqr (1/n)``` then this is a good hyper-parameter to tune 

### **Numerical Approximation**

```g =? [f(x-e) + f(x+e)] / 2e ``` is a good approximation 

**Gradient checking** 
W dW
and the loop on every j(theta) and calculate approx. gradient
then calculate Euclidean distance and it should be to the order of your epsilon

### Grad check implementation note

don't use in training - only debug remember regularization term. doesn't work with dropout 

## **Weak 2 - improving deep neural net** 

- mini batch. cost will be calculated after each mini batch.
- epoch= every iteration though the dataset
- use powers of 2 in mini batch size : 512, 256, 128, etc (for memory efficiency) 

### **Exponential weighted averages**


$$
V_t = B * V_t-1 + (1 - B) V_t
$$


how many data are we examining on each average? $$\frac{1}{1-B}$$ = number of averages

faster and memory efficient 
need bias correction because V0 is 0 and we use another formula $$\frac {v}{1- b^t}$$

### Gradient descent momentum

 works with exponential weighted averages to gradient descent. 

<img src="Deep Learning Specialization Courses 1,2md.assets/Image-1596294398635.png" alt="Image" style="zoom:38%;" />

### **Root mean squared prop (RMS PROP)**

<img src="Deep Learning Specialization Courses 1,2md.assets/Image-1596294409940.png" alt="Image" style="zoom:38%;" />

### Adam

adam = rmsprop + momentum 

adaptive moment estimation 

### **learning rate decay**

it helps. there a multiple kinds of formulas. 

### **problem of local optima**

 plateaus = the area when slope is zero for a large area 

## Week 3

- Hyper parameter tuning
  - Tuning process
    - alpha, beta (beta 1, beta 2, epsilon), # of layers, # of hidden units, learning rate decay, mini batch size
    - alpha, momentum term, # of layers, learning rate decade. are more important
    - don't try grids! Use random values instead.
    - Coarse to find: find some good samples of Hyper parameter, then zoom in that area and use more hyper parameters.
  - Using an appropriate scale to pick hyper parameters.
    - use random numbers in a specific range. you should do a appropriate scale and well distributed scale to ensure that you have the right numbers.
    - beta: you can't make a good range between (0.9 to 0.999). but you can make a (0.1 to 0.001) and then use 1-beta to make the correct list. (don't use linear scale) distributes more data near 1. 
  - Pandas vs Caviar
    - final tips and tricks: tuning is different on different domains.
    - intuitions get stale . re-evaluating occasionally 
    - babysitting one model over days, if you **don't have computational resources.** (panda approach)
    - or you can train many models in parallel (caviar approach) if you have a lot of **processing power**
- **Batch normalization**
  - Normalizing activations in a network
    - normalizing input features are useful. can we norm any hidden layers input? (yes) should we normalize Z or A ? (debatable, we use A in this course)
    - Batch norm will make the activation to have a normal distribution range. We can control mean and width of this normal distribution by Gamma and beta hyper parameters. 
    - <img src="Deep Learning Specialization Courses 1,2md.assets/Image-1596294782323.png" alt="Image" style="zoom:38%;" />
  - fitting a batch norm into a neural net
    - It was easy. no complications occurred. 
  - **why it works?**: it make you algorithm faster, add a small noise to it, has small regularization effect. with using a batch norm we can generalize better because of the mean and standard deviation of parameters. it computes beta and gamma one mini batch at a time. not on the whole dataset 
  - **BN at test time** 
    - bn used on one mini batch at a time, while in test we need to predict one record at a time 
    - we don't have mini batches on test time, so we make an estimated average on Mu and Standard dev. and use this for test time.
- Multi class classification
  - Softmax regression
  - <img src="Deep Learning Specialization Courses 1,2md.assets/Image-1596294903583.png" alt="Image" style="zoom:38%;" />
  - softmax gets a vector not a data.
  - it's in contrast with hard max function

