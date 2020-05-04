# Lecture 5 AI and Healthcare

## health care 

what q can we ask?

1. descriptive 
2. diagnostic (why?)
3. predictive (what will happen?)
4. perspective (what should we do?)

## research

3 case studies 

### 1d ecg

arrhythmia detection in important. electrical activities of heart. Zio patch (up to 2 weeks = 1.6 billion heart beep) 

the burst of data will help automatic detection aps grow

#### challenges 

difficult to diagnose with single lead ECG (normally we do that with 12 leads)

differences are quite subtle in heart arrhythmia 

1d conv net over time dimension of input

#### conv net

residual nets(short cuts for very deep net)

#### data 

64 k ecg records (600x bigger than mit-bih)

#### eval

surpass human level by 3 percent ! 

#### impact

coninously monitor patient 

apple watch 

### 2d chest x ray

detect nomina from chest x-rays pictures

2 bilion per year images 

#### arch

- 2d cnn 224*224
- pretrained on image net
- 121 layer dense net 

dense net : connect all of the layers together (all have shortcut to eachother)

#### dataset

112,120 frontal view x-ray 30k patient (lrgest sep 2017)

nlp systmes that reads reports 

420 test set with stanfird x-ray expert

#### eval

we don't have ground truth so we check them with eachother and compute f1 score

chex net (435 and expert 395)

#### model interpretation 



### 3d mri scan



## how can be involved?

