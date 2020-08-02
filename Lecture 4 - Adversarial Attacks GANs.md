# Lecture 4 - Adversarial Attacks GANs

c2m2 c2m1 courses are taken 

- Adversarial Attacks 
  - attack in the blind spots 2013
  - Defense
  - why vulnerable

**attack in the blind spots 2013**
goal : find an image that is not iguana but will be classified as iguana 

rephrase goal: find x (no constrains) that y^ is iguana define loss function optimize the image (we don't train) 

goal2: find a cat that looks like iguana 

<img src="Lecture 4 - Adversarial Attacks GANs.assets/Image.png" alt="Image" style="zoom:38%;" />

<img src="Lecture 4 - Adversarial Attacks GANs.assets/Image-1596343745643.png" alt="Image" style="zoom:38%;" />

## **defenses** 

2 type of attack: 

- target attacks
- non Target attack 

2 kinds of attacker knowledge

- white box
- black box 

estimate of numerical gradients can achieved by changing input layer a little bit and see how the output looks like 

transfer ability is seen in adversarial attacks. that means if you have an animal classifier that you don't have access to it, you can build you animal classifier and make an adversarial attack, then this attack will probably works on the other network. 

- solution 1 : create a SafetyNet
- solution 2 : train on labelled adversarial examples 

## **why vulnerable** 

why they exists, because of the linearity 

### insights:

if w is large the ```x* != x``` (so we compute sign(w) as X grows in Dimension the impact on y^ of +epsilon * sing(w) increase
why these exist in image? because of high dimensionality of images 

```x* = x + e * simg( delta x , y(w,x,y) )```

the more W is higher, its more vulnerable 

**DO neural net understand data ?**

1. motivation endowing computers with an understand our world 
   1. collect of data use it to generate similar data from scratch
   2. intuition: number of parameters of model << amount of data
2. g/d game
3. training GANs
4. nice results
5. evaluating GANs

application of GANs = generating patient record turning satellite images to map

loss of GANs are not stated here. 