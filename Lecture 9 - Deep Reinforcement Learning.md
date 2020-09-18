# Lecture 9

Function approximators. 



## Outline 

- Motivation
- Recycling is good: an intro to Reinforcement Learning
- Deep Q-learning
- Application of Deep Q-Learning: Breakout (Atari)
- Tips to train Deep Q-Network
- Advanced topics



## Motivation 

Deep Reinforcement learning is new. but reinforcement learning is an old topic

two examples of Deep RL

1.  Alpha Go
2. Human level control with deep RL



## Game of go

maximize your territory. white and black buttons. 19 * 19 board. How would we solve Go with classical supervised learning?

- input the picture of the game and predict the next move from a professional player. ( the problem is that the # of states are two big 10^170. it's about the strategy not discriminating and we have long term strategy )
- RL : automatic learning to make good sequence of decisions. (in RL we don't have the ground truth)

in RL we give the agent a reward and the agent tries to do the task by trial and error. 

example of RL applications

- Robotics
- Games 
- Advertisement



## Recycling is good: an Introduction to RL

States and rewards 

goal : maximize the return(rewards)

types of states

1. starting states

2. normal states

3. terminal states

in this games we can have limited number of moves. and we can have possible actions.

discounted return = every time passes -1 return. 

return = the sum of the returns with no penalties. 

<img src="Lecture 9 - Deep Reinforcement Learning.assets/image-20200918172113517.png" alt="image-20200918172113517" style="zoom:80%;" />

q table = the data of my knowledge 

we make a tree of states and actions. we calculate the discounted return for every path. 

in each action we calculate the long term reward, because we want to maximize it and we will update the Q-table. 

 <img src="Lecture 9 - Deep Reinforcement Learning.assets/image-20200918172406070.png" alt="image-20200918172406070" style="zoom:80%;" />

all of the optimal Q-tables should follow this equation. 

<img src="Lecture 9 - Deep Reinforcement Learning.assets/image-20200918172650928.png" alt="image-20200918172650928" style="zoom:80%;" />

we have policy pi as above. this is the decision making. tells us what to do. 

**why deep learning is helpful?**

because the q-table is very large for some applications and we can't afford to have such big matrixes (for go it is something like 10^170 * 19\*19 matrix)

<img src="Lecture 9 - Deep Reinforcement Learning.assets/image-20200918173203488.png" alt="image-20200918173203488" style="zoom:80%;" />

## Deep Q-learning

<img src="Lecture 9 - Deep Reinforcement Learning.assets/image-20200918173250159.png" alt="image-20200918173250159" style="zoom:80%;" />

we don't have labels this is a regression problem. We use L2 loss function for this purpose. 

the labels are moving. we iterate with the bellman equation. we guess at the first step, then try to reach that. by doing so we will have a better Q-net. when we have a better Q-net we can make better guesses and we will do this iteratively. 

we calculate bellman equation every time. 

we hack the way into learning with DRL. one hack is to fix the Q in the right side of the equation below. we should do this, other wise the network will go in an infinite loop. we fix the Q for 1 million or 100,000 iterations then Update it and repeat.

<img src="Lecture 9 - Deep Reinforcement Learning.assets/image-20200918174133722.png" alt="image-20200918174133722" style="zoom:80%;" />

episode = one game from start to the end

the one hard part of understanding and the part that is different from ordinary networks is that we should forward propagate twice, one for the S and one for the S'. because of the bellman equation that is a recursive function. 

<img src="Lecture 9 - Deep Reinforcement Learning.assets/image-20200918174557672.png" alt="image-20200918174557672" style="zoom:80%;" />

## Application of Deep Q-Network: Breakout (Atari)

Goal: destroy all of the breaks. 

<img src="Lecture 9 - Deep Reinforcement Learning.assets/image-20200918174748946.png" alt="image-20200918174748946" style="zoom:50%;" />

the network after a lot of iterations finds a trick to dig a tunnel in the breaks and finish the game faster. and this network finds it by its own and without supervision. 

we can get the position of every thing to the network, this approach is the feature based approach, another approach is to input the pixels to the network and give him the handle and let it play the game like we do!

we can not give one frame of the game to the network and ask it to generalize, because we are removing the moving objectives features. so we give it short clips (4 frames for example) to make its guess

pre processing ; 

- gray scale
- cropping
- history of 4 frames

architecture is conv layer. 

some tricks to learn better.

- keep track of terminal step ( we change the loss when at terminal step)
- experience replay: we might never see some states. 
- epsilon greedy action choose (exploration/exploitation trade off)

<img src="Lecture 9 - Deep Reinforcement Learning.assets/image-20200918180103229.png" alt="image-20200918180103229" style="zoom:80%;" />

we create an Experience Replay Database with all of our experiences. we sample from the sample of the Replay Memory. 

- data efficiency ( many epochs )
- decorrelate experiences 
- trading experience / exploitation 

we are now training on replay memory and not on the current experiment that we are having. 

### exploration vs exploitation

we always take the best action. we can explore more. 

5% of the time explore other wise exploit the knowledge 

the final pseudocode with tricks and hacks

<img src="Lecture 9 - Deep Reinforcement Learning.assets/image-20200918190105300.png" alt="image-20200918190105300" style="zoom:80%;" />

### with and without human knowledge 

huge difference between human in/out of the loop of learning. 

some games have some analogies for humans that machines don't understand. also long games that requires a long strategy to win is also hard for DQN to overcome. 

## Advanced Topics

### before alpha go

- Tree searching 
- ...

calculating the score of each image.

two AI paly with each other. a fixed one and another one that is dynamic and try to overcome the fixed one. if the dynamic one wins for N streaks, then the dynamic one becomes the fix one and replicate himself to try this algorithm again. 

### POLICY GRADENTS:

they try to update policy function each time. 

### meta learning

train on similar tasks then very few updates should get us to the special task that we want

### imitation learning

defining the reward is hard. we have human to the rescue! and we try to imitate the human. 