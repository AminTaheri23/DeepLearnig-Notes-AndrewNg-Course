# lecture 2 Intuition of DL

**lecture 1 is class intro and logistics and I didn't write note about it**

day and night Classification:
**Goal:** given an image, Classify if it is a day(0) or night(1)

- Data?  how much?
  -  think about the complexity of the problem(compare it to other tasks)  in this case we compare it to the the cat/non cat problem and it's easier than that, so we need less data, like 10 K image
  - Test train split = for small data sets like this, 80 to 20 is good, but if we had 100 K or 1 M the proportion of test will decrease, something like 98 to 2 percent.
  - Bias? We should have balanced data, half of them day, another half night
- input?
  - how much resolution? (as low as a human can classify them) high res images will make the learning slower. also keep in mind that if the complexity of the problem rises, we need more high res images.
- output?
  - 0 or 1
  - last activation ? sigmoid (because of between 0 or 1)
- architecture?
  - Shallow net will do the job pretty well
- Loss?
  -  logistic regression 

## Goal Face Verification

- data? name + pic
- input ? the camera input.
- res ? 400 * 400 (face is more complex and need more pixels to differ it.
- architecture? encoding image to a vector
- loss? training? same person pictures need to have similar vector and different persons need to have difference vectors. so we use triplets: 

<img src="Lecture 2 -  Deep Learning Intuition.assets/Image.png" alt="Image" style="zoom:38%;" />

Triplet loss is:

<img src="Lecture 2 -  Deep Learning Intuition.assets/Image-1596342538924.png" alt="Image" style="zoom:38%;" />

<img src="Lecture 2 -  Deep Learning Intuition.assets/Image-1596342549517.png" alt="Image" style="zoom:38%;" />

k nn = classification / regression / supervised = look at the neighbors

k means = cluster / unsupervised / centroids 

## Style transfer

input is a random image, if we use the content image, we will bias the output to the content image

we use trained network such as imagenet we update the image, not the parameters.

loss is = || content g - content c || + || style g - style c |

<img src="Lecture 2 -  Deep Learning Intuition.assets/Image-1596342764952.png" alt="Image" style="zoom:38%;" />

<img src="Lecture 2 -  Deep Learning Intuition.assets/Image-1596342773402.png" alt="Image" style="zoom:38%;" />

