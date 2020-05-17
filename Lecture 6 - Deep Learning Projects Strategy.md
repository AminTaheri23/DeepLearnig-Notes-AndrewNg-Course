# Lecture 6 - Deep Learning Projects Strategy

trigger word detection system. we are trying to build a trigger word detection to "Robert Turn On" or RTO for short

## how to research

- choose 3 paper and skim them. then focus on one and go deeper. do this in iteration 
- 10 papers : basic understanding 
  50 paper : descent understanding
  100 paper : very good understanding
- talking to experts. if you take time and read it and didn't understand it. you can email or get in touch with authors (but don't do it often)
- collect you data in 2 to 3 days

<img src="Lecture 6 - Deep Learning Projects Strategy.assets/image-20200517195353333.png" alt="image-20200517195353333" style="zoom:67%;" />

## what data do you collect ?

get a data from friends and colleague. 100 examples of 10 s of data.

RTO takes 3 seconds. we clip the 10 second data to clips of 3 seconds clips. 3000 of 3s data.

make dev set balanced to 0 or 1

you can label the data one, a bit more after the RTO. (like 0.5 seconds) 

two measures of trigger detection:

1. how often does it activate after it was said RTO.
2. how often it waked without saying RTO

<img src="Lecture 6 - Deep Learning Projects Strategy.assets/image-20200517195627818.png" alt="image-20200517195627818" style="zoom:67%;" />

we add bunch ones to even classes

## over fitting

we can use data augmentation. we can collect back ground noise data. 

use random words. with RTO

 ## how much time we need to gather data

- [x] use mic of your friends (4 hrs)
- [ ] download form online sources (6 hrs) we can not trust the source
- [x] Croud sourcing (Eg. Mechanical turk) (1 week)

brain storm some options and prioritize

