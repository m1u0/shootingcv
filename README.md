# shootingcv: basketball shooting release point detection

## how it works

given a video of a player shooting a basketball, shootingcv combines pose and object detection to identify the frame in which the ball leaves the player's hand. it functions by using pose and object detection along with a set of criteria that describe when the player is shooting and the ball has left their hand. It analyzes each frame, marking the first frame in which the ball has left the person's hand.

## applications

shootingcv can be used for comparing shooting forms, either with the same player (to determine consistency) or with different players (to identify areas of improvement). Using batchshootingcv.py, the user can input a folder of shooting clips that the software will analyze and align at the release frame. This allows for direct comparison at every stage of the shooting form.

## limitations / roadmap

Currently, the criteria for the release frame need to be refined. Mainly, extraneous actions such as spinning the ball to oneself results in false positives of the detection algorithm. one possible way to fix is this is to create a confidence rating for the release frame, analyzing the entire clip, and recording the highest confidence clip.

I wish to implement a method of statistically analyzing shooting form in the future (e.g. with angle velocity measurements). However, since the pose detection is currently only a 2 dimensional wireframe, I cannot analyze angles without perspective distortion. I am looking into ways to solve this, such as 3d pose estimation programs, multi camera setups, and 2d to 3d image transformations.
