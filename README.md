# Implementation of Lucas Kanade's Optical Flow
Complete implementation of Optical Flow with Lucas Kanade's algorithm on Python 3.8


Optical flow is the motion of objects between consecutive frames of sequence. Since, each video is a sequence of images every two consecutive frames can be given to optical flow algorithm which calculates the displacements of features spatially. <br />

### Assumptions of Optical Flow
Since the two consecutive frames captured from a video are taken with negligible time difference, we can assume :-
1) The displacement of any object is not large.
2) The intensity of the object doesn't change.
3) Neighboring pixels(containing the object) move in the same direction.

### Implementation
Using the above three assumptions, we calculate the motion of the objects in the given frames. <br />

#### Finding the features
- It is important to figure out which features to track. The object's Corners make a very good feature set that would dictate the movement of the object. <br />
- To detect the corner's we use Shi Tomasi's corner detection algorithm, since it performs better than Harris Corner Detector approach. <br />
- I have used cv2.goodFeaturesToTrack() api call to obtain corners by Shi Tomasi's approach. <br />

#### Calculating the flow of features
- From the assumptions stated above, we can consider that : *I(x,y,t) = I(x+dx, y+dy, t+dt)* <br />
Where, I(x,y,t) is the intensity at pixel (x,y) at time t and
       I( x+dx, y+dy, t+dt) is the same intensity of displaced corner after time dt. 
- By expanding the RHS with Taylor's series, we can cut down common terms and neglect the higher order derivatives since they become negligible. <br />
- So, we get : *f_x * u + f_y * v + f_t = 0*  <br />
Where, f_x is gradient w.r.to X direction <br />
       f_y is gradient w.r.to Y direction <br />
       f_t is change w.r.to time <br />

- Here we can't get u,v with single equation. So, we use the 3rd assumption stated above (neighbouring pixels would move similarly). <br />
- Hence, with a window of pixels (of size n), we get n * n equations that would help us solve for (u,v). <br />

- After getting (u,v), we plot the results.

### Results :- 
![](https://github.com/Utkal97/Object-Tracking/blob/main/Results/BasketBall_Seperate_Result.png)
![](https://github.com/Utkal97/Object-Tracking/blob/main/Results/Grove_Seperate_Result.png)

### Function calls :-

##### opticalFlow(old_frame, new_frame, window_size, min_quality=0.01) : 
1) old_frame and new_frame are two consecutive frames.
2) window_size is the size of window to be considered for neighbours (for populating equations for solving u,v).
3) min_quality is minimum quality to consider a feature

##### drawOnFrame(frame, U, V, output_filepath):
1) frame is the image onto which the displacement lines are drawn
2) U, V are displacements w.r.to X and Y directions respectively
3) output_filepath is the output file to which the result is to be saved to

##### drawSeperately(old_frame, new_frame, U, V, output_file):
1) old_frame and new_frame are two consecutive frames.
2) U, V are displacements
3) output_filepath is path to save the image


### Dependancies Required : 
CV2, Numpy, Matplotlib
