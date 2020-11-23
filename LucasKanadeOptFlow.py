import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt

''' 
    inRange checks whether the given cordinates line in the given image limits
 cordinates, limits are tuples i.e., (X,Y) 
'''
def inRange( cordinates, limits):
	x,y = cordinates
	X_Limit, Y_Limit = limits
	return 0 <= x and x < X_Limit and 0 <= y and y < Y_Limit

'''
    opticalFlow calculates the displacements in X and Y directions i.e., (u,v)
    given two consecutive images varying with time
'''
def optical_flow(old_frame, new_frame, window_size, min_quality=0.01):

    max_corners = 10000
    min_distance = 0.1
    feature_list = cv2.goodFeaturesToTrack(old_frame, max_corners, min_quality, min_distance)

    w = int(window_size/2)

    old_frame = old_frame / 255
    new_frame = new_frame / 255

    #Convolve to get gradients w.r.to X, Y and T dimensions
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    fx = cv2.filter2D(old_frame, -1, kernel_x)              #Gradient over X
    fy = cv2.filter2D(old_frame, -1, kernel_y)              #Gradient over Y
    ft = cv2.filter2D(new_frame, -1, kernel_t) - cv2.filter2D(old_frame, -1, kernel_t)  #Gradient over Time


    u = np.zeros(old_frame.shape)
    v = np.zeros(old_frame.shape)

    for feature in feature_list:        #   for every corner
            j, i = feature.ravel()		#   get cordinates of the corners (i,j). They are stored in the order j, i
            i, j = int(i), int(j)		#   i,j are floats initially

            I_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            I_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            I_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            b = np.reshape(I_t, (I_t.shape[0],1))
            A = np.vstack((I_x, I_y)).T

            U = np.matmul(np.linalg.pinv(A), b)     # Solving for (u,v) i.e., U

            u[i,j] = U[0][0]
            v[i,j] = U[1][0]
 
    return (u,v)



img1 = cv2.imread("basketball1.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("basketball2.png")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

_v, _u = optical_flow( img1, img2, 3, 0.05)

img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        u, v = _u[i][j], _v[i][j]
        if u and v:
            img2 = cv2.arrowedLine(img2, 
                                    (i, j), (np.ceil(i+u).astype('int64'), np.ceil(j+v).astype('int64')),
                                    (0, 255, 0),
                                    thickness=1
                                )
cv2.imshow('Optical Flow', img2)
cv2.waitKey(0)

cv2.imwrite('results.png', img2)