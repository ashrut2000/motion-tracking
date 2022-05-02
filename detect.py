import numpy as np
import cv2 as cv2

# def detect(frame):
#     #defining the HSV values for the green color
#     greenLower = (29, 86, 6)
#     greenUpper = (64, 255, 255)
    
#     output = frame.copy()
#     gray = cv2.cvtColor(np.float32(frame), cv2.COLOR_BGR2GRAY)
#     cv2.imshow('gray', gray)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, greenLower, greenUpper)
#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)

#     # find contours in the mask and initialize the current
#     # (x, y) center of the ball
#     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE)[-2]
#     centers=[]

#     # only proceed if at least one contour was found
#     if len(cnts) > 0:
#         # find the largest contour in the mask, then use
#         # it to compute the minimum enclosing circle and
#         # centroid
#         c = max(cnts, key=cv2.contourArea)
#         ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        

#         # only proceed if the radius meets a minimum size
#         if (radius < 200) & (radius > 1 ) : 
#             # draw the circle and centroid on the frame,
#             # then update the list of tracked points
            
#             centers.append(np.array([[x], [y]]))
#     cv2.imshow('contours', mask)
#     return centers


def detect(frame):
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)
    #output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    centers=[]

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)



        # only proceed if the radius meets a minimum size
        if (radius < 100) & (radius > 5 ) : 
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            centers.append(np.array([[x], [y]]))
    cv2.imshow('contours', mask)
    return centers 