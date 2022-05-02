import cv2
from detect import detect
from KalmanFilter import ExtendedKalmanFilter
from KalmanFilter import LinearKalmanFilter
import math
import time
from imutils.video import VideoStream
import numpy as np
import matplotlib.pyplot as plt

def main():
    f=input('''
    Which filter to use?
    Type l for linear 
    tyle nl for non-linear
    
    
    ''')

    # Create opencv video capture object
    VideoCap = cv2.VideoCapture('greenball.mp4')
    
    if(f=='l'):
       

        KF = LinearKalmanFilter(sigma=0.01)
    else:
        KF = ExtendedKalmanFilter(sigma=0.01)
    err_list=[] #list to keep track of errors
    


    while(True):
        # Read frame
        
        ret, frame = VideoCap.read()
        if(ret==False):
            plotdata(err_list)
    
        # Detect object
        centers = detect(frame)
        
                

        # If centroids are detected then track them
        if (len(centers) > 0):

            # Draw the detected circle
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 40, (0, 191, 255), 2)

            # Predict
            (x, y) = KF.predict()
            #update
            (x1, y1) = KF.update(centers
            #get mean square error using the function
            err=get_mean_square_error(centers,x1,y1)
            #add the error to a list to plot later
            err_list.append(err)
            # Draw a rectangle as the updated object position
            cv2.rectangle(frame, (int(x1 - 45), int(y1 - 45)), (int(x1 + 45), int(y1 + 45)), (0, 0, 255), 2)
            cv2.putText(frame, "Updated Position"  ,(10,100 ), 0, 0.5, (0, 0, 255), 2)
            #cv2.putText(frame, "Predicted Position", (10,60), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (10,20), 0, 0.5, (0,191,255), 2)

        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        cv2.waitKey(1)
    
    plotdata(err_list)

#function to get the distance between predicted point and measured point(MSE)
def get_mean_square_error(centers,x1,y1):
    x=centers[0][0]
    y=centers[0][1]
    return np.sqrt((x-x1)**2 + (y - y1)**2)

#plot datas
def plotdata(err_list):
    
    plt.plot(err_list, lw=0.1)
    plt.ylim(0, 80)
    plt.grid('on')
    plt.title("Error MSE(True Position - Predicted Position)")
    plt.xlabel("Frame No")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # execute main
    main()