
import numpy as np
import matplotlib.pyplot as plt

class ExtendedKalmanFilter():
    #initialize values
    def __init__(self, sigma):
        
        self.x = np.array([0.,  #x
                           0.,  #y
                           0.,  # heading
                           0.,  # rotation
                           0.]).reshape(-1, 1) # distance
        #initial process error matrix. 
        self.p = np.diag([1. for i in range(5)])
        #transformation matix h
        self.h = np.array([[1., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0.]])

        self.r = np.diag([sigma/100, sigma/100])
    
        x=self.x[0,0]
        y=self.x[1,0]
        h=self.x[2,0]
        r=self.x[3,0]
        d = self.x[4,0]

        #matrix which linearizes our non linear system

        self.f= np.array([
                [1, 0, float(-d*np.cos(h) + d*np.cos(r + h)), float(d*np.cos(r + h)), float(-np.sin(h) + np.sin(h + r))],
                [0, 1, float(-d*np.sin(h) - d*np.sin(r - h)), float(d*np.sin(r - h)), float(np.cos(h) - np.cos(h - r))],
                [0, 0,                                   1,                   1, 0],
                [0, 0,                                   0,                   1, 0],
                [0, 0,           0, 0, 1]])
        self.q = 0.001*self.f.dot(self.f.transpose())



    def predict(self):
       
        x=self.x[0,0]
        y=self.x[1,0]
        h=self.x[2,0]
        r=self.x[3,0]
        d = self.x[4,0]
        # x, y, h, r, d = self.x

        new_x = x - d * np.sin(h) + d * np.sin(h + r)
        new_y = y + d * np.cos(h) - d * np.cos(h - r)
        new_h = h + r
        new_r = r
        new_d = d

        self.x = np.array([new_x, new_y, new_h, new_r, new_d]).reshape(-1, 1)
        self.p = self.f.dot(self.p).dot(self.f.transpose())+ self.q
        return self.x[0:2]


    def update(self, measurement):
        z = np.array(measurement).reshape(-1, 1)
        y = z - self.h.dot(self.x)

        s = self.h.dot(self.p).dot(self.h.transpose()) + self.r
        k = self.p.dot(self.h.transpose()).dot(np.linalg.inv(s))

        self.x = self.x + k.dot(y)
        self.p = (np.eye(5) - k.dot(self.h)).dot(self.p)
        return self.x[0:2]



class LinearKalmanFilter():
    def __init__(self, sigma):
        """
        Kalman Filter Constructor
        
    
            X: state matrix [[ x ],
                             [ y ],
                             [ x_velocity ],
                             [ y_velocity ],
                             [ x_acceleration ],
                             [ y_acceleration ]]

            << State Transition >>
                x <- x + x_velocity
                y <- y + y_velocity
                x_velocity <- x_velocity + x_acceleration
                y_velocity <- y_velocity + y_acceleration
                x_acceleration <- x_acceleration
                y_acceleration <- y_acceleration
                
        Args:
            sigma (float): measurement uncertainty
        """
        self.x = np.array([[0. for i in range(6)]]).reshape(-1, 1)
        self.f = np.array([[1, 0, 1, 0, 0, 0],
                           [0, 1, 0, 1, 0, 0],
                           [0, 0, 1, 0, 1, 0],
                           [0, 0, 0, 1, 0, 1],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        self.p = np.diag([999. for i in range(6)])
        self.h = np.array([[1., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0., 0.]])
        self.r = np.array([[sigma, 0.],
                           [0., sigma]])
        self.i = np.identity(6)
        self.q = 0.001*self.f.dot(self.f.transpose())

    def predict(self):
        """
        Prediction Step in Kalman Filter
        
        Algorithms:
            x = f*x
            p = f*p*f^T
        """
        self.x = self.f.dot(self.x)
        self.p = self.f.dot(self.p).dot(self.f.transpose())+ self.q
        return self.x[0:2]

    def update(self, measurement):
        """
        Update Step in Kalman Filter.
        
        Algorithms:
            Z = measurement matrix (2x1)
            y = z - h * x (error between  measurement and predicted value)
            s = h * p * h^T + r
            k = p * h^T * s^-1 (kalman gain)

            x = x_real = x + k * y
            p = p_real = (I - k * h)*p       
                
        Args:
            measurement (list): [x, y] coordinates
        """

        z = np.array(measurement).reshape(-1, 1)
        y = z - self.h.dot(self.x)
        s = self.h.dot(self.p).dot(self.h.transpose()) + self.r
        k = self.p.dot(self.h.transpose()).dot(np.linalg.inv(s))

        self.x = self.x + k.dot(y)
        self.p = (self.i - k.dot(self.h)).dot(self.p)
        return self.x[0:2]
