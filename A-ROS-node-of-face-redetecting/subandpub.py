#!/usr/bin/env python

import sys
import os
import rospy
import math
import cv2
import glob
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class imag_pub:

    def __init__(self):
        ## initialize
        rospy.init_node('imag_pub', anonymous=True)
        self.bridge= CvBridge()
        self.pub = rospy.Publisher("sensor_msgs/Image", Image, queue_size=10)
        self.sub = rospy.Subscriber("/front_cam/camera/image", Image, self.callback)
        
        self.rate = rospy.Rate(0.5) # 1 Hz
        
        ## loads the required XML classifiers
        

    def callback(self, data):
        rospy.init_node('imag_pub', anonymous=True)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)                    
        else:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            file_path = '/home/zangxuechen/opencv-3.4.3/build/data/haarcascades/haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            face_cascade.load(file_path)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            ## detects faces & returns positions of faces as Rect(x,y,w,h)
            ## draws circles around the detected faces
            for (x,y,w,h) in faces:
                square =(w/3)**2 +(h/3)**2
                radius =int(math.sqrt(square))
                cv2.circle(cv_image,(x+w/2,y+h/2),radius,(0,0,255),2)

            ## converts OpenCV image to ROS image    
            out_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")      
        
            ## publishes the image with detected faces 
            self.pub.publish(out_image)
            self.rate.sleep()

def main(argv):
    imag_pub()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
  try:
    main(sys.argv)
  except rospy.ROSInterruptException:
    pass
