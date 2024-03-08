#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from std_srvs.srv import Empty
from datetime import datetime
import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import time

class CommandToJointState:

    def __init__(self):
        # PID parameters for movement
        self.center_pixel = 399
        self.prev_err = 0
        self.dt = (datetime.now() - datetime.now()).microseconds
        self.t = datetime.now()
        self.trough_pos_write = 0.0
        self.trough_pos_read = 0.0
        self.ball_pos_x = 0
        self.ball_pos_y = 0
        self.joint_name = 'rev'
        self.joint_state = JointState()
        self.joint_state.name.append(self.joint_name)
        self.joint_state.position.append(0.0)
        self.joint_state.velocity.append(0.0)
        self.bridge = CvBridge()
        self.joint_pub = rospy.Publisher("/trough/rev_position_controller/command", Float64, queue_size=1)
        self.trough_sub = message_filters.Subscriber('/trough/joint_states', JointState)
        self.ball_sub = message_filters.Subscriber("/wheel/camera1/image_raw", Image)
        ats = ApproximateTimeSynchronizer([self.trough_sub, self.ball_sub], queue_size=5, slop=0.1)
        self.reset_ball_pos()
        self.trough_sub.registerCallback(self.get_trough_pos_callback)
        self.ball_sub.registerCallback( self.get_ball_pos_callback)
        self.direction = 0 # 0 is CW; 1 is CCW
       # ats.registerCallback(self.pootLovato)

    def reset_ball_pos(self):    
        state_msg = ModelState()
        state_msg.model_name = 'ball'
        state_msg.pose.position.x = 0
        state_msg.pose.position.y = -0.025
        state_msg.pose.position.z = 0.18
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.loginfo('ball reset')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )
        except rospy.ServiceException:
            print( "Service call failed")

    def get_trough_pos_callback(self, msg):
        '''
        Moving the trough back and forth
        '''
        x_desired = 150
        self.dt = ((datetime.now() - self.t).microseconds)/1000.0
        self.t = datetime.now()

        x_err =  x_desired - self.ball_pos_x
        deriv = (x_err - self.prev_err)/self.dt

        Kp = -0.0002
        Kd = -1.0

        self.trough_pos_write = Kp*x_err + Kd*deriv

        print('\n------\n') 
        rospy.loginfo('trough pos read: '+ str(self.ball_pos_x))
        rospy.loginfo('trough x error: '+ str(Kp*x_err))
        rospy.loginfo('trough derivative error: '+ str(Kd*deriv))
        rospy.loginfo('trough pos write: '+ str(self.trough_pos_write))

        self.prev_err = x_err

        self.joint_pub.publish(self.trough_pos_write)

        
    def get_ball_pos_callback(self, img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, (0,0,0), (54, 255, 255))
        #gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        output = cv_image.copy()
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 2,20, 
                                    param1=50,
                                    param2=30,
                                    minRadius=15,
                                    maxRadius=40)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        rospy.loginfo('Image acquired')
        cv2.imshow("Raw image", output)
        rospy.loginfo('x: ' + str(x) + 'y:' + str(y))
        cv2.waitKey(3)
        self.ball_pos_x = x
        return 

if __name__ == '__main__':
    rospy.init_node('command_to_joint_state')
    command_to_joint_state = CommandToJointState()
    rospy.spin()