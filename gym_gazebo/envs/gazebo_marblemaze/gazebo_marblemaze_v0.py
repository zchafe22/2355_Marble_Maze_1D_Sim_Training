import gym
import rospy
import roslaunch
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError

from datetime import datetime

from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState

from geometry_msgs.msg import Twist
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from gym.utils import seeding

import math

import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber

import numpy as np

from std_srvs.srv import Empty
import copy
import os

from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState


class GazeboMarbleMazev0Env(gazebo_env.GazeboEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/maze_pkg/launch/1dmodel.launch")

        # Define end conditions, TODO angle end condition needs to be more restrictive angle<2? position < wall left+something?
        THETA_THRESHOLD_DEG = 4
        self.theta_threshold_radians = THETA_THRESHOLD_DEG * 2 * math.pi / 360
        self.v_threshold_m = 10
        self.x_threshold_high = 0.09
        self.x_threshold_low = -0.04

        self.prev_err = 0

        self.x_goal = 0.04

        self.step_count = 0

        # Setup pub/sub for state/action
        self.joint_pub = rospy.Publisher("/trough/rev_position_controller/command", Float64, queue_size=1)
        self.trough_sub = message_filters.Subscriber('/trough/joint_states', JointState)
        self.trough_sub.registerCallback(self.get_trough_callback)
        self.ball_sub = message_filters.Subscriber('gazebo/model_states', ModelStates)
        #self.ball_sub = message_filters.Subscriber("/wheel/camera1/image_raw", Image)
        self.ball_sub.registerCallback(self.get_ball_pos_callback)
        self.bridge = CvBridge()
        
        # Image is comented because we are polling for images instead of using callbacks
        #self.ball_sub = message_filters.Subscriber("/wheel/camera1/image_raw", Image)

        # Gazebo specific services to start/stop its behavior and
        # facilitate the overall RL environment

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.set_link = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
        rospy.wait_for_service('/gazebo/set_link_state')

        # Setup the environment
        self._seed()
        self.action_space = spaces.Discrete(2)

        high = np.array([
            self.v_threshold_m * 2,
            np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-high, high)

        # State
        self.ball_pos_x = None
        self.trough_vel = 0
        self.trough_pos = None

        # Round state to decrease state space size
        self.num_dec_places = 2

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_ball_pos_callback(self, data):
        self.ball_pos_x = data.pose[1].position.x
        

    def get_ball_pos_callbackw(self, img_msg):
        '''
        @brief Process image
        @retval returns ball position in trough
        '''
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
        #rospy.loginfo('Image acquired')
        cv2.imshow("Raw image", output)
        #rospy.loginfo('x: ' + str(x) + 'y:' + str(y))
        cv2.waitKey(3)
        self.ball_pos_x = x
        return 

    def set_trough_position(self, ball_pos_x):
        x_desired = 150
        
        x_err =  x_desired - ball_pos_x
        deriv = (x_err - self.prev_err)

        Kp = -0.0002*200
        Kd = -0.00012

        self.trough_vel = Kp*x_err + Kd*deriv

        print('\n------\n') 

        self.prev_err = x_err

        self.joint_pub.publish(self.trough_vel)

    def step(self, action):
        # Unpause simulation to make observations
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # Wait for data
        x_pos = None
        while x_pos is None:
            x_pos = self.ball_pos_x

        # Pause
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # Take action
        if action > 0.5:
            self.trough_vel += 0.02
        else:
            self.trough_vel += -0.02
        
        self.joint_pub.publish(self.trough_vel)

        #print(action)
        #print(self.trough_vel)
        
        #print('step')
    
        # Define state, TODO change
        state = [self.ball_pos_x,self.trough_vel]


        # Check for end condition
        done =  x_pos < self.x_threshold_low \
                or x_pos > self.x_threshold_high \
                or abs(self.trough_vel) > self.v_threshold_m
        done = bool(done)
        
        #print(x_pos)

        #TODO reward = 1/(err+1)?
        if not done:
            reward = 0.5 + 1.0/(100*abs(self.x_goal-self.ball_pos_x)+1)
            #reward = 1.0
        else:
            reward = 0

        self.step_count += 1

        if self.step_count == 800:
            print('Max step count')

        # Reset data

        return state, reward, done, {}
    
    
    def reset_ball_pos(self):    
        state_msg = ModelState()
        state_msg.model_name = 'ball'
        state_msg.pose.position.x = -0.01
        state_msg.pose.position.y = -0.034496
        state_msg.pose.position.z = 0.155751
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0
        rospy.wait_for_service('/gazebo/set_model_state')
        print('ball reset')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )
        except rospy.ServiceException:
            print( "Service call failed")
    
    def get_trough_callback(self,data):
        self.trough_pos = data.position
        #print(data.velocity)
    
    def reset_trough_angle(self):
        print('resetting trough')
        while self.trough_pos == None:
            pass

        offset = 0.2
        while abs(self.trough_pos[0]-offset) >= 0.001:
            #print(self.trough_pos[0])
            if self.trough_pos[0] - offset < 0:
                self.joint_pub.publish(0.1)
            else:
                self.joint_pub.publish(-0.1)
        self.joint_pub.publish(0)
        print('done resetting trough')
        time.sleep(0.2)
        return

    def reset(self):
        # Reset world
        rospy.wait_for_service('/gazebo/set_link_state')

        # Unpause simulation to make observations
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.reset_trough_angle()
        time.sleep(0.2)
        self.reset_ball_pos()

        # Wait for data
        x_pos = None
        while x_pos is None:
            x_pos = self.ball_pos_x

        # Pause
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        #print(str(x_pos))

        state = [x_pos, self.trough_vel]

        self.ball_pos_x = None
        self.trough_vel = 0

        self.step_count = 0

        # Reset data
        print('did_reset-------------------------------------------------')
        return state
