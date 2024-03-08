import gym
import rospy
import roslaunch
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError

from datetime import datetime

from geometry_msgs.msg import Twist
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from gym.utils import seeding

import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber

import math
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
        self.x_threshold_m = 0.05

        # Setup pub/sub for state/action
        # self._pub = rospy.Publisher('/cart_pole_controller/command', Float64, queue_size=1)
        # rospy.Subscriber("/cart_pole/joint_states", JointState, self.callback)

        self.joint_pub = rospy.Publisher("/trough/rev_position_controller/command", Float64, queue_size=1)
        self.trough_sub = message_filters.Subscriber('/trough/joint_states', JointState)
        # Image is comented because we are polling for images instead of using callbacks
        #self.ball_sub = message_filters.Subscriber("/wheel/camera1/image_raw", Image)
        self.trough_pos = 0.0

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
            self.x_threshold_m * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-high, high)

        # State
        self.current_vel = 0
        self.data = None

        # Round state to decrease state space size
        self.num_dec_places = 2

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_ball_pos_callback(self, img_msg):
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
        rospy.loginfo('Image acquired')
        cv2.imshow("Raw image", output)
        rospy.loginfo('x: ' + str(x) + 'y:' + str(y))
        cv2.waitKey(3)
        self.ball_pos_x = x
        return 

    def set_trough_position(self, ball_pos_x):
        x_desired = 150
        self.dt = ((datetime.now() - self.t).microseconds)/1000.0
        self.t = datetime.now()

        x_err =  x_desired - ball_pos_x
        deriv = (x_err - self.prev_err)/self.dt

        Kp = -0.0002
        Kd = -1.2

        self.trough_pos_write = Kp*x_err + Kd*deriv

        print('\n------\n') 
        rospy.loginfo('ball pos read: '+ str(ball_pos_x))
        rospy.loginfo('trough x error: '+ str(Kp*x_err))
        rospy.loginfo('trough derivative error: '+ str(Kd*deriv))
        rospy.loginfo('trough pos write: '+ str(self.trough_pos_write))

        self.prev_err = x_err

        self.joint_pub.publish(self.trough_pos_write)

    def step(self, action):
        # Unpause simulation to make observations
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # Wait for data
        data = None
        while data is None:
            try:
                data=rospy.wait_for_message('/wheel/camera1/image_raw"', Image, timeout=5)
            except:
                pass

        # Pause
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        ball_pos_x = self.get_ball_pos_callback(data)

        return [0,0,0,0],0,False

        # Take action
        self.set_trough_position(ball_pos_x)
        return

        # Define state, TODO change
        x = self.data.position[1]
        x_dot = self.data.velocity[1]
        theta = math.atan(math.tan(self.data.position[0]))
        theta_dot = self.data.velocity[0]
        state = [round(x, 2), round(x_dot, 1), round(theta, 2), round(theta_dot, 0)]

        # Limit state space
        state[0] = 0
        state[1] = 0

        # Check for end condition
        done =  x < -self.x_threshold_m \
                or x > self.x_threshold_m \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        #TODO reward = 1/(err+1)?
        if not done:
            reward = 1.0
        else:
            reward = 0

        # Reset data
        self.data = None
        return state, reward, done, {}

    def reset(self):
        # Reset world
        rospy.wait_for_service('/gazebo/set_link_state')
        return [0,0,0,0]
        self.set_link(LinkState(link_name='pole'))
        self.set_link(LinkState(link_name='cart'))

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # Wait for data
        data = self.data
        while data is None:
            data = self.data

        # Pause simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # Process state
        x = self.data.position[1]
        x_dot = self.data.velocity[1]
        theta = math.atan(math.tan(self.data.position[0]))
        theta_dot = self.data.velocity[0]
        state = [round(x, 2), round(x_dot, 1), round(theta, 2), round(theta_dot, 0)]

        # Limit state space
        state[0] = 0
        state[1] = 0

        self.current_vel = 0

        # Reset data
        self.data = None
        return state
