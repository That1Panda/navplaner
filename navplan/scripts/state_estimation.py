#!/usr/bin/env python3
from re import S
import rospy
import numpy as np
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension

rospy.init_node("state_estimation")

gps_observation = np.zeros((3,1)) # x, y, theta
steering = 0
velocity = 0


# Subscribe to the gps, steering, and velocity topics named below and update the global variables using callbacks
# /gps
# /car_actions/steer
# /car_actions/vel

def steerCallback(actions_message):
    global steering
    steering=actions_message.data

def velCallback(actions_message):
    global velocity
    velocity=actions_message.data

def gpsCallback(gps_message):
    global gps_observation
    gps_observation=gps_message.data


steer_topic="/car_actions/steer"
steer_subscriber=rospy.Subscriber(steer_topic,Float64,steerCallback)
    
vel_topic="/car_actions/vel"
vel_subscriber=rospy.Subscriber(vel_topic,Float64,velCallback)

gps_topic="/gps"
gps_subscriber=rospy.Subscriber(gps_topic,Float64MultiArray,gpsCallback)

# Publisher for the state
state_pub = rospy.Publisher('vehicle_model/state', Float64MultiArray, queue_size=10)

r = rospy.Rate(10)

# Initialize the start values and matrices here
car_state=np.zeros((3,1))
t=0.1
l=4.9


H=np.eye(3)
Q=np.eye(3)*100
I=np.eye(3)
esm_cov=np.zeros((3,3))
car_state_pred=np.zeros((3,1))
# Create msg to publish#
current_state = Float64MultiArray()
layout = MultiArrayLayout()
dimension = MultiArrayDimension()
dimension.label = "current_state"
dimension.size = 3
dimension.stride = 3
layout.data_offset = 0
layout.dim = [dimension]
current_state.layout = layout
while not rospy.is_shutdown():
    # Create the Kalman Filter here to  estimate the vehicle's x, y, and theta

    car_state_pred = car_state+np.array([   [velocity*np.cos(car_state[2][0])*t],
                                            [velocity*np.sin(car_state[2][0])*t],
                                            [velocity*np.tan(steering)*(t/l) ]  ])#3x1

    jacobian=np.array([  [1,0,(-1*velocity*np.sin(car_state[2][0])*t)],
                [0,1,(velocity*np.cos(car_state[2][0])*t)],
                [0,0,1]  ]) #3x3

    pred_cov=np.dot(np.dot(jacobian,esm_cov),jacobian.T)#3x3
    first=(np.dot(pred_cov,H.T))
    second=np.linalg.inv((np.dot(np.dot(H,pred_cov),H.T)+Q).astype('float64'))
    kg=np.dot(first,second)
            #        3x3      3x3                       3x3     3x3      3x3    3x3
    car_state=(car_state_pred+np.dot(kg,(gps_observation-car_state_pred)))
            #  3x1            3x3       3x1
    esm_cov=np.dot((I-np.dot(kg,H)),pred_cov)
    #print(car_state_pred.shape,jacobian.shape,pred_cov.shape,kg.shape,esm_cov.shape,)
    
    current_state.data = car_state

    state_pub.publish(current_state)
    r.sleep()