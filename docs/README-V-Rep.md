### Important notes ###

* Scenes - Anything loads with a scene on V-Rep. I understand it as an environment (although this is a key word used in V-Rep, which tells us the area that we are working on), which is the complete area, like the whole room or something where a lot might be happening in the simulation. Scene is loaded by the main script, which is explained later.
* Models - Scene consists of multiple models, which are small parts of the scene where something might be happening. Like model should be the smallest entity where a robot is interacting, or some objects are put together.
* Objects - these will be the blocks that we will use in the way of the robot. We need to set these objects as
	* collidable - meaning collision gets detected, 
	* detectable - visible in cameras or proximity sensors
These properties can be set using simSetObjectSpecialProperty function is part of the V-Rep API.
* Sensors - There are two types of sensors, vision sensors like camera and proximity sensors like kinect (I think, not sure we dont need proximity sensors as image is what we are going to use anyways). The quadcopter comes with the vision sensor already placed on it which we have to access and get the image frames from.

The important thing I understood while going through the documentation for V-Rep is, it is a simulation of the exact robotic scenario. You will place it somewhere and then send commands to it and it will execute. We need to remember that everything would be in reference may to the original position or amount of force would be calculated which will tell us how much the robot moved and so on. There might be a case that we might not get the GPS position at all. Then we might need to change our goal formulation which will essentially be the reward [formulation](#problem-formulation).

My previous explanation was a little obscure so let me try to explain it again. A robot usually works like this it has some in built hardware, which has first layer of device drivers, then ROS (Robot Operating System) installed, or may be a Linux based machine installed which has ROS running on it as an extension. So what usually happens is we SSH into the robot and send the commands through the connection. Now a simulator does the same behavior. So we may or may not get a lot of things (I am still checking on exact details), and everything for a robot becomes with respect to a reference frame. Like if we have to make a robot move, some come with a lot more structure, like a map and then we can give location to go to. But others don't and we have to specify move this joint by a certain degree which leads to certain amount of motion in the real world. Since this is a simulator there may or may not be a layer which can help us signify the motion amount. So underlying formulation or actions might change (these will be the results of the netowrk layer), but over that we would just need to discretize the system and run it. It can be move to point (x, y) or give it x amount of force.

### Problem Formulation ###

This is with respect to the Deep Q Learning network.

** Old formulation **
* Features - This will be defined by the image that we get from the camera on the quadcopter.
* Actions - Still to find how it is, but the idea is they create controlled motion in x and y plain.
* Goal - Either a GPS location from the starting point, in some reference behind the obstacle or away from the obstacle.
* Reward - This would be used for Q Learning. If we get precise geo locations then it could be logistic function over the euclidean distance between the points. If we only get degrees of movement for joints or something then we would need to think of how to get the distance from the goal state. If we go by strict images from the quadcoptor then we can preprocess this images and when there are no edges in the image that can be our goal state. And we can set up the reward based on the pixels that have a gradient (will explain this if not directly understandable).

** New formulation **
* Features - This can still be the camera image but I dont think it matters that much. A better feature is the orientation of the quadcopter which can be received by simGetObjectOrientation, this gives 3 values which are the alpha, beta and gamma (mostly x, y and z) orientation of the object. Giving these 12 numbers we would be able to learn the thrust of motors and orientation for the quadcopter.
* Actions - It is the thrust or the speed of with which the motors are running. We will have to set the amplitude on our own and multiply it between 0 and 1. This value will be learned from the network.
* Goal - GPS location of the object can be received, only problem here is that it keeps moving on the .
* Reward - logistic function over the euclidean distance from goal position multiplied by two constants.

### V-Rep API ###

* simxGetObjectHandle
* simxGetVisionSensorImage
* simxSetJointTargetVelocity
* simxPauseSimulation
* simxPauseCommunication
* simxStartSimulation
* simxStopSimulation
* simxGetObjectOrientation