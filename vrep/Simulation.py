import vrep

from VREPConnector import VREPConnector
from QuadCopter import QuadCopter
import math
import time

class Simulation(object):
	def __init__(self):
		self.vc = VREPConnector();
		self.vc.create();
		self.actions = [0.5, 0.6, 0.7, 0.8, 0.9, 1];
		self.goal = [0, 0, 1] #co-ordinates of where we want the quad-copter to be
		self.quadcopter = QuadCopter(self.vc.clientID);
		self.maxThrust = 7;
		self.maxDistance = 2; # used to define whether robot has stirred away too much from the goal
		self.epsilon = 0.0001; # if the quad copter is this much away from each axis then the goal has been reached
		self.dt = 0.05; # 50 ms is the smallest time frame set in vrep
		self.maxOrientation = 15.0;
		self.beta = -10; #how much effect should distance have
		self.maxTime = 2;
		## create a thread and a function


	def __calculateReward(self, position, orientation):
		'''
		returns value -1 if the life should end
		else returns a value based on the distance from goal which lies between [0, 1]
		'''
		if self.__checkEndLife(position, orientation):
			return -1;
		else:
			distance = self.__getDistanceFromGoal(position);
			return math.exp(self.beta*distance);


	def __calculateThrust(self, actions):
		return [action * self.maxThrust for action in actions]

	def __checkEndLife(self, position, orientation):
		# if the robot has drifted away by a set max distance from the goal
		# then we need to assume it as an end life scenario.
		distance = self.__getDistanceFromGoal(position);
		if distance > self.maxDistance:
			return True;

		# check orientation
		''' if any value of alpha beta gamma is greater than 15 degree then it is the end '''
		for i in range(0, len(orientation)):
			if abs(orientation[i]) > self.maxOrientation:
				return True;

		## check time
		if self.count > self.maxTime:
			return True;

		return False;


	def __checkGoalState(self, position, orientation):
		for i in range(0, len(self.goal)):
			if abs(position[i] - self.goal[i]) > self.epsilon:
				return False;
		
		return True;
	
	def __getDistanceFromGoal(self, position):
		return math.sqrt(sum([(self.position[i] - self.goal[i])^2 for i in range(0, len(self.goal))]));

	def __stop(self):
		vrep.simxStopSimulation(self.vc.clientID, vrep.simx_opmode_blocking);

	def __start(self):
		self.startTime = time.time();
		vrep.simxStartSimulation(self.vc.clientID, vrep.simx_opmode_oneshot);

	def __getRobotData(self):
		''' gets the position and orientation of the robot '''
		


	def run(self):
		
		while !self.__checkEndLife(self, position, orientation):
