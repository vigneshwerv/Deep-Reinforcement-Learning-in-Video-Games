import BaseRobot from BaseRobot
import Util from BaseRobot

import vrep

class QuadCopter(BaseRobot):
	def __init__(self, clientID):
		self.clientID = clientID;
		self.names = {};
		self.names['robot'] = 'Quadricopter';
		self.names['propeller'] = 'Quadricopter_propeller';
		self.__getHandles();

	def __getHandles(self):
		# get object handle
		_, self.robot = vrep.simxGetObjectHandle(self.clientID, self.names['robot'], vrep.simx_opmode_singleshot);
		if _ == -1:
			raise Exception("could not get the robot handle");
		
		self.propellers = [];
		for i in range(0, 4):
			_, self.propellers[i] = vrep.simxGetObjectHandle(self.clientID, self.names['propeller'] + str(i+1), vrep.simx_opmode_singleshot);
			if _ == -1:
				raise Exception("could not get propeller handle");


	def getLocation(self):
		_, location = vrep.simxGetObjectPosition(self.clientID, self.robot, -1, vrep.simx_opmode_oneshot);
		if _ == 0:
			return location;
		else:
			raise Exception("could not retrieve location of the robot");

	def getOrientation(self):
