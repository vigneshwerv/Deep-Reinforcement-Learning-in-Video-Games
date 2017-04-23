from BaseRobot import BaseRobot
from util import Util 

import vrep

class QuadCopter(BaseRobot):
	def __init__(self, clientID):
		self.clientID = clientID;
		self.names = {};
		self.names['robot'] = 'Quadricopter';
		self.names['propeller'] = 'Quadricopter_propeller';
		self.util = Util(self.clientID);
		self.__getHandles();

	def __getHandles(self):
		# get object handle
		self.robot = self.util.getHandle(self.names['robot']);
		self.propellers = [];
		for i in range(0, 4):
			self.propellers[i] = self.util.getHandle(self.names['properllers'] + str(i+1));


	def getLocation(self, obj):
		return util.getLocation(obj);

	def getOrientation(self, obj):
		return util.getOrientation(obj);
	
	def getVelocity(self, obj):
		return util.getVelocity(obj);
'''
	def setThrustValues(self, obj, paramName, paramValue):
		util.setParameter(obj, paramName, paramValue);
'''
