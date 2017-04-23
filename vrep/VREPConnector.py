from BaseConnector import BaseConnector

import vrep

class VREPConnector(BaseConnector):

	def __init__(self):
		self.clientID = -1

	def create(self, portNumber=19997):
		vrep.simxFinish(-1) # closes all opened connections
		self.clientID = vrep.simxStart('127.0.0.1', portNumber, True, True, 5000, 5)
		#5000 milliseconds connection time out, 5 ms time unit after which data is communicated to vrep
		if self.clientID == -1:
			raise Exception('Could not connect to the vrep client')

	def end(self):
		vrep.simxFinish(self.clientID)
