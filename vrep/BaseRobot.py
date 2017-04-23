class BaseRobot(object):
	def __init__(self, **kwargs):
		return NotImplementedError;
	
	def __getHandles(self, **kwargs):
		return NotImplementedError;

	def getLocation(self, **kwargs):
		return NotImplementedError;

	def getOrientation(self, **kwargs):
		return NotImplementedError;
