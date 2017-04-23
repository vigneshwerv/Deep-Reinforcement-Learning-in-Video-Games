class BaseConnector(object):
	def __init__(self, **kwargs):
		return NotImplementedError

	def create(self, **kwargs):
		'''creates a connection with the simulator'''
		return NotImplementedError

	def end(self, **kwargs):
		'''Ends the connection with simulator'''
		return NotImplementedError
