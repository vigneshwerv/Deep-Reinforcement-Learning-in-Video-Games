import vrep

class Util(object):
    def __init__(self, clientID):
        self.clientID = clientID;

    def getHandle(self, obj, opmode=vrep.simx_opmode_oneshot):
        _, handle = vrep.simxGetObjectHandle(self.clientID, obj, opmode);
        print _, handle;
        if _ < 2:
            return handle;
        else:
            raise Exception('could not retrieve handle for ', obj);

    def getLocation(self, obj, reference_frame=-1, opmode=vrep.simx_opmode_blocking):
        _, location = vrep.simxGetObjectPosition(self.clientID, obj, reference_frame, opmode);
        if _ < 2:
            return location;
        else:
            raise Exception('could not retrieve location for ', obj);
                
    def getOrientation(self, obj, reference_frame=-1, opmode=vrep.simx_opmode_blocking):
        _, orientation = vrep.simxGetObjectOrientation(self.clientID, obj, reference_frame, opmode);
        if _ < 2:
            return orientation;
        else:
            raise Exception('could not retrieve orientation for ', obj);

    def getVelocity(self, obj, reference_frame=-1, opmode=vrep.simx_opmode_blocking):
        _, velocity = vrep.simxGetObjectVelocity(self.clientID, obj, reference_frame, opmode);

        if _ < 2:
            return velocity;
        else:
            raise Exception('could not retrieve velocity for ', obj);
'''
    def setParameter(self, obj, paramName, paramValue):
		vrep
'''
