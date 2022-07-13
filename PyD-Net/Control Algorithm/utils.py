import numpy as np
import math
from termcolor import colored

def printScoreGrid(scoreGrid, goalAngleIdx, directionIdx):
    scoreString = ""
    for idx, score in enumerate(scoreGrid):
        if idx == directionIdx == goalAngleIdx:
            scoreString += colored(f" {score: .2f} ", "green")
        elif idx == goalAngleIdx:
            scoreString += colored(f" {score: .2f} ", "red")
        elif idx == directionIdx:
            scoreString += colored(f" {score: .2f} ", "yellow")
        else:
            scoreString += f" {score: .2f} "
    print(scoreString)
   
def getGoalAngle(goalPos, dronePos, actualYaw):
    angle = math.atan2(goalPos[1] - dronePos[1], goalPos[0] - dronePos[0])
    # Left front
    if 0 <= angle <= math.pi / 2:
        if actualYaw >= 0:
            return -angle + actualYaw
        else:
            if actualYaw <= -(math.pi - angle):
                return (2 * math.pi - angle) + actualYaw
            else:
                return -angle + actualYaw
    # Left back
    elif math.pi / 2 <= angle <= math.pi:
        if actualYaw >= 0:
            return -angle + actualYaw
        else:
            if actualYaw >= -(math.pi - angle):
                return -angle + actualYaw
            else:
                return (2 * math.pi - angle) + actualYaw
    # Right front
    elif -math.pi / 2 <= angle <= 0:
        if actualYaw <= 0:
            return -angle + actualYaw
        else:
            if actualYaw >= (math.pi + angle):
                return -(2 * math.pi + angle) + actualYaw
            else:
                return -angle + actualYaw
    # Right back
    elif -math.pi <= angle <= -math.pi / 2:
        if actualYaw <= 0:
            return -angle + actualYaw
        else:
            if actualYaw <= (math.pi + angle):
                return -angle + actualYaw
            else:
                return -(2 * math.pi + angle) + actualYaw
                
def getControlDict(N, yaw_rate, forward_speed):
    controlDict = {0: (0, +yaw_rate)}
    for i in range(1, N-1):
        if i == (N // 2):
            controlDict[i] = (forward_speed, 0)
        elif i < (N // 2):
            controlDict[i] = (forward_speed, yaw_rate/(i+1))
        elif i > (N // 2):
            controlDict[i] = (forward_speed, -yaw_rate/(N-i))
    controlDict[N-1] = (0, -yaw_rate)
    return controlDict

def getAngleIdx(N, fov, goalAngle):
    angles = np.linspace(-fov / 2 + fov / N, fov / 2 - fov / N, N - 1)
    for goalAngleIdx, angle in enumerate(angles):
        if goalAngle <= angle:
            return goalAngleIdx
    return goalAngleIdx + 1

def getDirectionOrder(N, goalAngleIdx):
    orderGrid = np.zeros(N)
    orderGrid[goalAngleIdx] = N - 1
    i = 1
    filled = False
    while not filled:
        filled = True
        if goalAngleIdx + i < N:
           filled = False
           orderGrid[goalAngleIdx + i] = N - i - 1
        if goalAngleIdx - i >= 0:
           filled = False
           orderGrid[goalAngleIdx - i] = N - i - 1
        i += 1
    return orderGrid

def calcDist(coords1, coords2):
    return np.sqrt((coords2[0]-coords1[0])**2 + (coords2[1]-coords1[1])**2)
 
def getDirection(scoreGrid, dronePos, goalPos, actualYaw, N, fov, depthThreshold, goalOverwrite):
    goalAngle = getGoalAngle(goalPos, dronePos, actualYaw)
    goalAngleIdx = getAngleIdx(N, fov, goalAngle)
    thresholdGrid = scoreGrid <= depthThreshold
    if goalOverwrite:
        return goalAngleIdx, goalAngleIdx
    directionOrder = getDirectionOrder(N, goalAngleIdx) 
    exceptionGrid = np.array([True] + [False] * (N-2) + [True])

    if (thresholdGrid == exceptionGrid).all() and goalAngleIdx == N // 2:
    	return np.argmin(scoreGrid), goalAngleIdx
    for directionIdx in np.argsort(directionOrder)[::-1]:
        if thresholdGrid[directionIdx]:
            return directionIdx, goalAngleIdx
    return 0, goalAngleIdx
    
def getObstaclePos(dronePos, goalPos, obstDist):
    distGoalDrone = calcDist(dronePos, goalPos)
    return (dronePos[0] + obstDist/distGoalDrone * (goalPos[0]-dronePos[0]), dronePos[1] + obstDist/distGoalDrone * (goalPos[1]-dronePos[1]))
