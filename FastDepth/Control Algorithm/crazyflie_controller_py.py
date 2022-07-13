from controller import Supervisor
from controller import Motor
from controller import InertialUnit
from controller import GPS
from controller import Gyro
from controller import Keyboard
from controller import Camera
from controller import DistanceSensor
from controller import TouchSensor

import cv2
import csv
import numpy as np
import random
import time
import math
from math import cos, sin
from collections import deque
from termcolor import colored
from utils import *
random.seed()

#Depthmap
import torch
from torchvision import transforms
import warnings
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

import sys
sys.path.append('../../../controllers/')
from pid_controller import init_pid_attitude_fixed_height_controller, pid_velocity_fixed_height_controller
from pid_controller import MotorPower_t, ActualState_t, GainsPID_t, DesiredState_t

supervisor = Supervisor()
droneNode = supervisor.getFromDef("Crazyflie")
dronePos = (0, -4)
prevPos = dronePos

timestep = int(supervisor.getBasicTimeStep())

# LEVELS
def writeRun(levelIdx, runIdx, lineIdx, N, distance, time, completed, coords):
    with open(f"results/level{levelIdx}.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([N, runIdx, completed, distance, time, str(coords)])
    with open("levels/line.txt", "w") as f:
        f.write(str(lineIdx + 1))

def getLevelSettings(levelIdx):
    with open("levels/line.txt", "r") as f:
        lineIdx = int(f.readline())
    if int(lineIdx) > 60:
        print("All runs done")
        sys.exit(0)
    with open(f"levels/level{levelIdx}.csv") as f:
        reader = csv.reader(f)
        for i, settings in enumerate(reader):
            if i == int(lineIdx):
                return int(lineIdx), settings

def generateLevel(levelIdx, supervisor):
    gate_node = supervisor.getFromDef("Gate")
    translation_gate = gate_node.getField("translation")
    if levelIdx == 1:
        lineIdx, (runIdx, N, goalX, goalY, obstDist) = getLevelSettings(levelIdx)
        print(f"Level: {levelIdx}, Resolution: {N}, Run: {runIdx}")
        # GOAL
        goalPos = (float(goalX), float(goalY))
        translation_gate.setSFVec3f([goalPos[0], goalPos[1], 0])
        # POLE
        pole_node = supervisor.getFromDef("Pole")
        translation_pole = pole_node.getField("translation")
        obstPos = getObstaclePos(dronePos, goalPos, float(obstDist))
        translation_pole.setSFVec3f([obstPos[0], obstPos[1], 0])
        return lineIdx, int(runIdx), int(N), goalPos
    elif levelIdx == 2:
        lineIdx, (runIdx, N, goalX, goalY, obst1Dist, obst2Dist, obst2Angle) = getLevelSettings(levelIdx)
        print(f"Level: {levelIdx}, Resolution: {N}, Run: {runIdx}")
        # GOAL
        goalPos = (float(goalX), float(goalY))
        translation_gate.setSFVec3f([goalPos[0], goalPos[1], 0])
        # POLE
        pole_node = supervisor.getFromDef("Pole")
        translation_pole = pole_node.getField("translation")
        obst1Pos = getObstaclePos(dronePos, goalPos, float(obst1Dist))
        translation_pole.setSFVec3f([obst1Pos[0], obst1Pos[1], 0])
        # PANEL
        panel_node = supervisor.getFromDef("Panel1")
        translation_panel = panel_node.getField("translation")
        rotation_panel = panel_node.getField("rotation")
        obst2Pos = getObstaclePos(dronePos, goalPos, float(obst2Dist))
        translation_panel.setSFVec3f([obst2Pos[0], obst2Pos[1], 0])
        rotation_panel.setSFRotation([-0.0352, 0, 1, float(obst2Angle)])
        return lineIdx, int(runIdx), int(N), goalPos
    elif levelIdx == 3:
        lineIdx, (runIdx, N, goalX, goalY, obst1Dist, obst2Dist, obst2Angle, obst3Dist, obst3Angle) = getLevelSettings(levelIdx)
        print(f"Level: {levelIdx}, Resolution: {N}, Run: {runIdx}")
        # GOAL
        goalPos = (float(goalX), float(goalY))
        translation_gate.setSFVec3f([goalPos[0], goalPos[1], 0])
        # POLE
        pole_node = supervisor.getFromDef("Pole")
        translation_pole = pole_node.getField("translation")
        obst1Pos = getObstaclePos(dronePos, goalPos, float(obst1Dist))
        translation_pole.setSFVec3f([obst1Pos[0], obst1Pos[1], 0])
        # PANEL
        panel_node = supervisor.getFromDef("Panel1")
        translation_panel = panel_node.getField("translation")
        rotation_panel = panel_node.getField("rotation")
        obst2Pos = getObstaclePos(dronePos, goalPos, float(obst2Dist))
        translation_panel.setSFVec3f([obst2Pos[0], obst2Pos[1], 0])
        rotation_panel.setSFRotation([-0.0352, 0, 1, float(obst2Angle)])
        # WHITE PANEL
        panelT_node = supervisor.getFromDef("PanelT")
        translation_panelT = panelT_node.getField("translation")
        rotation_panelT = panelT_node.getField("rotation")
        obst3Pos = getObstaclePos(dronePos, goalPos, float(obst3Dist))
        translation_panelT.setSFVec3f([obst3Pos[0], obst3Pos[1], 0])
        rotation_panelT.setSFRotation([0, 0, 1, float(obst3Angle)])
        return lineIdx, int(runIdx), int(N), goalPos
        
# START RUN
levelIdx = 3
lineIdx, runIdx, N, goalPos = generateLevel(levelIdx, supervisor)
distance = 0
coords = [dronePos]
supervisor.simulationSetMode(2)

## Initialize motors
m1_motor = supervisor.getDevice("m1_motor");
m1_motor.setPosition(float('inf'))
m1_motor.setVelocity(-1)
m2_motor = supervisor.getDevice("m2_motor");
m2_motor.setPosition(float('inf'))
m2_motor.setVelocity(1)
m3_motor = supervisor.getDevice("m3_motor");
m3_motor.setPosition(float('inf'))
m3_motor.setVelocity(-1)
m4_motor = supervisor.getDevice("m4_motor");
m4_motor.setPosition(float('inf'))
m4_motor.setVelocity(1)

## Initialize Sensors
imu = supervisor.getDevice("inertial unit")
imu.enable(timestep)
gps = supervisor.getDevice("gps")
gps.enable(timestep)
Keyboard().enable(timestep)
gyro = supervisor.getDevice("gyro")
gyro.enable(timestep)
camera = supervisor.getDevice("camera")
camera.enable(timestep)
bumper = supervisor.getDevice("touch sensor")
bumper.enable(timestep)
    
## Initialize variables
actualState = ActualState_t()
desiredState = DesiredState_t()
pastXGlobal = 0
pastYGlobal = -4
past_time = supervisor.getTime()

## Initialize PID gains.
gainsPID = GainsPID_t()
gainsPID.kp_att_y = 1
gainsPID.kd_att_y = 0.5
gainsPID.kp_att_rp =0.5
gainsPID.kd_att_rp = 0.1
gainsPID.kp_vel_xy = 2
gainsPID.kd_vel_xy = 0.5
gainsPID.kp_z = 10
gainsPID.ki_z = 50
gainsPID.kd_z = 5
init_pid_attitude_fixed_height_controller()

## Speeds
forward_speed = 0.2
yaw_rate = 0.5

## Initialize struct for motor power
motorPower = MotorPower_t()

# Modes
goalOverwrite = False

# Camera
w, h = camera.getWidth(), camera.getHeight()
fov = camera.getFov()
imgSize = 320

# Pytorch
ckpt = "ckpt/mobilenet-nnconv5dw-skipadd-pruned.pth.tar"
model = torch.load(ckpt)['model']
model.eval()
trans = transforms.Compose([transforms.ToTensor()])

# Obstacle avoidance strip
stepSize = imgSize // N
stepRemainder = imgSize - stepSize * N
scoreGrid = np.zeros(N)
frameDeque = deque(maxlen=50)
blur = transforms.GaussianBlur(5)
# Control algorithm
depthThreshold = 1.4

controlDict = getControlDict(N, yaw_rate, forward_speed)

# Main loop:
while supervisor.step(timestep) != -1:    
    time = supervisor.getTime()

    dt = time - past_time;
    dronePos = tuple(droneNode.getPosition()[:2])
    distance += calcDist(dronePos, prevPos)
    coords.append(dronePos)
    prevPos = dronePos
    # If crashed
    if bumper.getValue() == 1 or gps.getValues()[2] < 0:
       print("Crashed")
       writeRun(levelIdx, runIdx, lineIdx, N, distance, time, 0, coords)
       supervisor.simulationReset()
       droneNode.restartController()
       
    # If at goal
    diffPos = np.array(goalPos) - np.array(dronePos)
    if ((-0.2 <= diffPos) & (diffPos <= 0.2)).all():
       print("Completed")
       writeRun(levelIdx, runIdx, lineIdx, N, distance, time, 1, coords)
       supervisor.simulationReset()
       droneNode.restartController()
       
    ## Get measurements
    actualState.roll = imu.getRollPitchYaw()[0]
    actualState.pitch = imu.getRollPitchYaw()[1]
    actualState.yaw_rate = gyro.getValues()[2];
    actualState.altitude = gps.getValues()[2];
    xGlobal = gps.getValues()[0]
    vxGlobal = (xGlobal - pastXGlobal)/dt
    yGlobal = gps.getValues()[1]
    vyGlobal = (yGlobal - pastYGlobal)/dt
    
    ## Get body fixed velocities
    actualYaw = imu.getRollPitchYaw()[2];
    cosyaw = cos(actualYaw)
    sinyaw = sin(actualYaw)
    actualState.vx = vxGlobal * cosyaw + vyGlobal * sinyaw
    actualState.vy = - vxGlobal * sinyaw + vyGlobal * cosyaw

    ## Initialize setpoints
    desiredState.roll = 0
    desiredState.pitch = 0
    desiredState.vx = 0
    desiredState.vy = 0
    desiredState.yaw_rate = 0
    desiredState.altitude = 2.0

    forwardDesired = 0
    sidewaysDesired = 0
    yawDesired = 0

    ## Get camera image
    cameraData = camera.getImage()  # Note: uint8 string
    image = np.fromstring(cameraData, np.uint8).reshape(h, w, 4)
    rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Gate detection
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 160, 130])
    upper_yellow = np.array([30, 210, 180])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # If gate fully in sight
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if ((len(approx) == 4) and (cv2.contourArea(c) > 5000)) or ((-2 <= diffPos) & (diffPos <= 2)).all():
            goalOverwrite = True
        else:
            goalOverwrite = False
    
    rgb = cv2.resize(rgb, (imgSize, imgSize)).astype(np.float32) / 255.
    rgb = trans(rgb)[None, ...]
    with torch.no_grad():
        depth = model(rgb.cuda())
    depth = blur.forward(depth)
    depth = np.transpose(torch.squeeze(depth, 0).cpu(), (1, 2, 0))
    
    # ScoreGrid
    for x in range(0, imgSize - stepRemainder, stepSize):
        squareMean = torch.mean(depth[imgSize//2 - (stepSize//2):imgSize//2 + (stepSize//2), x:x+stepSize])
        scoreGrid[x // stepSize] = squareMean
    frameDeque.append(scoreGrid)
    scoreGrid = np.mean(frameDeque, axis=0, dtype=np.float32)
    
    # Controls
    dronePos = (round(xGlobal, 2), round(yGlobal, 2))
    directionIdx, goalAngleIdx = getDirection(scoreGrid, dronePos, goalPos, actualYaw, N, fov, depthThreshold, goalOverwrite)
    # printScoreGrid(scoreGrid, goalAngleIdx, directionIdx)
    forwardDesired, yawDesired = controlDict[directionIdx]

    # Manual override
    key = Keyboard().getKey()
    while key>0:
        if key == Keyboard.UP:
            forwardDesired = forward_speed
        elif key == Keyboard.DOWN:
            forwardDesired = -forward_speed
        elif key == Keyboard.RIGHT:
            sidewaysDesired  = -forward_speed
        elif key == Keyboard.LEFT:
            sidewaysDesired = forward_speed
        elif key == ord('Q'):
            yawDesired =  + yaw_rate
        elif key == ord('E'):
            yawDesired = - yaw_rate
        elif key == ord('G'):
            time.sleep(1)
            goalOverwrite = not goalOverwrite
            
        key = Keyboard().getKey()
    
    desiredState.yaw_rate = yawDesired;

    ## PID velocity controller with fixed height
    desiredState.vy = sidewaysDesired;
    desiredState.vx = forwardDesired;
    pid_velocity_fixed_height_controller(actualState, desiredState, gainsPID, dt, motorPower);

    m1_motor.setVelocity(-motorPower.m1)
    m2_motor.setVelocity(motorPower.m2)
    m3_motor.setVelocity(-motorPower.m3)
    m4_motor.setVelocity(motorPower.m4)
    
    past_time = time
    pastXGlobal = xGlobal
    pastYGlobal = yGlobal
