# -*- coding: utf-8 -*-
import sys, pygame, simulation.road, simulation.speedLimits, random, importlib, config
from simulation.car import Car
from representation import Representation
from simulationManager import SimulationManager
from simulation.trafficGenerators import *

if len(sys.argv) != 2:
    print("Usage: python pyTraffic.py module_with_config")
    exit()

#引数で指定されたシナリオをインポートする
config = importlib.import_module(sys.argv[1])

random.seed(config.seed)
pygame.init()
screen = pygame.display.set_mode(config.size)

clock = pygame.time.Clock()

#この辺が定義する必要のある定数-----------
simulation.car.Car.slowDownProbability = config.slowDownProbability
simulation.car.Car.laneChangeProbability = config.laneChangeProbability
speedLimits = simulation.speedLimits.SpeedLimits(config.speedLimits, config.maxSpeed)
road = simulation.road.Road(config.lanes, config.length, speedLimits)
simulation = SimulationManager(road, config.trafficGenerator, config.updateFrame) #updateFrameは何msに更新するかを定義するっぽい
representation = Representation(screen, road, simulation)
#-----------------------------------------

while simulation.running:
    for event in pygame.event.get():
        #キーが押されたら何らかの操作を実行
        if event.type == pygame.KEYDOWN:
            simulation.processKey(event.key)
    clock.tick_busy_loop(config.maxFps) #gameのruntime speedを制御するために定義
    dt = clock.get_time() #前回tickした時からの時間差分を取得
    simulation.update(dt) #時間進行してupdate
    representation.draw(dt * simulation.timeFactor)
    pygame.display.flip() #画面全体を更新

print("Goodbye")
