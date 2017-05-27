# -*- coding: utf-8 -*-
from simulation.speedLimits import *
from simulation.trafficGenerators import * 

#gameのruntime speedの上限値
maxFps= 40
#game画面の表示サイズ
size = width, heigth = 1280, 500
#500msで一回更新
updateFrame = 500

seed = None

#何レーンか
lanes = 5
#レーンのながさ
length = 200

#車両のmax speed
maxSpeed = 5

#何の長さだ？
maxLength = 10000

speedLimits = [SpeedLimit(range=((0,2),(50,2)), limit=0, ticks=0),
               SpeedLimit(range=((55,0),(145,0)), limit=0, ticks=0),
               SpeedLimit(range=((55,4),(145,4)), limit=0, ticks=0),
               SpeedLimit(range=((150,2),(length,2)), limit=0, ticks=0),
               ]

"""
SimpleTrafficGeneratorの条件

"""
trafficGenerator = SimpleTrafficGenerator()
slowDownProbability, laneChangeProbability = 0.5, 0.5 #減速，レーン変更の可能性
