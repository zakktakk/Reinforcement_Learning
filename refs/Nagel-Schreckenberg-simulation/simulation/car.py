# -*- coding: utf-8 -*-
import random

class Car:
    slowDownProbability = 0
    laneChangeProbability = 0
    def __init__(self, road, pos, velocity = 0):
        self.velocity = velocity #速度, 1updateごとに進むセル数
        self.road = road #自分がいる道路 #TODO check完全観測可能なのか
        self.pos = pos #自分の位置
        self.prevPos = pos #自分の1つ前の位置

    def updateLane(self):
        """
        y軸方面の移動
        レーンの移動
        """
        self.prevPos = self.pos
        if self.willingToChangeUp(): #もし上に動けるかつ行きたいなら
            if random.random() >= Car.laneChangeProbability: #確率的な挙動
                self.pos = self.pos[0], self.pos[1]-1
        elif self.willingToChangeDown():
            if random.random() >= Car.laneChangeProbability:
                self.pos = self.pos[0], self.pos[1]+1
        return self.pos

    def updateX(self):
        """
        x軸方面の移動
        進む
        """
        self.velocity = self.calcNewVelocity() #スピードアップできるならする

        if self.velocity > 0 and random.random() <= Car.slowDownProbability:
            self.velocity -= 1 #確率的に減速

        self.pos = self.pos[0] + self.velocity, self.pos[1] #x軸方向のみ更新
        return self.pos

    def calcNewVelocity(self):
        """
        スピードアップできるなら現在の速度+1を返す
        """
        return min(self.velocity + 1, self.road.getMaxSpeedAt(self.pos))

    def willingToChangeUp(self):
        return self.road.possibleLaneChangeUp(self.pos) and self.__willingToChangeLane(self.pos[1], self.pos[1] - 1)

    def willingToChangeDown(self):
        return self.road.possibleLaneChangeDown(self.pos) and self.__willingToChangeLane(self.pos[1], self.pos[1] + 1)

    def __willingToChangeLane(self, sourceLane, destLane):
        """
        より良い条件のlaneに移れるなら必ず移動
        """
        srcLaneSpeed = self.road.getMaxSpeedAt((self.pos[0], sourceLane)) #現在のlaneのmax speed
        destLaneSpeed = self.road.getMaxSpeedAt((self.pos[0], destLane)) #行き先のlaneのmax speed
        if destLaneSpeed <= srcLaneSpeed: #もし現在のレーンの方が早いなら動かない
            return False
        prevCar = self.road.findPrevCar((self.pos[0], destLane))
        if prevCar == None:
            #先行車がいない
            return True
        else:
            #先行車はいるけど十分離れてるならtrue
            distanceToPrevCar = self.pos[0] - prevCar.pos[0]
            return distanceToPrevCar > prevCar.velocity
