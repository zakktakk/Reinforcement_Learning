# -*- coding: utf-8 -*-
import random

class SimpleTrafficGenerator():
    def __init__(self, carPerUpdate=1):
        self.queue = 0
        self.carPerUpdate = carPerUpdate
    
    def generate(self, road):
        """
        amountが1ならどこかのレーンに新しい車が追加される
        """
        amount = random.randint(0, self.carPerUpdate) #0 or 1の整数
        self.tryGenerate(road, amount)

    def tryGenerate(self, road, amount):
        added = road.pushCarsRandomly(amount + self.queue)
        self.queue += (amount - added) #追加すべき車がjamかなんかで追加できなかった場合はqueueに追加しとく
