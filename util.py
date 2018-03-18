# encoding=utf-8

import random
import tensorflow as tf
import numpy as np


def randomdata(data):
    print "随机混淆数据"
    for i in xrange(len(data) - 1):
        randindex = random.randint(i + 1, len(data) - 1)
        data[i], data[randindex] = data[randindex], data[i]