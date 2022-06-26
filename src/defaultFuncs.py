import random
import string
import math

charset = list(string.ascii_letters)

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

def randomStrength():
    return random.randint(-100, 100) / 100

def randomName(n=8):
    return ''.join([random.choice(charset) for i in range(n)])