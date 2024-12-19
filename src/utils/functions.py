import numpy as np

def linear(x, m, c):
    return m * x + c

def gaussian(x, A, m, s):
    return A * np.exp(-((x-m)**2)/(2*(s**2)))

def decay(x, A, t):
    return A * np.exp(-x/t)
