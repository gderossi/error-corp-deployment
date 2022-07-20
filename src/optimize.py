#!/usr/bin/env python3

from scipy.optimize import minimize, NonlinearConstraint, Bounds
import numpy as np


x_lower_bounds = np.array([13, 0])
x_upper_bounds = np.array([100, 100])

def target_func(x):
    return (x[0]-10)**3+(x[1]-20)**3

def constraint_func1(x):
    return -(x[0]-5)**2-(x[1]-5)**2+100

def constraint_func2(x):
    return (x[0]-6)**2+(x[1]-5)**2-82.81

def create_guess():
    x = np.random.random(2) * (x_upper_bounds-x_lower_bounds) + x_lower_bounds
    return x

def optimize1(input):
    bounds = Bounds(x_lower_bounds, x_upper_bounds)
    constraint1 = NonlinearConstraint(constraint_func1, 0, 0)
    guess1 = minimize(target_func, input, bounds=bounds, constraints=constraint1)
    return guess1

def optimize2(input):
    bounds = Bounds(x_lower_bounds, x_upper_bounds)
    constraint1 = NonlinearConstraint(constraint_func1, 0, 0)
    constraint2 = NonlinearConstraint(constraint_func2, 0, 0)
    constraints = [constraint1, constraint2]
    guess2 = minimize(target_func, input, bounds=bounds, constraints=constraints)
    return guess2
