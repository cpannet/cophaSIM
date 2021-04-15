# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:52:21 2020

@author: cpannetier

Contains all the decorator used in the package.
So far, not used but will be useful for:
    - counting time in the simulation.

"""


def NbAppels(f):
    def wrapper(*args, **kwargs):
        wrapper.nbAppels = wrapper.nbAppels + 1
        res = f(*args, **kwargs)
        print(f'Appel {wrapper.nbAppels} de {f.__name__}')
        return res
    wrapper.nbAppels = 0
    return wrapper


def timer(f):
    def wrapper(*args,**dargs):
        import time
        start = time.time()
        res = f(*args , **dargs)
        print(f'Function {f.__name__} processing time: {round(time.time()-start)}s')
        return res
    return wrapper


@NbAppels
def calc(a,b):
    
    return a+b



