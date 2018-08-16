"""
    Collection of test functions in Numerical Analysis I/II
    @instructor: Lucius Anderson
                 David Thau
                 Harry Nguyen
"""
from __future__ import division
import numpy as np


""" ================================================================
    Example 1: IVP of ODE y' = y - t^2 + 1
                with initial value y(0) = 0.5
                Exact solution is y(t) = (t+1)^2 - exp(t)/2
                a =0.0 ,b = 2.0, N=20 , ya = 0.5
"""

def exmp1_def_fn(tau, w):
    
    return w - tau**2 + 1.0

def exmp1_sol(t):
   
    return (t+1.0)**2 - np.exp(t)/2.0




""" ================================================================
    Example 2: IVP of ODE y' = 2-2ty/t^2 + 1
                with initial value y(0) = 1
                Exact solution is y(t) = 2t + 1/ t^2 +1 
                t on [0,1]
                h = 0.1
                N = 10
                
    Exercise 5.4 No.4a on page 291 (Burden 9th edtn)
"""

def exmp2_def_fn(tau, w):
    
    return (2.0 - 2.0*tau*w)/(tau**2.0 + 1.0)

def exmp2_sol(t):
    
    return (2.0*t + 1.0)/(t**2.0 + 1.0)


""" =================================================================
    Example 3:  IVP of ODE y' = (1 + y)/(t + (y/t)^2)
                with intial value y(1) = 1
                Exact solution is y(t) = t(tan(ln(t)))
                from t in [1,3]
                h = 0.2
                Exercise 5.4 No. 3b on page 291
"""

def exmp3_def_fn(tau,w):
    
    return (1.0 + (w/tau) + ((w/tau)**2.0))

def exmp3_sol(t):
    
    return t*(np.tan(np.log(t)))

""" =================================================================
    Example 4:  IVP of ODE y' = -(y+1)(y+3)
                with intial value y(0) = -2
                Exact solution is y(t) = -3 + 2(1+e^-2t)^-1
                from t in [0,2]
                h = 0.2
                Exercise 5.4 No. 3c on page 291
"""


def exmp4_def_fn(tau,w):
   
    return (-1.0*((w + 1.0)*(w + 3.0)))

def exmp4_sol(t):
    
    return (-3.0 + 2.0*(1.0 + np.exp(-2.0*t))**-1.0)







