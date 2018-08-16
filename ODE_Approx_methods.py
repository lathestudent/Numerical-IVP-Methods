"""
    Several example functions appeared in Numerical Analysis II
    @instructor: Lucius, David Thau, Harry Nguyen, Numerical , Math & Stat, Georgia State University
"""
#import commonly used packages
from __future__ import division 
import numpy as np


# ==============================
# Euler's method
def euler(def_fn, a, b, ya, N):
    
    """
        Test the Euler's method to solve
        initial value problem y'=f(t,y) with t in [a,b] and y(a) = ya.
        Step size h is computed according to input number of mesh points N
    """
    
    f = def_fn  # call the input defining function as f
    
    h = (b-a)/N # step size
    
    t = np.arange(a, b+h, h)    # all mesh points t=(t_0, t_1, ..., t_N)
    y = np.zeros((N+1,))   # Euler approximations at mesh points
    y[0] = ya    # set initial value y(t_0) = ya
    
    # main iterations
    for i in range(0, N):
        
        tau = t[i]  # current mesh point t
        w = y[i]    # current value y(t)
        
        # compute y at next time point t using Euler's method
        y[i+1] = y[i] + h * f(tau, w)
    
    return (t, y)

#=============================================================================
    
#===============================
#Midpoint Method

def Midpoint_meth(def_fn, a, b, ya, N):
    
    """
        Test the Midpoint Method to solve
        initial value problem y'=f(t,y) with t in [a,b] and y(a) = ya.
        Step size h is computed according to input number of mesh points N
    """
    
    f = def_fn #input definining function
    
  
    h = (b-a)/N # developing step size h, from input values divided by N
    
    t = np.arange(a, b+h, h) #array intialized to hold mesh points t
    y = np.zeros((N+1,))     #array to hold Midpoint Method approximated y values
    y [0] = ya
    
    
    #iterative method
    
    for  i in range(0, N):
        
        tau = t[i]      #current mesh point t 
        w = y[i]        #current value y(t)
        
        
        # next iteration using midpoint method 
        y[i + 1] = w + h*f(tau + h/2.0, w + h*f(tau, w /2.0))
        
        
    return (t, y)
    




#=============================================================================
    
#=================================
#Modified Euler's Method
def modi_eulers(def_fn, a, b, ya, N):
    

    """
        Test the Modified Euler's method to solve
        initial value problem y'=f(t,y) with t in [a,b] and y(a) = ya.
        Step size h is computed according to input number of mesh points N
    """
    
    f = def_fn  # call input defining function as f
    h = (b-a)/N #step size
    t = np.arange(a, b+h, h)     # t mesh points creations
    y = np.zeros((N+1,))      # Modified Eulers approx at mesh points
    
    
    y[0] = ya
    
    
    
    
    # y[0] = ya                   # Sets intial value t0 = ya input

    # main iteration
    for i in range(0,N):
        
        tau = t[i]              # current mesh point t
        w = y[i]                # current value y(t)
        
        y[i + 1] = y[i] +   h * (f(tau,w) + f(t[i + 1], w + h * f(tau,w)))/2.0 #iterative step
        
    
    

 
    return (t, y)
#=============================================================================
    
#==================================================================
#Runge Kutta 4th Order Approx Method

def Runge_kutta4(def_fn, a, b, ya, N):
    
    
    """
        Test the Runge Kutta 4th Order Approx method to solve
        initial value problem y'=f(t,y) with t in [a,b] and y(a) = ya.
        Step size h is computed according to input number of mesh points N
    """
    
    #intialization of inputs
    
    f = def_fn          #intakes function to method to approximate 
    
    h = (b-a)/N         #creates step size based on input values of a, b, N 
    
    t = np.arange(a, b+h, h) #array intialized to hold mesh points t
    
    y = np.zeros((N+1,)) #array to hold Midpoint Method approximated y values
    
    y[0] = ya           #intial condition 
    


    # step 2
    
    # iterative method
    for i in range(0,N):
        
        #step 3
        k1 = h * f(t[i],y[i])
        k2 = h * f(t[i] + (h/2.0), y[i] +(k1/2.0))
        k3 = h * f(t[i] + (h/2.0), y[i] + (k2/2.0))
        k4 = h * f(t[i] + h, y[i] + k3)
        
        y[i + 1] = y[i] + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
   
        
        
    return (t,y)
#=============================================================================
    
#Adams-Bashford Four-Step Explict Method
    
def AB_Explicit_Meth(def_fn, a, b, ya, N):
    
    #intialization of inputs
    f = def_fn          #intakes function to method to approximate 
    
    h = (b-a)/N         #creates step size based on input values of a, b, N 
    
    t = np.arange(a, b+h, h) #array intialized to hold mesh points t
    
    y = np.zeros((N+1,)) #array to hold Midpoint Method approximated y values
    
    y[0] = ya           #intial condition 
    
    
    #using RK4 to obtain the first 3 points
    for i in range(0,N):    #establishes iteration for N mesh points
        if i in range(0,3): #for the first 3 iterations in N, runge kutta fourth order holds
            
            k1 = h * f(t[i],y[i])
            k2 = h * f(t[i] + (h/2.0), y[i] +(k1/2.0))
            k3 = h * f(t[i] + (h/2.0), y[i] + (k2/2.0))
            k4 = h * f(t[i] + h, y[i] + k3)
        
            y[i + 1] = y[i] + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        else:       #else, N>3, explicit AB method
        
            y[i + 1] = y[i] + h*(55.0 * f(t[i],y[i]) - 59.0 * f(t[i-1],y[i-1]) + 37.0 * f(t[i-2],y[i-2]) - 9.0 * f(t[i-3],y[i-3]))/24.0
    
    return(t,y)
    
#=============================================================================


#==============================================================================

#Predictor Corrector Method
    
#============================================================

def AB_Predictor_Corrector(def_fn, a, b, ya, N):
    
    f = def_fn          #intakes function to method to approximate 
    
    h = (b-a)/N         #creates step size based on input values of a, b, N 
    
    t = np.arange(a, b+h, h) #array intialized to hold mesh points t
    
    y = np.zeros((N+1,)) #array to hold Midpoint Method approximated y values
    
    y[0] = ya           #intial condition 

    
    #using RK4 to obtain the first 3 points
    for i in range(0,N):
        if i in range(0,3):
            k1 = h * f(t[i],y[i])
            k2 = h * f(t[i] + (h/2.0), y[i] +(k1/2.0))
            k3 = h * f(t[i] + (h/2.0), y[i] + (k2/2.0))
            k4 = h * f(t[i] + h, y[i] + k3)
        
            y[i + 1] = y[i] + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
            
        else:
            
             y[i + 1] = y[i] + h*(55.0 * f(t[i],y[i]) - 59.0 * f(t[i-1],y[i-1]) + 37.0 * f(t[i-2],y[i-2]) - 9.0 * f(t[i-3],y[i-3]))/24.0
             
             
             y[i + 1] = y[i] + h*(9.0 * f(t[i+1], y[i + 1]) + 19.0 * f(t[i],y[i]) - 5.0 * f(t[i-1],y[i-1]) + f(t[i-2],y[i-2]))/24.0
             
             
    return(t,y)