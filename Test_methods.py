"""
    Test Methods Page
    
    David Thau
    Lucius Anderson
    Harry Nguyen
    Numerical Analysis II
    
    This tester page tests the various approximation methods on different
    examples and shows their individual plots
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


import ODE_Approx_methods
import exmp_fn


"""
    Euler's Method Function Approximation Tests/Plotting
"""
##############################################################################
##############################################################################

# Eulers Method for example #1

N = 20  # number of mesh points
a = 0.0 # left end point of interval [a,b]
b = 2.0 # right end point of interval [a,b]
ya = 0.5 # initial value y(a)


# defining function and true solution of function #1 
def_fn = exmp_fn.exmp1_def_fn
sol = exmp_fn.exmp1_sol

# eulers approx for example 1 function
(t,y) = ODE_Approx_methods.euler(def_fn, a, b, ya, N)

# compute exact for example #1 solution for comparison
z = sol(t) 

#Plotting example #1 

print('Errors at time mesh points, Example #1:')
print(np.abs(np.transpose(np.ravel(y)-z)))


# plot comparison of exact solution z(t) and approximation y(t), example 1
plt.figure(1)

plt.rcParams.update({'font.size': 20})	# set plot font size
plt.plot(t, z, 'b-', marker='o', linewidth=2)	# plot true solution z
plt.plot(t, y, 'r--', marker='*', linewidth=2)	# plot approximation by Euler's method



plt.xlabel('t')	# set x-axis label as t
plt.ylabel('y(t)')	# set y-axis label as y(t)
plt.legend(['Exact solution', 'Euler approximation, Example #1 '], loc='lower right')	# set legend and location

plt.show()

#print('Exact solutions of example #1:',z)   
      
         

##############################################################################
##############################################################################


############ Example #2 3d 291 Given Points
N2 = 10     
a2 = 0.0
b2 = 1.0
ya2 = 1.0

# defining function and true solution of function #2
def_fn2 = exmp_fn.exmp2_def_fn
sol2 = exmp_fn.exmp2_sol


# run Euler's method from ODE_Approx_methods for example #2

(t2,w2) = ODE_Approx_methods.euler(def_fn2, a2, b2, ya2, N2)


# Exact solutions for comparison of example #2 

z2 = sol2(t2)

# plot comparison of exact solution w(t) and approximation y(t), example 2'
plt.figure(2)

print('Errors at time mesh points, Example #2: ')
print(np.abs(np.transpose(np.ravel(w2)-z2)))

plt.rcParams.update({'font.size': 20})
plt.plot(t2,z2, 'b-' , marker= 'o', linewidth=2)
plt.plot(t2,w2, 'r-', marker = '*', linewidth=2)
#
plt.xlabel('t2')
plt.ylabel('w(t2)')
plt.legend([' Exact Solution', 'Euler Approximation, Example #2'], loc = 'lower right' )
#        
plt.show()

#==============================================================================
########### Example #3; 3b 291  Given Points 

N3 = 20
a3 = 1.0
b3 = 3.0
ya3 = 0.0

# defining function and true solution of function #3
def_fn3 = exmp_fn.exmp3_def_fn
sol3 = exmp_fn.exmp3_sol


# run Euler's method from ODE_Approx_methods for example #3

(t3,w3) = ODE_Approx_methods.euler(def_fn3, a3, b3, ya3, N3)

# Exact solutions for comparison of example #3

z3 = sol3(t3)


# plot comparison of exact solution w(t) and approximation y(t), example 3'
plt.figure(3)

print('Errors at time mesh points, Example #3: ')
print(np.abs(np.transpose(np.ravel(w3)-z3)))

plt.rcParams.update({'font.size': 20})
plt.plot(t2,z2, 'b-' , marker= 'o', linewidth=2)
plt.plot(t2,w2, 'm', marker = '*', linewidth=2)
#
plt.xlabel('t3')
plt.ylabel('w(t3)')
plt.legend([' Exact Solution', 'Euler Approximation, Example #3'], loc = 'lower right' )
#        
plt.show()
#==============================================================================

############ Example #4 3c 291 Given Points 

N4 = 10
a4 = 0.0
b4 = 2.0
ya4 = -2.0

# defining function and true solution of function #4
def_fn4 = exmp_fn.exmp4_def_fn
sol4 = exmp_fn.exmp4_sol


# run Euler's method from ODE_Approx_methods for example #4

(t4, w4) = ODE_Approx_methods.euler(def_fn4, a4, b4, ya4, N4)

# Exact solutions for comparison of example #3

z4 = sol4(t4)


# plot comparison of exact solution w(t) and approximation y(t), example 4'
plt.figure(4)

print('Errors at time mesh points, Example #4: ')
print(np.abs(np.transpose(np.ravel(w4)-z4)))

plt.rcParams.update({'font.size': 20})
plt.plot(t4,z4, 'b-' , marker= 'o', linewidth=2)
plt.plot(t4,w4, 'c-', marker = '*', linewidth=2)
#
plt.xlabel('t4')
plt.ylabel('w(t4)')
plt.legend([' Exact Solution', 'Euler Approximation, Example #4'], loc = 'lower right' )
#        
plt.show()
############################################################











#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


"""
    Modified Eulers Testing
    The following section uses the midpoint method to approx ODE
"""


#==============================================================================

############ Example #1 Given Points 

N_mdeul1 = 20.0  # number of mesh points
a_mdeul1 = 0.0 # left end point of interval [a,b]
b_mdeul1 = 2.0 # right end point of interval [a,b]
ya_mdeul1 = 0.5 # initial value y(a)

# defining function and true solution of function #1
def_fn_mdeul1 = exmp_fn.exmp1_def_fn
sol_mdeul1 = exmp_fn.exmp1_sol


# run Euler's method from ODE_Approx_methods for example #1

(t_mdeul1,w_mdeul1) = ODE_Approx_methods.modi_eulers(def_fn_mdeul1, a_mdeul1, b_mdeul1, ya_mdeul1, int(N_mdeul1))

# Exact solutions for comparison of example #3

z_mdeul1 = sol(t_mdeul1)


# plot comparison of exact solution w(t) and approximation y(t), example 3'
plt.figure(5)

print('Errors at time mesh points, Example #1: ')
print(np.abs(np.transpose(np.ravel(w_mdeul1)-z_mdeul1)))


plt.rcParams.update({'font.size': 20})
plt.plot(t_mdeul1,z_mdeul1, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_mdeul1, w_mdeul1, 'c-', marker = '*', linewidth=2)
#
plt.xlabel('t')
plt.ylabel('w(t)')
plt.legend([' Exact Solution', 'Modified Eulers Approximation, Example #1'], loc = 'lower right' )
#        
plt.show()

#==============================================================================

############ Example #2 3d 291 Given Points
N_mdeul2 = 10     
a_mdeul2 = 0.0
b_mdeul2 = 1.0
ya_mdeul2 = 1.0

# defining function and true solution of function #2
def_fn_mdeul2 = exmp_fn.exmp2_def_fn
sol_mdeul2 = exmp_fn.exmp2_sol


# run Euler's method from ODE_Approx_methods for example #2

(t_mdeul2,w_mdeul2) = ODE_Approx_methods.modi_eulers(def_fn_mdeul2, a_mdeul2, b_mdeul2, ya_mdeul2, N_mdeul2)


# Exact solutions for comparison of example #2 

z_mdeul2 = sol_mdeul2(t_mdeul2)

# plot comparison of exact solution w(t) and approximation y(t), example 2'
plt.figure(6)

print('Errors at time mesh points, Example #2: ')
print(np.abs(np.transpose(np.ravel(w_mdeul2)-z_mdeul2)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_mdeul2,z_mdeul2, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_mdeul2,w_mdeul2, 'r-', marker = '*', linewidth=2)
#
plt.xlabel('t2')
plt.ylabel('w(t2)')
plt.legend([' Exact Solution', 'Modified Eulers Approximation, Example #2'], loc = 'lower right' )
#        
plt.show()
#==============================================================================

#==============================================================================
########### Example #3; 3b 291  Given Points 

N_mdeul3 = 20
a_mdeul3 = 1.0
b_mdeul3 = 3.0
ya_mdeul3 = 0.0

# defining function and true solution of function #3
def_fn_mdeul3 = exmp_fn.exmp3_def_fn
sol_mdeul3 = exmp_fn.exmp3_sol


# run Euler's method from ODE_Approx_methods for example #3

(t_mdeul3,w_mdeul3) = ODE_Approx_methods.modi_eulers(def_fn_mdeul3, a_mdeul3, b_mdeul3, ya_mdeul3, N_mdeul3)

# Exact solutions for comparison of example #3

z_mdeul3 = sol_mdeul3(t_mdeul3)


# plot comparison of exact solution w(t) and approximation y(t), example 3'
plt.figure(7)

print('Errors at time mesh points, Example #3: ')
print(np.abs(np.transpose(np.ravel(w_mdeul3)-z_mdeul3)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_mdeul3,z_mdeul3, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_mdeul3,w_mdeul3, 'm', marker = '*', linewidth=2)
#
plt.xlabel('t3')
plt.ylabel('w(t3)')
plt.legend([' Exact Solution', 'Modified Euler Approximation, Example #3'], loc = 'lower right' )
#        
plt.show()
#==============================================================================

############ Example #4 3c 291 Given Points 

N_mdeul4 = 10
a_mdeul4 = 0.0
b_mdeul4 = 2.0
ya_mdeul4 = -2.0

# defining function and true solution of function #4
def_fn_mdeul4 = exmp_fn.exmp4_def_fn
sol_mdeul4 = exmp_fn.exmp4_sol


# run Euler's method from ODE_Approx_methods for example #4

(t_mdeul4, w_mdeul4) = ODE_Approx_methods.modi_eulers(def_fn_mdeul4, a_mdeul4, b_mdeul4, ya_mdeul4, N_mdeul4)

# Exact solutions for comparison of example #3

z_mdeul4 = sol_mdeul4(t_mdeul4)


# plot comparison of exact solution w(t) and approximation y(t), example 4'
plt.figure(8)

print('Errors at time mesh points, Example #4: ')
print(np.abs(np.transpose(np.ravel(w_mdeul4)-z_mdeul4)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_mdeul4,z_mdeul4, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_mdeul4,w_mdeul4, 'c-', marker = '*', linewidth=2)
#
plt.xlabel('t4')
plt.ylabel('w(t4)')
plt.legend([' Exact Solution', 'Modified Euler Approximation, Example #4'], loc = 'lower right' )
#        
plt.show()
############################################################

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


"""
    Midpoint Method Function Approximation Tests/Plotting
"""
##############################################################################
##############################################################################

# Midpoint Method for example #1

N_mdpt1 = 20  # number of mesh points
a_mdpt1 = 0.0 # left end point of interval [a,b]
b_mdpt1 = 2.0 # right end point of interval [a,b]
ya_mdpt1 = 0.5 # initial value y(a)


# defining function and true solution of function #1 
def_fn_mdpt1 = exmp_fn.exmp1_def_fn
sol_mdpt1 = exmp_fn.exmp1_sol

# eulers approx for example 1 function
(t_mdpt1,y_mdpt1) = ODE_Approx_methods.Midpoint_meth(def_fn, a, b, ya, N)

# compute exact for example #1 solution for comparison
z_mdpt1 = sol(t_mdpt1) 

#Plotting example #1 

print('Errors at time mesh points, Example #1:')
print(np.abs(np.transpose(np.ravel(y_mdpt1)-z_mdpt1)))


# plot comparison of exact solution z(t) and approximation y(t), example 1
plt.figure(9)

plt.rcParams.update({'font.size': 20})	# set plot font size
plt.plot(t_mdpt1, z_mdpt1, 'b-', marker='o', linewidth=2)	# plot true solution z
plt.plot(t_mdpt1, y_mdpt1, 'r--', marker='*', linewidth=2)	# plot approximation by Euler's method



plt.xlabel('t')	# set x-axis label as t
plt.ylabel('y(t)')	# set y-axis label as y(t)
plt.legend(['Exact solution', 'Midpoint Method Approximation, Example #1 '], loc='lower right')	# set legend and location

plt.show()

#print('Exact solutions of example #1:',z)   
      
         

##############################################################################
##############################################################################


############ Example #2 3d 291 Given Points
N_mdpt2 = 10     
a_mdpt2 = 0.0
b_mdpt2 = 1.0
ya_mdpt2 = 1.0

# defining function and true solution of function #2
def_fn_mdpt2 = exmp_fn.exmp2_def_fn
sol_mdpt2 = exmp_fn.exmp2_sol


# run Euler's method from ODE_Approx_methods for example #2

(t_mdpt2,w_mdpt2) = ODE_Approx_methods.Midpoint_meth(def_fn_mdpt2, a_mdpt2, b_mdpt2, ya_mdpt2, N_mdpt2)


# Exact solutions for comparison of example #2 

z_mdpt2 = sol_mdpt2(t_mdpt2)

# plot comparison of exact solution w(t) and approximation y(t), example 2'
plt.figure(10)

print('Errors at time mesh points, Example #2: ')
print(np.abs(np.transpose(np.ravel(w_mdpt2)-z_mdpt2)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_mdpt2,z_mdpt2, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_mdpt2,w_mdpt2, 'r-', marker = '*', linewidth=2)
#
plt.xlabel('t2')
plt.ylabel('w(t2)')
plt.legend([' Exact Solution', 'Midpoint Approx, Example #2'], loc = 'lower right' )
#        
plt.show()

#==============================================================================
########### Example #3; 3b 291  Given Points 

N_mdpt3 = 20
a_mdpt3 = 1.0
b_mdpt3 = 3.0
ya_mdpt3 = 0.0

# defining function and true solution of function #3
def_fn_mdpt3 = exmp_fn.exmp3_def_fn
sol_mdpt3 = exmp_fn.exmp3_sol


# run Euler's method from ODE_Approx_methods for example #3

(t_mdpt3,w_mdpt3) = ODE_Approx_methods.Midpoint_meth(def_fn_mdpt3, a_mdpt3, b_mdpt3, ya_mdpt3, N_mdpt3)

# Exact solutions for comparison of example #3

z_mdpt3 = sol_mdpt3(t_mdpt3)


# plot comparison of exact solution w(t) and approximation y(t), example 3'
plt.figure(11)

print('Errors at time mesh points, Example #3: ')
print(np.abs(np.transpose(np.ravel(w_mdpt3)-z_mdpt3)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_mdpt3,z_mdpt3, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_mdpt3,w_mdpt3, 'm', marker = '*', linewidth=2)
#
plt.xlabel('t3')
plt.ylabel('w(t3)')
plt.legend([' Exact Solution', 'Midpoint Method Approx, Example #3'], loc = 'lower right' )
#        
plt.show()
#==============================================================================

############ Example #4 3c 291 Given Points 

N_mdpt4 = 10
a_mdpt4 = 0.0
b_mdpt4 = 2.0
ya_mdpt4 = -2.0

# defining function and true solution of function #4
def_fn_mdpt4 = exmp_fn.exmp4_def_fn
sol_mdpt4 = exmp_fn.exmp4_sol


# run Euler's method from ODE_Approx_methods for example #4

(t_mdpt4, w_mdpt4) = ODE_Approx_methods.Midpoint_meth(def_fn_mdpt4, a_mdpt4, b_mdpt4, ya_mdpt4, N_mdpt4)

# Exact solutions for comparison of example #3

z_mdpt4 = sol_mdpt4(t_mdpt4)


# plot comparison of exact solution w(t) and approximation y(t), example 4'
plt.figure(12)

print('Errors at time mesh points, Example #4: ')
print(np.abs(np.transpose(np.ravel(w_mdpt4)-z_mdpt4)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_mdpt4,z_mdpt4, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_mdpt4,w_mdpt4, 'c-', marker = '*', linewidth=2)
#
plt.xlabel('t4')
plt.ylabel('w(t4)')
plt.legend([' Exact Solution', 'Midpoint Approximation, Example #4'], loc = 'lower right' )
#        
plt.show()
############################################################


#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


"""
    Runge Kutta 4th Order Approximation Tests/Plotting
"""
##############################################################################
##############################################################################

# RK4 for example #1

N_rk1 = 20  # number of mesh points
a_rk1 = 0.0 # left end point of interval [a,b]
b_rk1 = 2.0 # right end point of interval [a,b]
ya_rk1 = 0.5 # initial value y(a)


# defining function and true solution of function #1 
def_fn_rk1 = exmp_fn.exmp1_def_fn
sol_rk1 = exmp_fn.exmp1_sol

# eulers approx for example 1 function
(t_rk1,y_rk1) = ODE_Approx_methods.Runge_kutta4(def_fn_rk1, a_rk1, b_rk1, ya_rk1, N_rk1)

# compute exact for example #1 solution for comparison
z_rk1 = sol(t_rk1) 

#Plotting example #1 

print('Errors at time mesh points, Example #1:')
print(np.abs(np.transpose(np.ravel(y_rk1)-z_rk1)))


# plot comparison of exact solution z(t) and approximation y(t), example 1
plt.figure(13)

plt.rcParams.update({'font.size': 20})	# set plot font size
plt.plot(t_rk1, z_rk1, 'b-', marker='o', linewidth=2)	# plot true solution z
plt.plot(t_rk1, y_rk1, 'r--', marker='*', linewidth=2)	# plot approximation by Euler's method



plt.xlabel('t')	# set x-axis label as t
plt.ylabel('y(t)')	# set y-axis label as y(t)
plt.legend(['Exact solution', 'RK4 Approx, Example #1 '], loc='lower right')	# set legend and location

plt.show()

#print('Exact solutions of example #1:',z)   
      
         

##############################################################################
##############################################################################


############ Example #2 3d 291 Given Points
N_rk2 = 10     
a_rk2 = 0.0
b_rk2 = 1.0
ya_rk2 = 1.0

# defining function and true solution of function #2
def_fn_rk2 = exmp_fn.exmp2_def_fn
sol_rk2 = exmp_fn.exmp2_sol


# run Euler's method from ODE_Approx_methods for example #2

(t_rk2,w_rk2) = ODE_Approx_methods.Runge_kutta4(def_fn_rk2, a_rk2, b_rk2, ya_rk2, N_rk2)


# Exact solutions for comparison of example #2 

z_rk2 = sol_rk2(t_rk2)

# plot comparison of exact solution w(t) and approximation y(t), example 2'
plt.figure(14)

print('Errors at time mesh points, Example #2: ')
print(np.abs(np.transpose(np.ravel(w_rk2)-z_rk2)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_rk2,z_rk2, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_rk2,w_rk2, 'r-', marker = '*', linewidth=2)
#
plt.xlabel('t2')
plt.ylabel('w(t2)')
plt.legend([' Exact Solution', 'RK4, Example #2'], loc = 'lower right' )
#        
plt.show()

#==============================================================================
########### Example #3; 3b 291  Given Points 

N_rk3 = 20
a_rk3 = 1.0
b_rk3 = 3.0
ya_rk3 = 0.0

# defining function and true solution of function #3
def_fn_rk3 = exmp_fn.exmp3_def_fn
sol_rk3 = exmp_fn.exmp3_sol


# run Euler's method from ODE_Approx_methods for example #3

(t_rk3,w_rk3) = ODE_Approx_methods.Runge_kutta4(def_fn_rk3, a_rk3, b_rk3, ya_rk3, N_rk3)

# Exact solutions for comparison of example #3

z_rk3 = sol_rk3(t_rk3)


# plot comparison of exact solution w(t) and approximation y(t), example 3'
plt.figure(15)

print('Errors at time mesh points, Example #3: ')
print(np.abs(np.transpose(np.ravel(w_rk3)-z_rk3)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_rk3,z_rk3, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_rk3,w_rk3, 'm', marker = '*', linewidth=2)
#
plt.xlabel('t3')
plt.ylabel('w(t3)')
plt.legend([' Exact Solution', 'RK4 Approximation, Example #3'], loc = 'lower right' )
#        
plt.show()
#==============================================================================

############ Example #4 3c 291 Given Points 

N_rk4 = 10
a_rk4 = 0.0
b_rk4 = 2.0
ya_rk4 = -2.0

# defining function and true solution of function #4
def_fn_rk4 = exmp_fn.exmp4_def_fn
sol_rk4 = exmp_fn.exmp4_sol


# run Euler's method from ODE_Approx_methods for example #4

(t_rk4, w_rk4) = ODE_Approx_methods.Runge_kutta4(def_fn_rk4, a_rk4, b_rk4, ya_rk4, N_rk4)

# Exact solutions for comparison of example #3

z_rk4 = sol_rk4(t_rk4)


# plot comparison of exact solution w(t) and approximation y(t), example 4'
plt.figure(16)

print('Errors at time mesh points, Example #4: ')
print(np.abs(np.transpose(np.ravel(w_rk4)-z_rk4)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_rk4,z_rk4, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_rk4,w_rk4, 'c-', marker = '*', linewidth=2)
#
plt.xlabel('t4')
plt.ylabel('w(t4)')
plt.legend([' Exact Solution', 'RK4 Approx, Example #4'], loc = 'lower right' )
#        
plt.show()
############################################################


#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================




"""
    Adams-Bashforth Explicit Approximation Tests/Plotting
"""
##############################################################################
##############################################################################

# RK4 for example #1

N_ab1 = 20  # number of mesh points
a_ab1 = 0.0 # left end point of interval [a,b]
b_ab1 = 2.0 # right end point of interval [a,b]
ya_ab1 = 0.5 # initial value y(a)


# defining function and true solution of function #1 
def_fn_ab1 = exmp_fn.exmp1_def_fn
sol_ab1 = exmp_fn.exmp1_sol

# eulers approx for example 1 function
(t_ab1,y_ab1) = ODE_Approx_methods.AB_Explicit_Meth(def_fn_ab1, a_ab1, b_ab1, ya_ab1, N_ab1)

# compute exact for example #1 solution for comparison
z_ab1 = sol(t_ab1) 

#Plotting example #1 

print('Errors at time mesh points, Example #1:')
print(np.abs(np.transpose(np.ravel(y_ab1)-z_ab1)))


# plot comparison of exact solution z(t) and approximation y(t), example 1
plt.figure(15)

plt.rcParams.update({'font.size': 20})	# set plot font size
plt.plot(t_ab1, z_ab1, 'b-', marker='o', linewidth=2)	# plot true solution z
plt.plot(t_ab1, y_ab1, 'r--', marker='*', linewidth=2)	# plot approximation by Euler's method



plt.xlabel('t')	# set x-axis label as t
plt.ylabel('y(t)')	# set y-axis label as y(t)
plt.legend(['Exact solution', 'AB Explicit Ap, Example #1 '], loc='lower right')	# set legend and location

plt.show()

#print('Exact solutions of example #1:',z)   
      
         

##############################################################################
##############################################################################


############ Example #2 3d 291 Given Points
N_ab2 = 10     
a_ab2 = 0.0
b_ab2 = 1.0
ya_ab2 = 1.0

# defining function and true solution of function #2
def_fn_ab2 = exmp_fn.exmp2_def_fn
sol_ab2 = exmp_fn.exmp2_sol


# run Euler's method from ODE_Approx_methods for example #2

(t_ab2,w_ab2) = ODE_Approx_methods.AB_Explicit_Meth(def_fn_ab2, a_ab2, b_ab2, ya_ab2, N_ab2)


# Exact solutions for comparison of example #2 

z_ab2 = sol_ab2(t_ab2)

# plot comparison of exact solution w(t) and approximation y(t), example 2'
plt.figure(16)

print('Errors at time mesh points, Example #2: ')
print(np.abs(np.transpose(np.ravel(w_ab2)-z_ab2)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_ab2,z_ab2, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_ab2,w_ab2, 'r-', marker = '*', linewidth=2)
#
plt.xlabel('t2')
plt.ylabel('w(t2)')
plt.legend([' Exact Solution', 'AB Explicit Approx, Example #2'], loc = 'lower right' )
#        
plt.show()

#==============================================================================
########### Example #3; 3b 291  Given Points 

N_ab3 = 20
a_ab3 = 1.0
b_ab3 = 3.0
ya_ab3 = 0.0

# defining function and true solution of function #3
def_fn_ab3 = exmp_fn.exmp3_def_fn
sol_ab3 = exmp_fn.exmp3_sol


# run ab method from ODE_Approx_methods for example #3

(t_ab3,w_ab3) = ODE_Approx_methods.AB_Explicit_Meth(def_fn_ab3, a_ab3, b_ab3, ya_ab3, N_ab3)

# Exact solutions for comparison of example #3

z_ab3 = sol_rk3(t_ab3)


# plot comparison of exact solution w(t) and approximation y(t), example 3'
plt.figure(17)

print('Errors at time mesh points, Example #3: ')
print(np.abs(np.transpose(np.ravel(w_ab3)-z_ab3)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_ab3,z_ab3, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_ab3,w_ab3, 'm', marker = '*', linewidth=2)
#
plt.xlabel('t3')
plt.ylabel('w(t3)')
plt.legend([' Exact Solution', 'AB Explicit Approx, Example #3'], loc = 'lower right' )
#        
plt.show()
#==============================================================================

############ Example #4 3c 291 Given Points 

N_ab4 = 10
a_ab4 = 0.0
b_ab4 = 2.0
ya_ab4 = -2.0

# defining function and true solution of function #4
def_fn_ab4 = exmp_fn.exmp4_def_fn
sol_ab4 = exmp_fn.exmp4_sol


# run Euler's method from ODE_Approx_methods for example #4

(t_ab4, w_ab4) = ODE_Approx_methods.AB_Explicit_Meth(def_fn_ab4, a_ab4, b_ab4, ya_ab4, N_ab4)

# Exact solutions for comparison of example #3

z_ab4 = sol_ab4(t_ab4)


# plot comparison of exact solution w(t) and approximation y(t), example 4'
plt.figure(18)

print('Errors at time mesh points, Example #4: ')
print(np.abs(np.transpose(np.ravel(w_ab4)-z_ab4)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_ab4,z_ab4, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_ab4,w_ab4, 'c-', marker = '*', linewidth=2)
#
plt.xlabel('t4')
plt.ylabel('w(t4)')
plt.legend([' Exact Solution', 'AB Explicit, Example #4'], loc = 'lower right' )
#        
plt.show()
############################################################


#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================




"""
    Adams-Bashforth Predictor Approximation Tests/Plotting
"""
##############################################################################
##############################################################################

# AB Predictor for example #1

N_abpc1 = 20  # number of mesh points
a_abpc1 = 0.0 # left end point of interval [a,b]
b_abpc1 = 2.0 # right end point of interval [a,b]
ya_abpc1 = 0.5 # initial value y(a)


# defining function and true solution of function #1 
def_fn_abpc1 = exmp_fn.exmp1_def_fn
sol_abpc1 = exmp_fn.exmp1_sol

# eulers approx for example 1 function
(t_abpc1,y_abpc1) = ODE_Approx_methods.AB_Predictor_Corrector(def_fn_abpc1, a_abpc1, b_abpc1, ya_abpc1, N_abpc1)

# compute exact for example #1 solution for comparison
z_abpc1 = sol(t_abpc1) 

#Plotting example #1 

print('Errors at time mesh points, Example #1:')
print(np.abs(np.transpose(np.ravel(y_abpc1)-z_abpc1)))


# plot comparison of exact solution z(t) and approximation y(t), example 1
plt.figure(19)

plt.rcParams.update({'font.size': 20})	# set plot font size
plt.plot(t_abpc1, z_abpc1, 'b-', marker='o', linewidth=2)	# plot true solution z
plt.plot(t_abpc1, y_abpc1, 'r--', marker='*', linewidth=2)	# plot approximation by Euler's method



plt.xlabel('t')	# set x-axis label as t
plt.ylabel('y(t)')	# set y-axis label as y(t)
plt.legend(['Exact solution', 'AB Predictor Corrector, Example #1 '], loc='lower right')	# set legend and location

plt.show()

#print('Exact solutions of example #1:',z)   
      
         

##############################################################################
##############################################################################


############ Example #2 3d 291 Given Points
N_abpc2 = 10     
a_abpc2 = 0.0
b_abpc2 = 1.0
ya_abpc2 = 1.0

# defining function and true solution of function #2
def_fn_abpc2 = exmp_fn.exmp2_def_fn
sol_abpc2 = exmp_fn.exmp2_sol


# run Euler's method from ODE_Approx_methods for example #2

(t_abpc2,w_abpc2) = ODE_Approx_methods.AB_Predictor_Corrector(def_fn_abpc2, a_abpc2, b_abpc2, ya_abpc2, N_abpc2)


# Exact solutions for comparison of example #2 

z_abpc2 = sol_abpc2(t_abpc2)

# plot comparison of exact solution w(t) and approximation y(t), example 2'
plt.figure(20)

print('Errors at time mesh points, Example #2: ')
print(np.abs(np.transpose(np.ravel(w_abpc2)-z_abpc2)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_abpc2,z_abpc2, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_abpc2,w_abpc2, 'r-', marker = '*', linewidth=2)
#
plt.xlabel('t2')
plt.ylabel('w(t2)')
plt.legend([' Exact Solution', 'AB Predictor Corrector, Example #2'], loc = 'lower right' )
#        
plt.show()

#==============================================================================
########### Example #3; 3b 291  Given Points 

N_abpc3 = 20
a_abpc3 = 1.0
b_abpc3 = 3.0
ya_abpc3 = 0.0

# defining function and true solution of function #3
def_fn_abpc3 = exmp_fn.exmp3_def_fn
sol_abpc3 = exmp_fn.exmp3_sol


# run ab method from ODE_Approx_methods for example #3

(t_abpc3,w_abpc3) = ODE_Approx_methods.AB_Predictor_Corrector(def_fn_abpc3, a_abpc3, b_abpc3, ya_abpc3, N_abpc3)

# Exact solutions for comparison of example #3

z_abpc3 = sol_abpc3(t_abpc3)


# plot comparison of exact solution w(t) and approximation y(t), example 3'
plt.figure(21)

print('Errors at time mesh points, Example #3: ')
print(np.abs(np.transpose(np.ravel(w_abpc3)-z_abpc3)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_abpc3,z_abpc3, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_abpc3,w_abpc3, 'm', marker = '*', linewidth=2)
#
plt.xlabel('t3')
plt.ylabel('w(t3)')
plt.legend([' Exact Solution', 'AB Predictor Corrector Method, Example #3'], loc = 'lower right' )
#        
plt.show()
#==============================================================================

############ Example #4 3c 291 Given Points 

N_abpc4 = 10
a_abpc4 = 0.0
b_abpc4 = 2.0
ya_abpc4 = -2.0

# defining function and true solution of function #4
def_fn_abpc4 = exmp_fn.exmp4_def_fn
sol_abpc4 = exmp_fn.exmp4_sol


# run Euler's method from ODE_Approx_methods for example #4

(t_abpc4, w_abpc4) = ODE_Approx_methods.AB_Predictor_Corrector(def_fn_abpc4, a_abpc4, b_abpc4, ya_abpc4, N_abpc4)

# Exact solutions for comparison of example #3

z_abpc4 = sol_abpc4(t_abpc4)


# plot comparison of exact solution w(t) and approximation y(t), example 4'
plt.figure(22)

print('Errors at time mesh points, Example #4: ')
print(np.abs(np.transpose(np.ravel(w_abpc4)-z_abpc4)))

plt.rcParams.update({'font.size': 20})
plt.plot(t_abpc4,z_abpc4, 'b-' , marker= 'o', linewidth=2)
plt.plot(t_abpc4,w_abpc4, 'c-', marker = '*', linewidth=2)
#
plt.xlabel('t4')
plt.ylabel('w(t4)')
plt.legend([' Exact Solution', 'AB Predictor Corrector Method, Example #4'], loc = 'lower right' )
#        
plt.show()
############################################################


#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================






"""
    This section is used for testing the various methods on an extended range system
    per function #1. 
    This section we will increase the amount of points to be iterated for function #1
    so that valid comparisons of accuracy and stability of each method can be made
    
    
"""

# Euler's Method on Extended Mesh system

N_ext_eul = 50  # number of mesh points
a_ext_eul = 0.0 # left end point of interval [a,b]
b_ext_eul = 5.0 # right end point of interval [a,b]
ya_ext_eul = 0.5 # initial value y(a)


# defining function and true solution of function #1 
def_fn_ext_eul = exmp_fn.exmp1_def_fn
sol_ext_eul = exmp_fn.exmp1_sol

# eulers approx for example 1 function
(t_ext_eul, y_ext_eul) = ODE_Approx_methods.euler(def_fn_ext_eul, a_ext_eul, b_ext_eul, ya_ext_eul, N_ext_eul)

# compute exact for example #1 solution for comparison
z_ext_eul = sol(t_ext_eul) 

#Plotting example #1 

print('Errors at time mesh points, Example #1:')
print(np.abs(np.transpose(np.ravel(y_ext_eul)-z_ext_eul)))


# plot comparison of exact solution z(t) and approximation y(t), example 1
plt.figure(23)

plt.rcParams.update({'font.size': 20})	# set plot font size
plt.plot(t_ext_eul, z_ext_eul, 'b-', marker='o', linewidth=2)	# plot true solution z
plt.plot(t_ext_eul, y_ext_eul, 'r--', marker='*', linewidth=2)	# plot approximation by Euler's method



plt.xlabel('t')	# set x-axis label as t
plt.ylabel('y(t)')	# set y-axis label as y(t)
plt.legend(['Exact solution, Function #1 with extended N points', 'Eulers Method' ], loc='lower right')	# set legend and location

plt.show()

#print('Exact solutions of example #1:',z)   
      
         

##############################################################################
##############################################################################

# Modified Eulers Method on Extended Mesh system

N_ext_modeul = 50  # number of mesh points
a_ext_modeul = 0.0 # left end point of interval [a,b]
b_ext_modeul = 5.0 # right end point of interval [a,b]
ya_ext_modeul = 0.5 # initial value y(a)


# defining function and true solution of function #1 
def_fn_ext_modeul = exmp_fn.exmp1_def_fn
sol_ext_modeul = exmp_fn.exmp1_sol

# eulers approx for example 1 function
(t_ext_modeul, y_ext_modeul) = ODE_Approx_methods.modi_eulers(def_fn_ext_modeul, a_ext_modeul, b_ext_modeul, ya_ext_modeul, N_ext_modeul)

# compute exact for example #1 solution for comparison
z_ext_modeul = sol(t_ext_modeul) 

#Plotting example #1 

print('Errors at time mesh points, Example #1:')
print(np.abs(np.transpose(np.ravel(y_ext_modeul)-z_ext_modeul)))


# plot comparison of exact solution z(t) and approximation y(t), example 1
plt.figure(24)

plt.rcParams.update({'font.size': 20})	# set plot font size
plt.plot(t_ext_modeul, z_ext_modeul, 'b-', marker='o', linewidth=2)	# plot true solution z
plt.plot(t_ext_modeul, y_ext_modeul, 'r--', marker='*', linewidth=2)	# plot approximation by Euler's method



plt.xlabel('t')	# set x-axis label as t
plt.ylabel('y(t)')	# set y-axis label as y(t)
plt.legend(['Exact solution, Function #1 with extended N points', 'Modified Eulers Method' ], loc='lower right')	# set legend and location

plt.show()

#print('Exact solutions of example #1:',z)  

##############################################################################
##############################################################################

# Midpoint Method on Extended Mesh system

N_ext_mid = 50  # number of mesh points
a_ext_mid = 0.0 # left end point of interval [a,b]
b_ext_mid = 5.0 # right end point of interval [a,b]
ya_ext_mid = 0.5 # initial value y(a)


# defining function and true solution of function #1 
def_fn_ext_mid = exmp_fn.exmp1_def_fn
sol_ext_mid = exmp_fn.exmp1_sol

# eulers approx for example 1 function
(t_ext_mid, y_ext_mid) = ODE_Approx_methods.Midpoint_meth(def_fn_ext_mid, a_ext_mid, b_ext_mid, ya_ext_mid, N_ext_mid)

# compute exact for example #1 solution for comparison
z_ext_mid = sol(t_ext_mid) 

#Plotting example #1 

print('Errors at time mesh points, Example #1:')
print(np.abs(np.transpose(np.ravel(y_ext_mid)-z_ext_mid)))


# plot comparison of exact solution z(t) and approximation y(t), example 1
plt.figure(25)

plt.rcParams.update({'font.size': 20})	# set plot font size
plt.plot(t_ext_mid, z_ext_mid, 'b-', marker='o', linewidth=2)	# plot true solution z
plt.plot(t_ext_mid, y_ext_mid, 'r--', marker='*', linewidth=2)	# plot approximation by Euler's method



plt.xlabel('t')	# set x-axis label as t
plt.ylabel('y(t)')	# set y-axis label as y(t)
plt.legend(['Exact solution, Function #1 with extended N points', 'Midpoint Method' ], loc='lower right')	# set legend and location

plt.show()


##############################################################################
##############################################################################

# RK4 on Extended Mesh system

N_ext_rk4 = 50  # number of mesh points
a_ext_rk4 = 0.0 # left end point of interval [a,b]
b_ext_rk4 = 5.0 # right end point of interval [a,b]
ya_ext_rk4 = 0.5 # initial value y(a)


# defining function and true solution of function #1 
def_fn_ext_rk4 = exmp_fn.exmp1_def_fn
sol_ext_rk4 = exmp_fn.exmp1_sol

# eulers approx for example 1 function
(t_ext_rk4, y_ext_rk4) = ODE_Approx_methods.Runge_kutta4(def_fn_ext_rk4, a_ext_rk4, b_ext_rk4, ya_ext_rk4, N_ext_rk4)

# compute exact for example #1 solution for comparison
z_ext_rk4 = sol(t_ext_rk4) 

#Plotting example #1 

print('Errors at time mesh points, Example #1:')
print(np.abs(np.transpose(np.ravel(y_ext_rk4)-z_ext_rk4)))


# plot comparison of exact solution z(t) and approximation y(t), example 1
plt.figure(26)

plt.rcParams.update({'font.size': 20})	# set plot font size
plt.plot(t_ext_rk4, z_ext_rk4, 'b-', marker='o', linewidth=2)	# plot true solution z
plt.plot(t_ext_rk4, y_ext_rk4, 'r--', marker='*', linewidth=2)	# plot approximation by Euler's method



plt.xlabel('t')	# set x-axis label as t
plt.ylabel('y(t)')	# set y-axis label as y(t)
plt.legend(['Exact solution, Function #1 with extended N points', 'RK4 Method' ], loc='lower right')	# set legend and location

plt.show()

##############################################################################
##############################################################################

# AB on Extended Mesh system

N_ext_ab = 50  # number of mesh points
a_ext_ab = 0.0 # left end point of interval [a,b]
b_ext_ab = 5.0 # right end point of interval [a,b]
ya_ext_ab = 0.5 # initial value y(a)


# defining function and true solution of function #1 
def_fn_ext_ab = exmp_fn.exmp1_def_fn
sol_ext_ab = exmp_fn.exmp1_sol

# eulers approx for example 1 function
(t_ext_ab, y_ext_ab) = ODE_Approx_methods.AB_Explicit_Meth(def_fn_ext_ab, a_ext_ab, b_ext_ab, ya_ext_ab, N_ext_ab)

# compute exact for example #1 solution for comparison
z_ext_ab = sol(t_ext_ab) 

#Plotting example #1 

print('Errors at time mesh points, Example #1:')
print(np.abs(np.transpose(np.ravel(y_ext_ab)-z_ext_ab)))


# plot comparison of exact solution z(t) and approximation y(t), example 1
plt.figure(27)

plt.rcParams.update({'font.size': 20})	# set plot font size
plt.plot(t_ext_ab, z_ext_ab, 'b-', marker='o', linewidth=2)	# plot true solution z
plt.plot(t_ext_ab, y_ext_ab, 'r--', marker='*', linewidth=2)	# plot approximation by Euler's method



plt.xlabel('t')	# set x-axis label as t
plt.ylabel('y(t)')	# set y-axis label as y(t)
plt.legend(['Exact solution, Function #1 with extended N points', 'AB Bashforth Method' ], loc='lower right')	# set legend and location

plt.show()
     
         
##############################################################################
##############################################################################

# Predictor Corrector on Extended Mesh system

N_ext_pc = 50  # number of mesh points
a_ext_pc = 0.0 # left end point of interval [a,b]
b_ext_pc = 5.0 # right end point of interval [a,b]
ya_ext_pc = 0.5 # initial value y(a)


# defining function and true solution of function #1 
def_fn_ext_pc = exmp_fn.exmp1_def_fn
sol_ext_pc = exmp_fn.exmp1_sol

# eulers approx for example 1 function
(t_ext_pc, y_ext_pc) = ODE_Approx_methods.AB_Predictor_Corrector(def_fn_ext_pc, a_ext_pc, b_ext_pc, ya_ext_pc, N_ext_pc)

# compute exact for example #1 solution for comparison
z_ext_pc = sol(t_ext_pc) 

#Plotting example #1 

print('Errors at time mesh points, Example #1:')
print(np.abs(np.transpose(np.ravel(y_ext_pc)-z_ext_pc)))


# plot comparison of exact solution z(t) and approximation y(t), example 1
plt.figure(28)

plt.rcParams.update({'font.size': 20})	# set plot font size
plt.plot(t_ext_pc, z_ext_pc, 'b-', marker='o', linewidth=2)	# plot true solution z
plt.plot(t_ext_pc, y_ext_pc, 'r--', marker='*', linewidth=2)	# plot approximation by Euler's method



plt.xlabel('t')	# set x-axis label as t
plt.ylabel('y(t)')	# set y-axis label as y(t)
plt.legend(['Exact solution, Function #1 with extended N points', 'Predictor Corrector' ], loc='lower right')	# set legend and location

plt.show()