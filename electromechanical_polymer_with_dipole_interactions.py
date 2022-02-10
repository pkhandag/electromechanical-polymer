""" 
Created in: 2022
Purpose: Obtain equilibrium properties of a polarizable polymer chain (such as average chain density, average polarization, and electric field in the domain) 
under applied electric field while accouting for non-local dipole-dipole interactions among polymer segments. 
Contact: Pratik Khandagale (pkhandag@andrew.cmu.edu)
"""



#imports
from __future__ import print_function
from fenics import *
from ufl import *
from boxfield import *
from scipy.optimize import fsolve
from numpy.linalg import svd
from sympy import Matrix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import linalg, matrix
from scipy.integrate import odeint
from tempfile import TemporaryFile
from dolfin import *
from mshr import *
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations_with_replacement  
from numpy import linalg as LA
from scipy.linalg import sqrtm
from xlwt import Workbook 

import numpy as np
import matplotlib.pyplot as plt
import math 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import xlwt 
from fenicstools import interpolate_nonmatching_mesh




## orientation of applied electric field
E0_theta=pi/2

#scaling factor for dipole-dipole interaction
alpha=1. #alpha lies between 0 and 1

## magnitude of applied electric field
E0_magn= 1.5

E0_1=E0_magn*cos(E0_theta)
E0_2=E0_magn*sin(E0_theta)




# # polymer parameters
k_B=1.0 # Boltzmann constant (in normalized setting)
Temp=1.0 #temperature of the polymer network  (in normalized setting)
epsilon_0=8.854e-12 # vaccum permittivity



#computational parameters
no_of_elements= 30 #no of finite elements alng each X, Y and Z axis 


nx_V= no_of_elements #no of finite elements along x-axis
ny_V= no_of_elements #no of finite elements along y-axis
nz_V= no_of_elements




dt=0.1  # should be < 0.1   ,  Time step along chain contour. To satisfy CFL numerical stability condition, we need dt < (((x_max_box-x_min_box)/nx_V)**2)/G_chain

lambda_stretch=1.0
lambda_parameter=0.01 # vary between 0.001 to 10 or 100
chi_parallel= 1.0
chi_perpendicular= 0.5

   


#### BC on phi_electrostatic
phi_electrostatic_on_boundary = Expression('- (E0_1*x[0]+E0_2*x[1])', E0_1=E0_1, E0_2=E0_2, degree=2)




delta_Q_ratio_threshold=1e-3 ## iteration stopping criteria: threshold for relative change in total free energy 

L_chain=10.0
c_dirac=0.1 #choose as 0.1, standard deviation of Gaussian used to approximate Dirac delta function in the initial condition for q and q*
c_dirac_z=c_dirac*pi

dirac_constant=1.
field_constant=1.
polarization_constant=1.
box_length_constant=1.3



N_chain=100
a_chain= 1/sqrt(N_chain)
l_RMS= a_chain*(N_chain)**(1/2) ## RMS end-to-end length of one chain



T=10.0 # final value of chain parameter 's' 
n_t=int(T/dt +1) # number of time steps along the chain contour 
round_const= 12 # no of significant digits  
tol=2e-16 ## tolerance to form submeshes




## Point 0 coordinates
# x0=  -round( L_chain/2 , round_const )   #x component of X1 vector
x0=  round( 0 , round_const )    #x component of X1 vector
y0= -round( 0.35*l_RMS , round_const )   #y component of X1 vector
z0=  round( 0 , round_const )   #z component of X1 vector    
X0=np.array([x0,y0,z0]) #position vector of 1st chain start end


## Point 1 coordinates
x1=  round( 0 , round_const )    #x component of X1 vector
y1=  round( 0.35*l_RMS , round_const )   #y component of X1 vector
z1=  round( 0 , round_const )  #z component of X1 vector    
X1=np.array([x1,y1,z1]) #position vector of 1st chain start end







#####################
## initializing variable values
#####################

delta_Q_ratio_array=np.zeros(1000) 
delta_Q_ratio_array[0]=5
delta_Q_ratio=5

############################################################### 
## V mesh forming
############################################################### 


x_min_box= -round( box_length_constant*l_RMS , round_const )
y_min_box= -round( box_length_constant*l_RMS , round_const )
z_min_box=  round( 0 , round_const )

x_max_box=  round( box_length_constant*l_RMS , round_const )
y_max_box=  round( box_length_constant*l_RMS , round_const )
z_max_box=  round( 2*pi , round_const )




## domian volume as constant of proportionality for Q and rho 
V_domain= N_chain*(a_chain**2)



## Create mesh and define function space and dof coordinates
mesh = BoxMesh(Point(x_min_box, y_min_box, z_min_box), Point(x_max_box, y_max_box, z_max_box), nx_V, ny_V, nz_V)   


############################################################### 

                                         
# V = FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=PeriodicBoundary())        
V = FunctionSpace(mesh, 'Lagrange', 1)        



n_mesh = V.dim()   #no of dof points, n_mesh=(nx+1)*(ny+1)                                                                
d_mesh = mesh.geometry().dim()                                                        
dof_coordinates = V.tabulate_dof_coordinates()                   
dof_coordinates.resize((n_mesh, d_mesh))                                                   
dof_x = dof_coordinates[:, 0]                                                    
dof_y = dof_coordinates[:, 1]     
dof_z = dof_coordinates[:, 2]                                                  


############################################################### 
## V_2D mesh forming
############################################################### 

nx_V= no_of_elements #no of finite elements along x-axis
ny_V= no_of_elements #no of finite elements along y-axis

## domian volume as constant of proportionality for Q and rho 
V_domain_2D= (x_max_box- x_min_box)* (y_max_box- y_min_box)



## Create mesh and define function space and dof coordinates
mesh_2D = RectangleMesh(Point(x_min_box, y_min_box), Point(x_max_box, y_max_box), nx_V, ny_V)   



############################################################### 

                                         
# V = FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=PeriodicBoundary())        
V_2D = FunctionSpace(mesh_2D, 'Lagrange', 1)        


n_mesh_2D = V_2D.dim()   #no of dof points, n_mesh=(nx+1)*(ny+1)                                                                
d_mesh_2D = mesh_2D.geometry().dim()                                                        
dof_coordinates_2D = V_2D.tabulate_dof_coordinates()                   
dof_coordinates_2D.resize((n_mesh_2D, d_mesh_2D))                                                   
dof_x_2D = dof_coordinates_2D[:, 0]                                                    
dof_y_2D = dof_coordinates_2D[:, 1]     





############################################################### 
## defining vector field function space
############################################################### 

V_3D_vector = VectorFunctionSpace(mesh, "Lagrange", 1)

V_2D_vector = VectorFunctionSpace(mesh_2D, "Lagrange", 1)



theta = Expression("x[2]", degree=1)






############################################################### 
## functions for solving q, q_star, Q, phi 
############################################################### 


################################################################################
## Function for computing q 

def q_computation(X, w):


    
    ##initial q and q_star at t=0
    q_n= Function(V) 
   
    
    ##initial condition for q
    x_cord= X[0]
    y_cord= X[1]
    z_cord= X[2]

    ## dirac in only x and y
    q_0_expression=  Expression( ' dirac_constant*  V_domain*  ( pow(  1/(sqrt(2*pi)*c_dirac) , 2 ) ) * exp(   ( -1/ ( 2*pow(c_dirac,2) ) )  * (    pow(( x[0]- x_cord), 2) +  pow(( x[1]- y_cord), 2)  )    )  ' , pi=pi, lambda_parameter=lambda_parameter, V_domain=V_domain, dirac_constant=dirac_constant, c_dirac=c_dirac, c_dirac_z=c_dirac_z, x_cord=x_cord, y_cord=y_cord, z_cord=z_cord, degree=2 )    


    q_0= interpolate(q_0_expression, V)

    ## write initial condition to file
    xdmf_q.write_checkpoint(q_0, "q_label", 0,  XDMFFile.Encoding.HDF5, False)
      
    ##initialize q value at t=0
    q_n.assign(q_0)
    
    
    ######## time stepping for computing q
    for n in range(1,n_t): 
        
        # print(n)
        
        t=dt*n

        
        #defining q, v
        q = TrialFunction(V)
        v = TestFunction(V)
        
        
        ## dielectric chain - Crank-Nicolson time stepping
        a= q*v*dx +(dt/2)*w*q*v*dx + (dt/2)* cos(theta)*(q.dx(0))*v*dx + (dt/2)* sin(theta)*(q.dx(1))*v*dx + (dt/(4*lambda_parameter))*(q.dx(2))*(v.dx(2))*dx 
        L= q_n*v*dx -(dt/2)*w*q_n*v*dx - (dt/2)* cos(theta)*(q_n.dx(0))*v*dx - (dt/2)* sin(theta)*(q_n.dx(1))*v*dx - (dt/(4*lambda_parameter))*(q_n.dx(2))*(v.dx(2))*dx 
        
        
        
        #solve variational problem
        q = Function(V)
        # solve(a == L, q, solver_parameters={'linear_solver': 'gmres', 'preconditioner': 'ilu'})
        solve(a == L, q)
        # solve(a == L, q, solver_parameters={'linear_solver': 'mumps'}  )
        # solve(a == L, q, solver_parameters={'linear_solver': 'cg'}  )



        
        #saving solution  to file        
        xdmf_q.write_checkpoint(q, "q_label", t,  XDMFFile.Encoding.HDF5, True)
        
        #Update previous solution
        q_n.assign(q)
        
    
    ######## end of time stepping
    xdmf_q.close()
    
    
    ## returning value of function
    return (q)



        
    
################################################################################
## Function for computing q_star

def q_star_computation(X, w):

    
    ##initial q_star at t=0
    q_star_n= Function(V) 
       
    
    ##initial condition for q_star 
    x_cord= X[0]
    y_cord= X[1]
    z_cord= X[2]
 
    ## dirac in only x and y
    q_star_0_expression=  Expression( ' dirac_constant*  V_domain*  ( pow(  1/(sqrt(2*pi)*c_dirac) , 2 ) ) * exp(   ( -1/ ( 2*pow(c_dirac,2) ) )  * (    pow(( x[0]- x_cord), 2) +  pow(( x[1]- y_cord), 2)  )    )  ' , pi=pi, lambda_parameter=lambda_parameter, V_domain=V_domain, dirac_constant=dirac_constant, c_dirac=c_dirac, c_dirac_z=c_dirac_z, x_cord=x_cord, y_cord=y_cord, z_cord=z_cord, degree=2 )    



    q_star_0= interpolate(q_star_0_expression, V)

    #write
    xdmf_q_star.write_checkpoint(q_star_0, "q_star_label", 0,  XDMFFile.Encoding.HDF5, False)
      
    ##initialize q_star value at t=0
    q_star_n.assign(q_star_0)

    ######## time stepping for q_star
    for n in range(1,n_t): 
        
        
        t=dt*n
       
        ######################################### computing q_star
        
        #defining q* and v*
        q_star = TrialFunction(V)
        v_star = TestFunction(V)
        
        
        ## dielectric chain - Crank-Nicolson time stepping
        a_star= q_star*v_star*dx +(dt/2)*w*q_star*v_star*dx + (dt/2)* cos(theta)*(q_star.dx(0))*v_star*dx + (dt/2)* sin(theta)*(q_star.dx(1))*v_star*dx + (dt/(4*lambda_parameter))*(q_star.dx(2))*(v_star.dx(2))*dx 
        L_star= q_star_n*v_star*dx -(dt/2)*w*q_star_n*v_star*dx - (dt/2)* cos(theta)*(q_star_n.dx(0))*v_star*dx - (dt/2)* sin(theta)*(q_star_n.dx(1))*v_star*dx - (dt/(4*lambda_parameter))*(q_star_n.dx(2))*(v_star.dx(2))*dx 
        
        
                
        #solve variational problem
        q_star=Function(V)
        # solve(a_star == L_star, q_star, solver_parameters={'linear_solver': 'gmres', 'preconditioner': 'ilu'})
        solve(a_star == L_star, q_star)
        # solve(a_star == L_star, q_star, solver_parameters={'linear_solver': 'mumps'}  )
        # solve(a_star == L_star, q_star, solver_parameters={'linear_solver': 'cg'}  )




    
        #saving solution to file
        xdmf_q_star.write_checkpoint(q_star, "q_star_label", t,  XDMFFile.Encoding.HDF5, True)
        
         
        #Update previous solution
        q_star_n.assign(q_star)
    
    
    
    #### end of time stepping for q_star
    xdmf_q_star.close()

    ## returning values from function
    return (q_star)

################################################################################       


############################################################### 
## for calculating chain density at chosen z
############################################################### 


# an Expression class to evaluate f(x,y,z) at a fixed position z,
# with x and y from the subspace
class my2DExpression(UserExpression):
    def __init__(self,f_3d,zpos):
        super().__init__(zpos)
        self.f_3d = f_3d
        self.zpos = zpos
        self.value = np.array([0.])
        self.point = np.array([0.,0.,0.])
    def eval(self, values, x):
        # x[] is a location in the 2d subspace, as a two element numpy array
        self.point[0] = x[0]
        self.point[1] = x[1]
        self.point[2] = self.zpos
        self.f_3d.eval(self.value,self.point)
        values[0] = self.value[0]
    def value_shape(self):
        return ()


    



############################################################### 
## for integration along 1D
############################################################### 

## Create mesh and define function space 
mesh_1D = IntervalMesh(nz_V, z_min_box, z_max_box)                                          
V_1D = FunctionSpace(mesh_1D, 'Lagrange', 1)        



# an Expression class to evaluate f(x,y,z) at a fixed position in (x,y) 
# with z from the subspace
class my1DExpression(UserExpression):
    def __init__(self,f_3d, xpos, ypos):
        super().__init__(xpos)
        super().__init__(ypos)
        self.f_3d = f_3d
        self.xpos = xpos
        self.ypos = ypos
        self.value = np.array([0.])
        self.point = np.array([0.,0.,0.])
    def eval(self, values, x):
        # x[] is a location in the 1d subspace, as a one element numpy array
        self.point[0] = self.xpos
        self.point[1] = self.ypos
        self.point[2] = x[0]
        self.f_3d.eval(self.value,self.point)
        values[0] = self.value[0]
    def value_shape(self):
        return ()


#DEfining a UserExpression for the F(x,y) function
class FxyExpression(UserExpression):
    def __init__(self,f_3d):
        super().__init__()
        self.f_3d = f_3d
    def eval(self, values, x):
        _Fxy = interpolate(my1DExpression(self.f_3d,xpos=x[0], ypos=x[1]),V_1D)
        values[0] = assemble(_Fxy*dx)
    def value_shape(self):
        return ()






################################################################################
   
# Function for single chain computation

def single_chain_computation():
       
    ##computing Q (Complete Partition Function for single chain)
    
    Q=np.zeros(n_t) # Complete Partition Function Q at each position along the chain
    
    phi_chain=Function(V)  # phi function
    phi_chain_numr= phi_chain.vector().get_local() 
       
       
    ## for calculating Q at s=0.3    
    q_temp_multiplied_3D= Function(V)
    q_temp_multiplied_3D_numr= q_temp_multiplied_3D.vector().get_local() 

    
    
    for i1 in range(n_t):
        
        # print(i1)

        q_temp = Function(V)
        xdmf_q_call =  XDMFFile("q.xdmf")
        xdmf_q_call.read_checkpoint(q_temp,"q_label",i1) 
        xdmf_q_call.close()
        
        q_star_temp = Function(V)
        xdmf_q_star_call =  XDMFFile("q_star.xdmf")
        xdmf_q_star_call.read_checkpoint(q_star_temp,"q_star_label", n_t-1-i1) 
        xdmf_q_star_call.close()
             
        
        ######## 
        q_temp_2D = Function(V_2D)
        q_star_temp_2D = Function(V_2D)
        
        q_temp_multiplied_2D = Function(V_2D)
        q_temp_multiplied_2D_numr= q_temp_multiplied_2D.vector().get_local() 
       

        
        for k in range(0, round(nz_V/2)):
            z1= (z_max_box-z_min_box)*(k/nz_V)
            z1_star= (z_max_box-z_min_box)*(round(k+nz_V/2)/nz_V)
            
            
            q_temp_2D= interpolate(my2DExpression(q_temp,zpos=z1),V_2D)
            q_star_temp_2D= interpolate(my2DExpression(q_star_temp,zpos=z1_star),V_2D)
            
            q_temp_2D_numr= q_temp_2D.vector().get_local() 
            q_star_temp_2D_numr= q_star_temp_2D.vector().get_local() 
            
            q_temp_multiplied_2D_numr= q_temp_2D_numr* q_star_temp_2D_numr
            q_temp_multiplied_2D.vector().set_local(q_temp_multiplied_2D_numr )
            q_temp_multiplied_2D.vector().apply('insert') 
            
                            
            for i2 in range(n_mesh):     
                # print(i2)
                x_tmp=dof_x[i2]
                y_tmp=dof_y[i2] 
                z_tmp=dof_z[i2]
                
                # if z_tmp==k:
                if near(z_tmp, z1):
                    phi_chain_numr[i2]= phi_chain_numr[i2] + q_temp_multiplied_2D(x_tmp, y_tmp)

                    if i1==round(n_t/2):                    
                        q_temp_multiplied_3D_numr[i2]=  q_temp_multiplied_2D(x_tmp, y_tmp)
                    
        
        
        for k in range(round(nz_V/2), round(nz_V+1)):
            
            z1= (z_max_box-z_min_box)*(k/nz_V)
            z1_star= (z_max_box-z_min_box)*(round(k-nz_V/2)/nz_V)

            q_temp_2D= interpolate(my2DExpression(q_temp,zpos=z1),V_2D)
            q_star_temp_2D= interpolate(my2DExpression(q_star_temp,zpos=z1_star),V_2D)
            
            q_temp_2D_numr= q_temp_2D.vector().get_local() 
            q_star_temp_2D_numr= q_star_temp_2D.vector().get_local() 
            
            q_temp_multiplied_2D_numr= q_temp_2D_numr* q_star_temp_2D_numr
            q_temp_multiplied_2D.vector().set_local(q_temp_multiplied_2D_numr )
            q_temp_multiplied_2D.vector().apply('insert') 
            
                            
            for i2 in range(n_mesh):     
                # print(i2)
                x_tmp=dof_x[i2]
                y_tmp=dof_y[i2] 
                z_tmp=dof_z[i2]
                
                # if z_tmp==k:
                if near(z_tmp, z1):
                    phi_chain_numr[i2]= phi_chain_numr[i2] + q_temp_multiplied_2D(x_tmp, y_tmp)
                                    
                    if i1==round(n_t/2):                    
                        q_temp_multiplied_3D_numr[i2]=  q_temp_multiplied_2D(x_tmp, y_tmp)
                    
                    
    ##############################################################
    ## Calculating Q at s=0.5
    ##############################################################
                        
    q_temp_multiplied_3D.vector().set_local(q_temp_multiplied_3D_numr )
    q_temp_multiplied_3D.vector().apply('insert')         
   
    Q_chain=assemble(q_temp_multiplied_3D*dx)/V_domain #Q is normalized with dividing by volume of the domain


    ##############################################################
    ## Calculating avg. chain density
    ##############################################################


    phi_chain_numr= (1/(V_domain*Q_chain))*phi_chain_numr        
         
    phi_chain.vector().set_local(phi_chain_numr )
    phi_chain.vector().apply('insert')                 

    return (Q_chain, phi_chain, phi_chain_numr)

##################################################
        
            



############################################################### 
## Solving for initial phi_electrostatic 
############################################################### 


## boundary condition
bc_phi_electrostatic = DirichletBC(V_2D, phi_electrostatic_on_boundary, 'on_boundary')


## declare trial and test function
phi_electrostatic  = TrialFunction(V_2D)
v = TestFunction(V_2D)

## define a and L
a = dot(grad(phi_electrostatic), grad(v))*dx
L = Constant(0)*v*dx



# Compute solution
phi_electrostatic=Function(V_2D)
solve(a == L, phi_electrostatic, bc_phi_electrostatic)

# File('dielectric_phi_electrostatic.pvd') << (phi_electrostatic)
    








############################################################### 
## Electric field 
############################################################### 


E1= project(- phi_electrostatic.dx(0), V_2D)
E2= project(- phi_electrostatic.dx(1), V_2D)

    
E=Function(V_2D_vector)

assigner = FunctionAssigner(V_2D_vector, [V_2D, V_2D])
assigner.assign(E, [E1, E2])




############################################################### 
## forming w
############################################################### 


w = Function(V)
w_numr= w.vector().get_local() 

                    
for i2 in range(n_mesh):     
    # print(i2)
    x_tmp=dof_x[i2]
    y_tmp=dof_y[i2] 
    z_tmp=dof_z[i2]

    w_numr[i2]= - V_domain* epsilon_0 * ( (chi_parallel-chi_perpendicular)* ( E1(x_tmp, y_tmp)**2 * np.cos(z_tmp)**2 + E2(x_tmp, y_tmp)**2 * np.sin(z_tmp)**2  + 2*E1(x_tmp, y_tmp)*E2(x_tmp, y_tmp) *np.sin(z_tmp)* np.cos(z_tmp) ) + chi_perpendicular*( E1(x_tmp, y_tmp)**2 + E2(x_tmp, y_tmp)**2 ) )



w.vector().set_local(w_numr)
w.vector().apply('insert')   

# File('w.pvd') << (w)




############################################################### 
## initial chain computation
############################################################### 

 
############################################################### 
print('step 3: point 0')
############################################################### 


# generating vtk files to store q_point_1
xdmf_q = XDMFFile("q.xdmf")

q_point_0=Function(V) 
q_point_0= q_computation(X0, w)


############################################################### 
print('step 4: point 1')
############################################################### 



# generating vtk files to store q_point_1
xdmf_q_star = XDMFFile("q_star.xdmf")

q_point_1=Function(V) 
q_point_1= q_star_computation(X1, w)




############################################################### 
print('step 5: chain computation')
############################################################### 


Q1_mid, phi_chain_1, phi_chain_1_numr = single_chain_computation()




# File('dielectric_q_point_0.pvd') << (q_point_0)
# File('dielectric_q_point_1.pvd') << (q_point_1)
# File('dielectric_phi_chain_1.pvd') << (phi_chain_1)






count=1
print('count:', count)
###############################################################  Iterating for finding equilibrium mean field w

while abs(delta_Q_ratio) > delta_Q_ratio_threshold:



    
    ############################################################### 
    ## get polarization from chain density 
    ############################################################### 
    
       
    ############################################################### 
    print('step 6: forming polarization')
    ############################################################### 
        
    phi_chain_numr= phi_chain_1.vector().get_local() 
    
    
    f_tmp_1 = Function(V)
    f_tmp_2 = Function(V)
    
    f_tmp_1_numr=f_tmp_1.vector().get_local() 
    f_tmp_2_numr=f_tmp_2.vector().get_local() 
    
                        
    for i2 in range(n_mesh):     
        # print(i2)
        x_tmp=dof_x[i2]
        y_tmp=dof_y[i2] 
        z_tmp=dof_z[i2]
        
        f_tmp_1_numr[i2]= polarization_constant*  V_domain* epsilon_0 * ( (chi_parallel-chi_perpendicular)* ( E1(x_tmp, y_tmp)*np.cos(z_tmp)**2 * phi_chain_numr[i2] + E2(x_tmp, y_tmp)*np.sin(z_tmp)*np.cos(z_tmp)* phi_chain_numr[i2] ) + chi_perpendicular* E1(x_tmp, y_tmp)* phi_chain_numr[i2] )
        f_tmp_2_numr[i2]= polarization_constant*  V_domain* epsilon_0 * ( (chi_parallel-chi_perpendicular)* ( E1(x_tmp, y_tmp)*np.sin(z_tmp)*np.cos(z_tmp) * phi_chain_numr[i2] + E2(x_tmp, y_tmp)*np.sin(z_tmp)**2 * phi_chain_numr[i2] ) + chi_perpendicular* E2(x_tmp, y_tmp)* phi_chain_numr[i2] )
    
    
                           
    f_tmp_1.vector().set_local(f_tmp_1_numr )
    f_tmp_1.vector().apply('insert')         
                
    f_tmp_2.vector().set_local(f_tmp_2_numr )
    f_tmp_2.vector().apply('insert')         
      

     
    # File('f_tmp_1.pvd') << (f_tmp_1)
    # File('f_tmp_2.pvd') << (f_tmp_2) 
     
    
    ######################################
     
    
    ############################################################### 
    ## integrating in one dimension
    ############################################################### 
    
    
    f_tmp_1_integrated_on_2D= Function(V_2D)
    f_tmp_2_integrated_on_2D= Function(V_2D)
    
    f_tmp_1_integrated_on_2D = interpolate(FxyExpression(f_tmp_1), V_2D)
    f_tmp_2_integrated_on_2D = interpolate(FxyExpression(f_tmp_2), V_2D)
    
    
    # File('dielectric_f_tmp_1_integrated_on_2D.pvd') << (f_tmp_1_integrated_on_2D)
    # File('dielectric_f_tmp_2_integrated_on_2D.pvd') << (f_tmp_2_integrated_on_2D)
    
    
    ## forming polarization vector field
    p= Function(V_2D_vector) 
    assigner = FunctionAssigner(V_2D_vector, [V_2D, V_2D])
    assigner.assign(p, [f_tmp_1_integrated_on_2D, f_tmp_2_integrated_on_2D])
    
    File('dielectric_p.pvd') << (p)
    
    
    ############################################################### 
    ## Solving for phi_electrostatic using polarization
    ############################################################### 
    
    
    ############################################################### 
    print('step 8: get electic potential from polarization ')
    ############################################################### 
    
    
    ## boundary condition
    bc_phi_electrostatic = DirichletBC(V_2D, phi_electrostatic_on_boundary, 'on_boundary')
    
    
    ## div. of polarization
    div_of_p=div(p)
    
    
    ## declare trial and test function
    phi_electrostatic  = TrialFunction(V_2D)
    v = TestFunction(V_2D)
    
    ## define a and L
    a = - dot(grad(phi_electrostatic), grad(v))*dx
    L = alpha*(1/epsilon_0)* div_of_p*v*dx
    
    
    # Compute solution
    phi_electrostatic=Function(V_2D)
    solve(a == L, phi_electrostatic, bc_phi_electrostatic)
    
    # File('dielectric_phi_electrostatic.pvd') << (phi_electrostatic)
    
 
    
    
    ############################################################### 
    ## Obtain w
    ############################################################### 
    
    
    ############################################################### 
    print('step 9: obtain new field ')
    ###############################################################     
    
    
    ############################################################### 
    ## Obtain electric field from phi_electrostatic
    ############################################################### 
    
    
    E1= project(- phi_electrostatic.dx(0), V_2D)
    E2= project(- phi_electrostatic.dx(1), V_2D)
        
        
    E=Function(V_2D_vector)
    
    assigner = FunctionAssigner(V_2D_vector, [V_2D, V_2D])
    assigner.assign(E, [E1, E2])
    

    
    
    ############################################################### 
    ## forming w 
    ############################################################### 
    
    
    w = Function(V)
    w_numr= w.vector().get_local() 
    
                        
    for i2 in range(n_mesh):     
        # print(i2)
        x_tmp=dof_x[i2]
        y_tmp=dof_y[i2] 
        z_tmp=dof_z[i2]
    
        w_numr[i2]= - V_domain* epsilon_0 * ( (chi_parallel-chi_perpendicular)* ( E1(x_tmp, y_tmp)**2 * np.cos(z_tmp)**2 + E2(x_tmp, y_tmp)**2 * np.sin(z_tmp)**2  + 2*E1(x_tmp, y_tmp)*E2(x_tmp, y_tmp) *np.sin(z_tmp)* np.cos(z_tmp) ) + chi_perpendicular*( E1(x_tmp, y_tmp)**2 + E2(x_tmp, y_tmp)**2 ) )
    
    
    
    w.vector().set_local(w_numr)
    w.vector().apply('insert')   
    
    # File('w.pvd') << (w)



    
    
    ############################################################### 
    ## chain computation
    ############################################################### 
    
     
    ############################################################### 
    print('step 3: point 0')
    ############################################################### 
    
    # generating vtk files to store q_point_1
    xdmf_q = XDMFFile("q.xdmf")
    
    q_point_0=Function(V) 
    q_point_0= q_computation(X0, w)
    
    
    ############################################################### 
    print('step 4: point 1')
    ############################################################### 
    
    # generating vtk files to store q_point_1
    xdmf_q_star = XDMFFile("q_star.xdmf")
    
    q_point_1=Function(V) 
    q_point_1= q_star_computation(X1, w)
    
    
     
    
    ############################################################### 
    print('step 5: chain computation')
    ############################################################### 
    
    
    Q1_mid, phi_chain_1, phi_chain_1_numr = single_chain_computation()
   
    
    # File('dielectric_q_point_0.pvd') << (q_point_0)
    # File('dielectric_q_point_1.pvd') << (q_point_1)
    # File('dielectric_phi_chain_1.pvd') << (phi_chain_1)
    
       


    ############################################################### 
    ## checking stopping criteria
    ############################################################### 
    

    
    if count != 1:
        count=count+1
        delta_Q_ratio=((Q1_mid-Q1_mid_next)/Q1_mid_next)
        delta_Q_ratio_array[count]=delta_Q_ratio
        print('count:', count)
        print(delta_Q_ratio)
    
    
    if count != 1 and abs((Q1_mid-Q1_mid_next)/Q1_mid_next) < delta_Q_ratio_threshold:
        Q1_mid_next=Q1_mid
        print((Q1_mid-Q1_mid_next)/Q1_mid_next)
        print('iteration ended after break')
        break
    



    
    ############################################################### 
    ## getting w for next iteration
    ###############################################################  


    ############################################################### 
    ## get polarization from chain density 
    ############################################################### 
    
       
    ############################################################### 
    print('step 6: forming polarization')
    ############################################################### 
    
    
    phi_chain_numr= phi_chain_1.vector().get_local() 
    
    
    f_tmp_1 = Function(V)
    f_tmp_2 = Function(V)
    
    f_tmp_1_numr=f_tmp_1.vector().get_local() 
    f_tmp_2_numr=f_tmp_2.vector().get_local() 
    
                        
    for i2 in range(n_mesh):     
        # print(i2)
        x_tmp=dof_x[i2]
        y_tmp=dof_y[i2] 
        z_tmp=dof_z[i2]
        
        f_tmp_1_numr[i2]= polarization_constant*  V_domain* epsilon_0 * ( (chi_parallel-chi_perpendicular)* ( E1(x_tmp, y_tmp)*np.cos(z_tmp)**2 * phi_chain_numr[i2] + E2(x_tmp, y_tmp)*np.sin(z_tmp)*np.cos(z_tmp)* phi_chain_numr[i2] ) + chi_perpendicular* E1(x_tmp, y_tmp)* phi_chain_numr[i2] )
        f_tmp_2_numr[i2]= polarization_constant*  V_domain* epsilon_0 * ( (chi_parallel-chi_perpendicular)* ( E1(x_tmp, y_tmp)*np.sin(z_tmp)*np.cos(z_tmp) * phi_chain_numr[i2] + E2(x_tmp, y_tmp)*np.sin(z_tmp)**2 * phi_chain_numr[i2] ) + chi_perpendicular* E2(x_tmp, y_tmp)* phi_chain_numr[i2] )
    
    
                           
    f_tmp_1.vector().set_local(f_tmp_1_numr )
    f_tmp_1.vector().apply('insert')         
                
    f_tmp_2.vector().set_local(f_tmp_2_numr )
    f_tmp_2.vector().apply('insert')         
      
    

    # File('f_tmp_1.pvd') << (f_tmp_1)
    # File('f_tmp_2.pvd') << (f_tmp_2) 
     
     
    
    ############################################################### 
    ## integrating in one dimension
    ############################################################### 
    
    
    f_tmp_1_integrated_on_2D= Function(V_2D)
    f_tmp_2_integrated_on_2D= Function(V_2D)
    
    f_tmp_1_integrated_on_2D = interpolate(FxyExpression(f_tmp_1), V_2D)
    f_tmp_2_integrated_on_2D = interpolate(FxyExpression(f_tmp_2), V_2D)
    
    
    # File('dielectric_f_tmp_1_integrated_on_2D.pvd') << (f_tmp_1_integrated_on_2D)
    # File('dielectric_f_tmp_2_integrated_on_2D.pvd') << (f_tmp_2_integrated_on_2D)
    
    
    ## forming polarization vector field
    p= Function(V_2D_vector) 
    assigner = FunctionAssigner(V_2D_vector, [V_2D, V_2D])
    assigner.assign(p, [f_tmp_1_integrated_on_2D, f_tmp_2_integrated_on_2D])
    
    # File('dielectric_p.pvd') << (p)
    
    

        
    #########################################################


  
    ############################################################### 
    ## Solving for next phi_electrostatic using polarization
    ############################################################### 
    
    
    ############################################################### 
    print('step 8: get electic potential from polarization ')
    ############################################################### 
    
    
    ## boundary condition
    bc_phi_electrostatic = DirichletBC(V_2D, phi_electrostatic_on_boundary, 'on_boundary')
    
    
    ## div. of polarization
    div_of_p=div(p)
    
    
    ## declare trial and test function
    phi_electrostatic  = TrialFunction(V_2D)
    v = TestFunction(V_2D)
    
    ## define a and L
    a = - dot(grad(phi_electrostatic), grad(v))*dx
    L = alpha*(1/epsilon_0) * div_of_p*v*dx
    
    
    # Compute solution
    phi_electrostatic=Function(V_2D)
    solve(a == L, phi_electrostatic, bc_phi_electrostatic)
    
    # File('dielectric_phi_electrostatic.pvd') << (phi_electrostatic)


    
    ############################################################### 
    print('step 9: getting w for next iteration step ')
    ############################################################### 
    
    
    ############################################################### 
    ## Obtain next electric field from next phi_electrostatic
    ############################################################### 
        
        
    E1= project(- phi_electrostatic.dx(0), V_2D)
    E2= project(- phi_electrostatic.dx(1), V_2D)
        
        
    E=Function(V_2D_vector)
    
    assigner = FunctionAssigner(V_2D_vector, [V_2D, V_2D])
    assigner.assign(E, [E1, E2])
    
    
    ############################################################### 
    ## forming next w
    ############################################################### 
    
    w_next = Function(V)
    w_next_numr= w_next.vector().get_local() 
    
                        
    for i2 in range(n_mesh):     
        # print(i2)
        x_tmp=dof_x[i2]
        y_tmp=dof_y[i2] 
        z_tmp=dof_z[i2]
    
        w_next_numr[i2]= - V_domain* epsilon_0 * ( (chi_parallel-chi_perpendicular)* ( E1(x_tmp, y_tmp)**2 * np.cos(z_tmp)**2 + E2(x_tmp, y_tmp)**2 * np.sin(z_tmp)**2  + 2*E1(x_tmp, y_tmp)*E2(x_tmp, y_tmp) *np.sin(z_tmp)* np.cos(z_tmp) ) + chi_perpendicular*( E1(x_tmp, y_tmp)**2 + E2(x_tmp, y_tmp)**2 ) )
    
    
    w_next.vector().set_local(w_next_numr)
    w_next.vector().apply('insert')   
    
    # print(w_numr)
    # File('w_next.pvd') << (w_next)



    ###############################################################################
    ## computation for w_next
    ###############################################################################


    
    ############################################################### 
    ## chain computation
    ############################################################### 
    
     
    ############################################################### 
    print('step 3: point 0')
    ############################################################### 
    

    
    # generating vtk files to store q_point_1
    xdmf_q = XDMFFile("q.xdmf")
    
    q_point_0=Function(V) 
    q_point_0= q_computation(X0, w_next)
    
    
    ############################################################### 
    print('step 4: point 1')
    ############################################################### 
    
    
    # generating vtk files to store q_point_1
    xdmf_q_star = XDMFFile("q_star.xdmf")
    
    q_point_1=Function(V) 
    q_point_1= q_star_computation(X1, w_next)
    
    
     
    
    ############################################################### 
    print('step 5: chain computation')
    ############################################################### 
    
    
    Q1_mid_next, phi_chain_1, phi_chain_1_numr = single_chain_computation()

    
    
    # File('dielectric_q_point_0.pvd') << (q_point_0)
    # File('dielectric_q_point_1.pvd') << (q_point_1)
    # File('dielectric_phi_chain_1.pvd') << (phi_chain_1)
    
    
    
    
    ############################################################### 
    ## checking stopping criteria
    ############################################################### 
    
    count=count+1
    # computing relative change in Q and Q_next
    delta_Q_ratio=(Q1_mid_next-Q1_mid)/Q1_mid
    delta_Q_ratio_array[count]=delta_Q_ratio

    print('count:', count)
    print(delta_Q_ratio)


####################################################################
## END OF iteration
####################################################################


## converged free energy 
Q1_mid_converged=Q1_mid_next






############################################################### 
## save w
############################################################### 


File('w.pvd') << (w)


############################################################### 
## save electrostatic potential
############################################################### 


File('dielectric_phi_electrostatic.pvd') << (phi_electrostatic)


############################################################### 
## save equilibrium electric field
############################################################### 

File('dielectric_E.pvd') << (E)


############################################################### 
## save equilibrium chain density
############################################################### 

File('dielectric_phi_chain_1.pvd') << (phi_chain_1)



############################################################### 
## integrating chain density in one dimension
############################################################### 

 
############################################################### 
print('step : integrating chain density ')
############################################################### 


phi_chain_1_integrated_on_2D = interpolate(FxyExpression(phi_chain_1), V_2D)
File('dielectric_phi_chain_1_integrated_on_2D.pvd') << (phi_chain_1_integrated_on_2D)



############################################################### 
## get equilibrium polarization
############################################################### 

   
############################################################### 
print('step 6: forming polarization')
############################################################### 


phi_chain_numr= phi_chain_1.vector().get_local() 


f_tmp_1 = Function(V)
f_tmp_2 = Function(V)

f_tmp_1_numr=f_tmp_1.vector().get_local() 
f_tmp_2_numr=f_tmp_2.vector().get_local() 

                    
for i2 in range(n_mesh):     
    # print(i2)
    x_tmp=dof_x[i2]
    y_tmp=dof_y[i2] 
    z_tmp=dof_z[i2]
    
    f_tmp_1_numr[i2]= polarization_constant*  V_domain* epsilon_0 * ( (chi_parallel-chi_perpendicular)* ( E1(x_tmp, y_tmp)*np.cos(z_tmp)**2 * phi_chain_numr[i2] + E2(x_tmp, y_tmp)*np.sin(z_tmp)*np.cos(z_tmp)* phi_chain_numr[i2] ) + chi_perpendicular* E1(x_tmp, y_tmp)* phi_chain_numr[i2] )
    f_tmp_2_numr[i2]= polarization_constant*  V_domain* epsilon_0 * ( (chi_parallel-chi_perpendicular)* ( E1(x_tmp, y_tmp)*np.sin(z_tmp)*np.cos(z_tmp) * phi_chain_numr[i2] + E2(x_tmp, y_tmp)*np.sin(z_tmp)**2 * phi_chain_numr[i2] ) + chi_perpendicular* E2(x_tmp, y_tmp)* phi_chain_numr[i2] )


                       
f_tmp_1.vector().set_local(f_tmp_1_numr )
f_tmp_1.vector().apply('insert')         
            
f_tmp_2.vector().set_local(f_tmp_2_numr )
f_tmp_2.vector().apply('insert')         
  

 
File('f_tmp_1.pvd') << (f_tmp_1)
File('f_tmp_2.pvd') << (f_tmp_2) 
 

############################################################### 
## integrating in one dimension
############################################################### 


f_tmp_1_integrated_on_2D= Function(V_2D)
f_tmp_2_integrated_on_2D= Function(V_2D)

f_tmp_1_integrated_on_2D = interpolate(FxyExpression(f_tmp_1), V_2D)
f_tmp_2_integrated_on_2D = interpolate(FxyExpression(f_tmp_2), V_2D)


File('dielectric_f_tmp_1_integrated_on_2D.pvd') << (f_tmp_1_integrated_on_2D)
File('dielectric_f_tmp_2_integrated_on_2D.pvd') << (f_tmp_2_integrated_on_2D)


## forming polarization vector field
p= Function(V_2D_vector) 
assigner = FunctionAssigner(V_2D_vector, [V_2D, V_2D])
assigner.assign(p, [f_tmp_1_integrated_on_2D, f_tmp_2_integrated_on_2D])

File('dielectric_p.pvd') << (p)






############################################################### 
## plotting vector field E
############################################################### 

# c1=plot(E, width=.008)
# plt.colorbar(c1)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Electric field')


############################################################### 
## plotting vector field p
############################################################### 


# c2=plot(p, width=.008)
# plt.colorbar(c2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Avg polarization')



############################################################### 
## Calculating ratio of electrical to thermal energy
############################################################### 

# electrical_energy= epsilon_0*( assemble(dot(E,E)*dx) )* (1e-6)**2

# kB=1.3806e-23  # J/K
# thermal_energy=((T*1e-6)/(1.54e-10))*(kB)*300
# electrical_to_thermal_ratio= electrical_energy/ thermal_energy

# print('elecrical energy:' , electrical_energy )
# print('thermal energy:' , thermal_energy )
# print('elecrical energy/ thermal energy:' , electrical_to_thermal_ratio )


















