#!/usr/bin/env python
# coding: utf-8

# Import all the relevant modules.

# In[ ]:


import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from math import *
from cmath import *
import math
import matplotlib
import scipy.optimize as op
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import re


# # Electric Field Calculation(s)

# This code is built to simulate the electric dipole field at the secondary focus of an aplanatic lens, having been collimated by a high numerical aperture parabolic mirror. More specifically, we model the electric field introduced in eqn. (7) of "_A high numerical aperture parabolic mirror as imaging device for confocal microscopy_" by M. A. Lieb and A. J. Meixner, (2001). I have used their notation, so it should be possible to follow the code by using the paper's treatment as a reference.

# In[ ]:


fm = 1*10**(-3)     # Focal length of the mirror
fcl = 25*10**(-3)    # Focal length of the collection lens
NA = 1              # Numerical aperture of the mirror
M = fcl / fm          # Magnification factor of the setup
λ = 1550*10**(-9)     # Trapping wavelength
k = 2*pi / λ          # Wavenumber 
r_samp = 0*10**(-9)   # Size of the sample holder occluding the scattered field.
n = 1
print('Magnification Factor: {}'.format(M))


# In[ ]:


# Define the upper and lower limits of integration.

IntLimitUpper = np.arctan(4*10**(-3) / fcl); #print(IntLimitUpper)
IntLimitLower = np.arctan(r_samp / fcl); #print(IntLimitLower)

# Here θ is θ_(0) and ψ is θ_(m) from the analytical treatment.

def ψ(θ): return 2*np.arctan(M / 2 * np.sin(θ))

q0 = 500*10**(-9)         # Magnitude of displacement from the trap center.
q = np.array([0, 0, 0]) # Displacement vector of the nanoparticle
r = np.array([0, 0, 0]) # Observation vector for the electric field
p = np.array([1, 0, 0])   # Polarisation vector

'''
All of the definitions below are the unit vectors defined in equation(s) (8), and the variety of (real and complex) 
field components. This seems the most automated way of doing it, and resembles the method used in the scipy.integrate 
tutorial (https://docs.scipy.org/doc/scipy/tutorial/integrate.html).
'''

def em_par(ϕ, θ): return np.array([cos(ψ(θ))*cos(ϕ), cos(ψ(θ))*sin(ϕ), sin(ψ(θ))])
def e0_par(ϕ, θ): return np.array([cos(θ)*cos(ϕ), cos(θ)*sin(ϕ), sin(θ)])
def e_perp(ϕ): return np.array([-sin(ϕ), cos(ϕ), 0])
def sm(ϕ, θ): return np.array([sin(ψ(θ))*cos(ϕ), sin(ψ(θ))*sin(ϕ), -cos(ψ(θ))])
def s0(ϕ, θ): return np.array([sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), -cos(θ)])

def QWP(α):
    return exp(-1j * pi / 4) * np.array([[cos(α)*cos(α) + 1j*sin(α)*sin(α), (1 - 1j)*sin(α)*cos(α), 0],
                                         [(1 - 1j)*sin(α)*cos(α), sin(α)*sin(α) + 1j*cos(α)*cos(α), 0],
                                         [0, 0, 0]])

def HWP(β):
    return exp(-1j * pi / 2) * np.array([[cos(β)*cos(β) - sin(β)*sin(β), 2*cos(β)*sin(β), 0],
                                         [2*cos(β)*sin(β), sin(β)*sin(β) - cos(β)*cos(β), 0],
                                         [0, 0, 0]])

def PerturbTerm(ϕ, θ, q):
    return -1j * k * np.dot(q, sm(ϕ, θ))

def Ex(ϕ, θ): 
    return np.dot(np.dot(p, em_par(ϕ, θ)) * e0_par(ϕ, θ) \
           - np.dot(p, e_perp(ϕ)) * e_perp(ϕ), np.array([1,0,0]))
def Ey(ϕ, θ): 
    return np.dot(np.dot(p, em_par(ϕ, θ)) * e0_par(ϕ, θ) \
           - np.dot(p, e_perp(ϕ)) * e_perp(ϕ), np.array([0,1,0]))
def Ez(ϕ, θ): 
    return np.dot(np.dot(p, em_par(ϕ, θ)) * e0_par(ϕ, θ) \
           - np.dot(p, e_perp(ϕ)) * e_perp(ϕ), np.array([0,0,1]))

def E0(ϕ, θ): 
    return (1+cos(ψ(θ))) * sqrt(cos(θ)) * sin(θ)

def expFactor(ϕ, θ, r):
    return exp(1j * k * np.dot(r, s0(ϕ, θ)))

def EfieldXRe(ϕ, θ, q, r):
    return np.real(E0(ϕ, θ) * Ex(ϕ, θ) * expFactor(ϕ, θ, r) * PerturbTerm(ϕ, θ, q))

def EfieldXIm(ϕ, θ, q, r):
    return np.imag(E0(ϕ, θ) * Ex(ϕ, θ) * expFactor(ϕ, θ, r) * PerturbTerm(ϕ, θ, q))

def EfieldYRe(ϕ, θ, q, r):
    return np.real(E0(ϕ, θ) * Ey(ϕ, θ) * expFactor(ϕ, θ, r) * PerturbTerm(ϕ, θ, q))

def EfieldYIm(ϕ, θ, q, r):
    return np.imag(E0(ϕ, θ) * Ey(ϕ, θ) * expFactor(ϕ, θ, r) * PerturbTerm(ϕ, θ, q))

def EfieldZRe(ϕ, θ, q, r):
    return np.real(E0(ϕ, θ) * Ez(ϕ, θ) * expFactor(ϕ, θ, r) * PerturbTerm(ϕ, θ, q))

def EfieldZIm(ϕ, θ, q, r):
    return np.imag(E0(ϕ, θ) * Ez(ϕ, θ) * expFactor(ϕ, θ, r) * PerturbTerm(ϕ, θ, q))

'''
NOTE: we have had to split up each polarisation component into real and complex parts, as scipy.integrate cannot 
handle complex numbers apparently. Hence, we treat them separately and recombine in the final step of each calculation.
'''

# Generate x and y values
x_values = np.linspace(-7e-6, 7e-6, 100)
y_values = np.linspace(-7e-6, 7e-6, 100)

# Initialize an array to store the absolute values of the complex results
absolute_values = np.zeros((len(x_values), len(y_values)))
Ex_values = np.zeros((len(x_values), len(y_values)), dtype=np.complex_)
Ey_values = np.zeros((len(x_values), len(y_values)), dtype=np.complex_)
Ez_values = np.zeros((len(x_values), len(y_values)), dtype=np.complex_)

'''
This is the code for calculating a bunch of fields all at once. Loop through x and y values, 
and calculate the absolute value of the complex result.
'''

disps = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
labels = ['FirstOrder_x', 'FirstOrder_y', 'FirstOrder_z']
for l in range(len(disps)):
    q = disps[l]
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            r = np.array([x, y, 0])  # Observation vector for the electric field
            resultReX = integrate.dblquad(EfieldXRe, IntLimitLower, IntLimitUpper, 0, 2*np.pi, args=(q, r))[0]
            resultImX = integrate.dblquad(EfieldXIm, IntLimitLower, IntLimitUpper, 0, 2*np.pi, args=(q, r))[0]
            resultReY = integrate.dblquad(EfieldYRe, IntLimitLower, IntLimitUpper, 0, 2*np.pi, args=(q, r))[0]
            resultImY = integrate.dblquad(EfieldYIm, IntLimitLower, IntLimitUpper, 0, 2*np.pi, args=(q, r))[0]
            resultReZ = integrate.dblquad(EfieldZRe, IntLimitLower, IntLimitUpper, 0, 2*np.pi, args=(q, r))[0]
            resultImZ = integrate.dblquad(EfieldZIm, IntLimitLower, IntLimitUpper, 0, 2*np.pi, args=(q, r))[0]
            result_complex_x = resultReX + 1j * resultImX
            result_complex_y = resultReY + 1j * resultImY
            result_complex_z = resultReZ + 1j * resultImZ
            absolute_values[i, j] = np.abs(result_complex_x)**2 + np.abs(result_complex_y)**2 + np.abs(result_complex_z)**2
            Ex_values[i, j] = result_complex_x
            Ey_values[i, j] = result_complex_y
            Ez_values[i, j] = result_complex_z
            
    np.savetxt('dipolefields\dpfield_pol(x)_q0({})_M(25)_res(100)_Intensity.csv'.format(labels[l]), absolute_values, delimiter=",")
    np.savetxt('dipolefields\dpfield_pol(x)_q0({})_M(25)_res(100)_Ex.csv'.format(labels[l]), Ex_values, delimiter=",")
    np.savetxt('dipolefields\dpfield_pol(x)_q0({})_M(25)_res(100)_Ey.csv'.format(labels[l]), Ey_values, delimiter=",")
    np.savetxt('dipolefields\dpfield_pol(x)_q0({})_M(25)_res(100)_Ez.csv'.format(labels[l]), Ez_values, delimiter=",")


# # Calculating the LP Modes

# This cell creates the transverse profiles of the LP modes. An aperture is used to _cut_ the profile off at the radius of the multimode fiber used in our experimental setup. Additionally, the angle $\theta$ is used to specify a rotation angle for all LP modes, as we know that the fiber modes are perfectly _flat_ in the lab frame.

# In[ ]:


# Generate x and y values
x_values = np.linspace(-7e-6, 7e-6, 100)
y_values = np.linspace(-7e-6, 7e-6, 100)

r = 4e-6   # Radius of the input fiber to the demux.
NA = 0.3     # Numerical aperture of the input fiber to the demux.

wz = 2 * λ / (NA * np.pi) * sqrt(1 + (r * NA)**2 * np.pi / (4 * λ))

offset = 14 # Introduce a rotation angle for the fiber modes, in degrees.


# In[ ]:


'''
First let us define some key parameters, and initialize key variables. In particular, we define (a) the offset angle
of each LP mode, and (b) the spatial amplitude distribution for each LP mode.
'''

θ_0 = (np.pi / 180) * offset
θ_a = (np.pi / 180) * offset
θ_b = (np.pi / 180) * offset

# Define a set of empty arrays.

LP01_field  = np.zeros((len(x_values), len(y_values)), dtype=np.complex_)
LP11a_field = np.zeros((len(x_values), len(y_values)), dtype=np.complex_)
LP11b_field = np.zeros((len(x_values), len(y_values)), dtype=np.complex_)

# These are the LP modes expressed in cartesian coordinates, and can be re-written in terms of the Hermite-Gaussian modes.

def LP01(x, y):  return np.exp(-(x*cos(θ_0) - y*sin(θ_0))**2 / wz**2) * np.exp(-(x*sin(θ_0) + y*cos(θ_0))**2 / wz**2)
def LP11b(x, y): return np.exp(-(x*cos(θ_a) - y*sin(θ_a))**2 / wz**2) * np.exp(-(x*sin(θ_a) + y*cos(θ_a))**2 / wz**2) * (2*sqrt(2)/wz)*(x*cos(θ_a) - y*sin(θ_a))
def LP11a(x, y): return np.exp(-(x*cos(θ_b) - y*sin(θ_b))**2 / wz**2) * np.exp(-(x*sin(θ_b) + y*cos(θ_b))**2 / wz**2) * (2*sqrt(2)/wz)*(x*sin(θ_b) + y*cos(θ_b))

# This works to "cut-off" the modes at the radius corresponding to the size of our multimode fiber.

for i, x in enumerate(x_values):
    for j, y in enumerate(y_values):
        if math.sqrt(x**2+y**2) <= r:
            LP01_field[i, j]  = LP01(x, y)
            LP11a_field[i, j] = LP11a(x, y)
            LP11b_field[i, j] = LP11b(x, y)


# # Overlap & Sensitivity Calculations

# The cell below contains a function which calculates the spatial overlap integral defined:
# 
# $\begin{equation}
#     \hspace{45mm} \eta = \frac{\big|\int \textbf{E}_{\mathrm{scat}}^{*} \cdot \textbf{E}_{\mathrm{fiber}} ~ \mathrm{d}A\big|}{\sqrt{\int\big|\textbf{E}_{\mathrm{scat}}\big|^{2}~\mathrm{d}A~\times~\int\big|\textbf{E}_{\mathrm{fiber}}\big|^{2}~\mathrm{d}A}} ~ .   
# \end{equation}$
# 
# By taking combination of displacement vectors and LP modes, we can therefore build up a transfer matrix for the system. As shown below, the coupling into the LP$_{01}$ mode is dominant compared to that into the LP$_{11\mathrm{a}}$ and LP$_{11\mathrm{b}}$ mode(s). Hence, we decide to normalize the contributions to each channel. The resultant plotted quantity is then a kind of _relative spectral intensity_, which can be used to predict prospective de-multiplexing performance.

# In[ ]:


'''
This is a function which takes the x- and y-polarised components of two electric fields, and calculates the
corresponding (normalized) spatial overlap integral. It's used repeatedly in the next cell.
'''

def overlap(mode_one_x, mode_one_y, mode_one_z, mode_two_x, mode_two_y, mode_two_z):

    numer_one = np.multiply(np.conj(mode_one_x), mode_two_x)             # E1* x E2 (x)
    
    numer_two = np.multiply(np.conj(mode_one_y), mode_two_y)             # E1* x E2 (y)
    
    numer_three = np.multiply(np.conj(mode_one_z), mode_two_z)           # E1* x E2 (y)
    
    numer_tot = np.sum(numer_one + numer_two + numer_three)              # 'Integrate' the sum

    denom_one = np.sum(np.abs(mode_one_x)**2 + np.abs(mode_one_y)**2 + np.abs(mode_one_z)**2)    # S |E1|^2 dxdy

    denom_two = np.sum(np.abs(mode_two_x)**2 + np.abs(mode_two_y)**2 + np.abs(mode_two_z)**2)    # S |E2|^2 dxdy

    denom_tot = sqrt(denom_one * denom_two)                              # sqrt(S |E1|^2 dxdy S |E2|^2 dxdy)

    overlap = np.abs(numer_tot) / denom_tot
    return np.abs(overlap)

'''
As an easy to check, to see if everything's been defined correctly, we can calculate the orthogonality of the LP modes
calculated in the cell above.
'''

print('Overlap of LP01 and LP11a: {}'.format(overlap(LP01_field, LP01_field, LP01_field, LP11a_field, LP11a_field, LP11a_field)))
print('Overlap of LP01 and LP11b: {}'.format(overlap(LP01_field, LP01_field, LP01_field, LP11b_field, LP11b_field, LP11b_field)))
print('Overlap of LP11a and LP11b: {}'.format(overlap(LP11a_field, LP11a_field, LP11a_field, LP11b_field, LP11b_field, LP11b_field)))

print('\n ------------------ (Check Normalization) ------------------ \n')

print('Overlap of LP01 and LP01: {}'.format(overlap(LP01_field, LP01_field, LP01_field, LP01_field, LP01_field, LP01_field)))
print('Overlap of LP11a and LP11a: {}'.format(overlap(LP11a_field, LP11a_field, LP11a_field, LP11a_field, LP11a_field, LP11a_field)))
print('Overlap of LP11b and LP11b: {}'.format(overlap(LP11b_field, LP11b_field, LP11b_field, LP11b_field, LP11b_field, LP11b_field)))


# ### Coupling Coefficient(s)

# Here, we calculate the coupling coefficient into each LP mode of the input fiber for a particular polarization axis, as the nanoparticle is displaced by an amount $q_{0}\hat{\boldsymbol{e}}_{q}$ for $q=x,y$, and $z$.

# In[ ]:


pol = 'x'                                                            # Polarization axis of the nanoparticle.
disps = ['FirstOrder_x', 'FirstOrder_y', 'FirstOrder_z']             # Displacements to be considered, of the form q_(0)e_(q)
path  = 'dipolefields\dpfield_pol({})_q0({})_M(25)_res(100)_{}.csv'  # Path to the scattered field.

modes = [LP01_field, LP11a_field, LP11b_field]                       # Fiber modes to be considered.
labels = ['LP01', 'LP11a', 'LP11b']

long_pos = 0.7 * pi / 2 * (1 / k)  
plocal_osc = 1j * k * long_pos
local_osc = 100 * np.ones((100, 100)) * np.exp(plocal_osc)


# In[ ]:


coupling_coeffs = np.zeros((len(modes), len(disps)))


# In[ ]:


'''
If you're interested in polarisation filtering, just set the desired pol. component(s) to "np.zeros((50, 50))"
'''

for i, m in enumerate(modes): # For each fiber mode ...
    
    # Define the polarisation components of the scattered field.
    
    Ex = np.flipud(np.loadtxt(path.format(pol, '0', 'Ex'), delimiter=",", dtype=np.complex_).transpose())
    Ey = np.flipud(np.loadtxt(path.format(pol, '0', 'Ey'), delimiter=",", dtype=np.complex_).transpose())
    Ez = np.flipud(np.loadtxt(path.format(pol, '0', 'Ez'), delimiter=",", dtype=np.complex_).transpose())
    
    # This works to "cut-off" the modes at the radius corresponding to the size of our multimode fiber.
    
    for w, x in enumerate(x_values):
        for k, y in enumerate(y_values):
            if math.sqrt(x**2 + y**2) >= r:
                Ex[w, k] = 0
                Ey[w, k] = 0
                Ez[w, k] = 0
                local_osc[w, k] = 0
    
    print('Overlap with {} mode for stationary dipole: {}'.format(labels[i], overlap(Ex, Ey, Ez, m, cos(15 * pi/180) * m, -sin(8 * pi/180) * m)))
    print('Overlap with {} mode for reference field: {}'.format(labels[i], overlap(local_osc, 0, 0, m, cos(15 * pi/180) * m, -sin(8 * pi/180) * m)))
    
# ---------------------------------------------------------------------------------------------------------
    
    for j, q in enumerate(disps): # For nanoparticle displacement along each spatial axis ...
        
        # Define the polarisation components of the scattered field.
        
        Ex = np.flipud(np.loadtxt(path.format(pol, q, 'Ex'), delimiter=",", dtype=np.complex_).transpose())
        Ey = np.flipud(np.loadtxt(path.format(pol, q, 'Ey'), delimiter=",", dtype=np.complex_).transpose())
        Ez = np.flipud(np.loadtxt(path.format(pol, q, 'Ez'), delimiter=",", dtype=np.complex_).transpose())

        for w, x in enumerate(x_values):
            for k, y in enumerate(y_values):
                if math.sqrt(x**2 + y**2) >= r:
                    Ex[w, k] = 0
                    Ey[w, k] = 0
                    Ez[w, k] = 0
                    
        print('Overlap with {} mode for {}: {}'.format(labels[i], q, overlap(Ex, Ey, Ez, m, cos(5 * pi/180) * m, -sin(5 * pi/180) * m)))
        coupling_coeffs[i, j] = overlap(Ex, Ey, Ez, m, cos(5 * pi/180) * m, -sin(5 * pi/180) * m)
    print('-----------------------')

