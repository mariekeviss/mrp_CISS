#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Green's function transport calculations for a tight binding model of a helical toy 
model that includes nearest-neighbor hoppings, Spin-Orbit coupling and the Breit 
interaction. 

@author: mariekevisscher
"""
import numpy as np
import matplotlib.pyplot as plt
from N_order_integrator_only_functions import integrate_n_flipaxes

"""Constants"""
global Vpi_pp, Vsig_pp, gamma, Vso, e_ps
alpha = 0.74e-3
VB_xz = -0.139j*alpha
VB_yz = 0.0556j*alpha
Vpi_pp = -3.44 #eV
Vsig_pp = 7.09 #eV
gamma = 1
Vso = 0.006
e_ps = 7.5

"""Helicene parameters"""
a0 = 1.4e-10
N_hex = 23
pitch = 3.6e-10/a0

#distance between nearest neighbors
R_I = 1
R_M = 2
R_A = np.sqrt(7)

#angles between nearest neighbors
Phi1 = 60*np.pi/180
Phi2 = 19.1*np.pi/180

#Pauli matrices
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]])  

"""Functions to create the molecule"""
def Spiral_map(n_atom):
    """
    Maps number of atom to helix-number
    0 => inner helix
    1 => middle helix
    2,3 => outer helix
    """
    Spiral_arr = np.array([0,1,2,2])
    return Spiral_arr[n_atom]

def Angle_site(N_hex, n_atom):
    del_phis = np.array([0,0,Phi2,Phi1-Phi2])
    phi_tot = N_hex*Phi1 + del_phis[n_atom]
    return phi_tot

def Atom_pos(N_hex, n_atom, pitch):
    """
    Returns the position of atoms at the N_hex hexagon at the n_atom
    Input:
        N_hex : Either integer or an array of integers 
    Returns: 
        Pos : Where Pos[i,j] gives the i-th coordinate of the j-th atom
    """
    Spiral_N = Spiral_map(n_atom)
    Radius_dict = np.array([R_I,R_M,R_A])
    Radius_n = Radius_dict[Spiral_N]
    Angle_tot = Angle_site(N_hex, n_atom)
    Pos = np.array([Radius_n*np.cos(Angle_tot), Radius_n*np.sin(Angle_tot), pitch/(2*np.pi)*Angle_tot])
    return Pos

def Add_final_atoms(N_fin, pitch):
    """
    Adds positions of the 2 final atoms (since Atom_pos only generates the first 4 sites of the final hexagon)
    to the final hexagaon.
    """
    Pos1 = Atom_pos(N_fin + 1, 0, pitch); Pos2 = Atom_pos(N_fin + 1, 1, pitch)
    Pos_tot = np.zeros([3,2])
    Pos_tot[:,0] = Pos1; Pos_tot[:,1] = Pos2
    return Pos_tot


def create_molecule(N_hex, pitch):
    """Generates an array containing the positions of the atoms in the molecule
    and an array containing the angles of the atoms
    Returns:
        helix: the positions of the atoms. The array is 'sorted' by type of atom 
        of the helix, so first all the atoms of type A, then of type B etc."""
    #positions of the atoms
    A_pos = (Atom_pos(np.linspace(0,N_hex-1,N_hex),0,pitch)) #inner helix
    Fin_pos = (Add_final_atoms(N_hex-1,pitch)) #A and B
    N_A = np.shape(A_pos)[1]+1 #this accounts for the fact that Fin_pos also has an atom of type A/B
    
    #combining all the positions in one array
    inner_helix = np.insert(A_pos, np.shape(A_pos)[1], Fin_pos[:,0], axis = 1)
    
    
    #obtaining the angle phi for each position
    phi_A = Angle_site(np.linspace(0,N_hex,N_hex+1), 0)
    
    return inner_helix, N_A, phi_A

"""Functions to get the hamiltonians"""
def get_tunneling_elements():
    """Calculates the tunneling parameter"""
    txz = gamma/(1+gamma**2)*0.5*Vpi_pp - gamma/((1+gamma**2)*(1+gamma**2*np.pi**2/9))*(np.sqrt(3)/2-np.pi/3)*(np.sqrt(3)/2-gamma*np.pi/3)*(Vsig_pp-Vpi_pp)
    tyz = gamma/np.sqrt(1+gamma**2)*0.5*Vpi_pp + gamma/((1+gamma**2)*(1+np.pi**2*gamma**2/9))*(np.sqrt(3)/2-np.pi/3)*(Vsig_pp - Vpi_pp)
    t0 = 1/(1+gamma**2)*(1/gamma**2+1/2)*Vpi_pp + gamma**2*(np.sqrt(3)/2-np.pi/3)**2/((1+gamma**2)*(1+np.pi**2*gamma**2/9))
    return txz, tyz, t0

def get_SO_strength():
    """Calculates the tunneling parameter that includes spin orbit interaction"""
    tso = Vso/(2*e_ps)*(gamma/(1+gamma**2)**(3/2)*Vpi_pp/2 - gamma/((1+gamma**2)**(3/2)*(1+np.pi**2*gamma**2/9))*(np.sqrt(3)/2-np.pi/3)*(np.sqrt(3)/2-gamma*np.pi/3)*(Vsig_pp-Vpi_pp)\
                  +gamma/np.sqrt(1+gamma**2)*np.sqrt(3)/2*Vpi_pp + gamma/((1+gamma**2)*(1+np.pi**2*gamma**2/9))*(3/2-np.pi/np.sqrt(3))*(Vsig_pp-Vpi_pp))
    return tso

def get_Breit_ham(N_sites, txz, tyz, angles, VB_xz, VB_yz):
    """Creates the matrices used for the Breit Hamiltonian"""
    V_Breit_1 = np.zeros((N_sites, N_sites, N_sites, 2,2), dtype = 'complex128')
    V_Breit_2 = np.zeros((N_sites, N_sites, N_sites, 2,2), dtype = 'complex128')
    
    for i in range(N_sites-1):
        sx_proj = sx*np.cos(angles[i]) + sy*np.sin(angles[i])
        sy_proj = -sx*np.sin(angles[i]) + sy*np.cos(angles[i])
        sx_proj2 = sx*np.cos(angles[i+1]) + sy*np.sin(angles[i+1])
        sy_proj2 = -sx*np.sin(angles[i+1]) + sy*np.cos(angles[i+1])
        
        V_Breit_1[i, i+1, i+1, :, :] = (VB_xz*sy_proj*txz + VB_yz*sx_proj*tyz)/e_ps
        V_Breit_1[i+1, i, i+1, :, :] = (np.conj(VB_xz)*sy_proj*txz + np.conj(VB_yz)*sx_proj*tyz)/e_ps
        V_Breit_1[i+1, i, i, :, :] = (-VB_xz*sy_proj2*txz - VB_yz*sx_proj2*-tyz)/e_ps
        V_Breit_1[i, i+1, i, :, :] = (np.conj(-VB_xz)*sy_proj2*txz + np.conj(-VB_yz)*sx_proj2*-tyz)/e_ps
        
        V_Breit_2[i, i+1, i+1, :, :] = (VB_xz*sy_proj2*txz + VB_yz*sx_proj2*tyz)/e_ps
        V_Breit_2[i+1, i, i+1, :, :] = (np.conj(VB_xz)*sy_proj2*txz + np.conj(VB_yz)*sx_proj2*tyz)/e_ps
        V_Breit_2[i+1, i, i, :, :] = (-VB_xz*sy_proj*txz - VB_yz*sx_proj*-tyz)/e_ps
        V_Breit_2[i, i+1, i, :, :] = (np.conj(-VB_xz)*sy_proj*txz + np.conj(-VB_yz)*sx_proj*-tyz)/e_ps
    
    return -2*V_Breit_1, -4*V_Breit_2

def get_U_Breit(N_sites, n_occ):
    """Matrices containing the Breit terms as used in the Green's function"""
    U_Breit_1 = np.zeros((N_sites, 2, N_sites, 2), dtype = 'complex128')
    U_Breit_2 = np.zeros((N_sites, 2, N_sites, 2), dtype = 'complex128')
    
    for i in range(0, N_sites):
        if i == 0:
            U_Breit_1[i, [0,1], i, [0,1]] += -V_Breit_1[i+1, i, i, [0,1], [1,0]]*n_occ[i+1, [0,1], i, [1,0]]
            U_Breit_1[i, [0,1], i, [0,1]] += -V_Breit_1[i, i+1, i, [1,0], [0,1]]*n_occ[i, [1,0], i+1, [0,1]]
            U_Breit_1[i, [0,1], i+1, [1,0]] = -V_Breit_1[i, i+1, i+1, [0,1], [1,0]]*n_occ[i+1, [0,1], i+1, [0,1]]
            U_Breit_1[i, [0,1], i+1, [1,0]] += -V_Breit_1[i, i+1, i, [0,1], [1,0]]*n_occ[i, [1,0], i, [1,0]]
            
            U_Breit_2[i, [0,1], i, [0,1]] += V_Breit_2[i+1, i, i, [0,1], [1,0]]*n_occ[i+1,[0,1],i,[1,0]]
            U_Breit_2[i, [0,1], i, [0,1]] += V_Breit_2[i, i+1, i, [1,0], [0,1]]*n_occ[i, [1,0],i+1,[0,1]]
            U_Breit_2[i, [0,1], i+1, [1,0]] = -V_Breit_2[i, i+1, i+1, [0,1], [1,0]]*n_occ[i+1, [0,1], i+1, [0,1]]
            U_Breit_2[i, [0,1], i+1, [1,0]] += -V_Breit_2[i, i+1, i, [0,1], [1,0]]*n_occ[i, [1,0], i, [1,0]]
            

        elif i == N_sites-1:
            U_Breit_1[i, [0,1], i, [0,1]] = -V_Breit_1[i-1, i, i, [0,1], [1,0]]*n_occ[i-1, [0,1], i, [1,0]]
            U_Breit_1[i, [0,1], i, [0,1]] += -V_Breit_1[i, i-1, i, [1,0], [0,1]]*n_occ[i, [1,0], i-1, [0,1]]
            U_Breit_1[i, [0,1], i-1, [1,0]] = -V_Breit_1[i, i-1, i-1, [0,1], [1,0]]*n_occ[i-1, [0,1], i-1, [0,1]]
            U_Breit_1[i, [0,1], i-1, [1,0]] += -V_Breit_1[i, i-1, i, [0,1], [1,0]]*n_occ[i, [1,0], i, [1,0]]
            
            
            U_Breit_2[i, [0,1], i, [0,1]] = V_Breit_2[i-1, i, i, [0,1], [1,0]]*n_occ[i-1,[0,1],i,[1,0]]
            U_Breit_2[i, [0,1], i, [0,1]] += V_Breit_2[i, i-1, i, [1,0], [0,1]]*n_occ[i, [1,0],i-1,[0,1]]
            U_Breit_2[i, [0,1], i-1, [1,0]] = -V_Breit_2[i, i-1, i-1, [0,1], [1,0]]*n_occ[i-1, [0,1], i-1, [0,1]]
            U_Breit_2[i, [0,1], i-1, [1,0]] += -V_Breit_2[i, i-1, i, [0,1], [1,0]]*n_occ[i, [1,0], i, [1,0]]
            
        else:
            U_Breit_1[i, [0,1], i, [0,1]] = -V_Breit_1[i-1, i, i, [0,1], [1,0]]*n_occ[i-1, [0,1], i, [1,0]]
            U_Breit_1[i, [0,1], i, [0,1]] += -V_Breit_1[i, i-1, i, [1,0], [0,1]]*n_occ[i, [1,0], i-1, [0,1]]
            U_Breit_1[i, [0,1], i, [0,1]] += -V_Breit_1[i+1, i, i, [0,1], [1,0]]*n_occ[i+1, [0,1], i, [1,0]]
            U_Breit_1[i, [0,1], i, [0,1]] += -V_Breit_1[i, i+1, i, [1,0], [0,1]]*n_occ[i, [1,0], i+1, [0,1]]
            
            U_Breit_1[i, [0,1], i+1, [1,0]] = -V_Breit_1[i, i+1, i+1, [0,1], [1,0]]*n_occ[i+1, [0,1], i+1, [0,1]]
            U_Breit_1[i, [0,1], i+1, [1,0]] += -V_Breit_1[i, i+1, i, [0,1], [1,0]]*n_occ[i, [1,0], i, [1,0]]
            U_Breit_1[i, [0,1], i-1, [1,0]] = -V_Breit_1[i, i-1, i-1, [0,1], [1,0]]*n_occ[i-1, [0,1], i-1, [0,1]]
            U_Breit_1[i, [0,1], i-1, [1,0]] += -V_Breit_1[i, i-1, i, [0,1], [1,0]]*n_occ[i, [1,0], i, [1,0]]
        
            U_Breit_2[i, [0,1], i, [0,1]] = V_Breit_2[i-1, i, i, [0,1], [1,0]]*n_occ[i-1,[0,1],i,[1,0]]
            U_Breit_2[i, [0,1], i, [0,1]] += V_Breit_2[i, i-1, i, [1,0], [0,1]]*n_occ[i, [1,0],i-1,[0,1]]
            U_Breit_2[i, [0,1], i, [0,1]] += V_Breit_2[i+1, i, i, [0,1], [1,0]]*n_occ[i+1,[0,1],i,[1,0]]
            U_Breit_2[i, [0,1], i, [0,1]] += V_Breit_2[i, i+1, i, [1,0], [0,1]]*n_occ[i, [1,0],i+1,[0,1]]
            
            U_Breit_2[i, [0,1], i+1, [1,0]] = -V_Breit_2[i, i+1, i+1, [0,1], [1,0]]*n_occ[i+1, [0,1], i+1, [0,1]]
            U_Breit_2[i, [0,1], i+1, [1,0]] += -V_Breit_2[i, i+1, i, [0,1], [1,0]]*n_occ[i, [1,0], i, [1,0]]
            U_Breit_2[i, [0,1], i-1, [1,0]] = -V_Breit_2[i, i-1, i-1, [0,1], [1,0]]*n_occ[i-1, [0,1], i-1, [0,1]]
            U_Breit_2[i, [0,1], i-1, [1,0]] += -V_Breit_2[i, i-1, i, [0,1], [1,0]]*n_occ[i, [1,0], i, [1,0]]
            
    return U_Breit_1 + U_Breit_2


def get_single_ham(N_sites, t0, tso):
    ham = np.zeros((N_sites,2,N_sites,2), dtype = 'complex128')
    
    for i in range(N_sites-1):
        ham[i, [0,1], i+1, [0,1]] = t0
        ham[i, 0, i+1, 1] = -1j*tso*np.exp(-2j*np.pi/(N_sites-1)*(i+i+1)/2)
        ham[i, 1, i+1, 0] = -1j*tso*np.exp(+2j*np.pi/(N_sites-1)*(i+i+1)/2)
    
    ham_res = np.reshape(ham, (2*N_sites, 2*N_sites))
    ham_res = ham_res + np.transpose(np.conj(ham_res))
    
    return ham_res

"""Functions to calculate transport"""
def get_spin_proj(N_sites):
    #Defining spin-projection matrices for computation of spin-dependent transmission (T_uu, T_ud, ...)
    Spin_proj_down = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    Spin_proj_up = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    
    Spin_proj_down[:,0,:,0] = np.eye(N_sites,N_sites, k=0)
    Spin_proj_up[:,1,:,1] = np.eye(N_sites,N_sites, k=0)
    return Spin_proj_down, Spin_proj_up

def GF_ret(E, U_breit_res):
    GF_res = np.linalg.inv(E*Id_res - ham_res - U_breit_res  - Sigma_tot_res)
    
    return GF_res

def FD_dist(E, mu, beta=1):
    return 1/(np.exp((E - mu)*beta) + 1)


def get_GF_tot(N_sites, e_arr, mu_L, mu_R, Occ, beta = 1):
    G_less_store = np.zeros([len(e_arr), N_sites, 2, N_sites, 2], dtype = 'complex128')
    U_array_Breit0 = get_U_Breit(N_sites, Occ)
    U_array_Breit0_res = np.reshape(U_array_Breit0, [2*N_sites,2*N_sites])
    
    for i in range(0,len(e_arr)):
        E = e_arr[i]
        
        Sigma_less_res = 1j*(Gamma_L_res*FD_dist(E,mu_L,beta) + Gamma_R_res*FD_dist(E,mu_R,beta))
        G_ret_res = GF_ret(E, U_array_Breit0_res)
        G_adv_res = np.conj(np.transpose(G_ret_res))
        
        G_less_res = np.dot(G_ret_res, np.dot(Sigma_less_res, G_adv_res))
        G_less = np.reshape(G_less_res, [N_sites,2,N_sites,2])
        G_less_store[i] = G_less
    return G_less_store

def get_Occ(N_sites, G_less_store, e_arr):
    delta_E = np.abs(e_arr[1] - e_arr[0])
    Occ_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
    Occ_arr = 1/(2*np.pi)*integrate_n_flipaxes(-1j*G_less_store, 3, delta_E)
    return Occ_arr

def get_Occ_step(N_sites, e_arr, mu_L, mu_R, Occ, beta = 1):
    G_less_store = get_GF_tot(N_sites, e_arr, mu_L, mu_R, Occ, beta)
    Occ_arr = get_Occ(N_sites, G_less_store, e_arr)
    return Occ_arr

def get_Occ_SC(N_sites, e_arr, mu_L, mu_R, Occ_init, n_iter, beta = 1):
    for j in range(0,n_iter):
        if j==0:
            Occ_i = get_Occ_step(N_sites, e_arr, mu_L, mu_R, Occ_init, beta)
            #Occ_i = get_Occ_step(np.linspace(-20,-10,60000), mu_L, mu_R, Occ_init, beta)  + get_Occ_step(np.linspace(-10,0,60000), mu_L, mu_R, Occ_init, beta) + get_Occ_step(np.linspace(0,10,60000), mu_L, mu_R, Occ_init, beta)  + get_Occ_step(np.linspace(10,20,60000), mu_L, mu_R, Occ_init, beta)
        if j!=0:
            Occ_i = get_Occ_step(N_sites, e_arr, mu_L, mu_R, Occ_i, beta)
            #Occ_i = get_Occ_step(np.linspace(-20,-10,60000), mu_L, mu_R, Occ_i, beta)  + get_Occ_step(np.linspace(-10,0,60000), mu_L, mu_R, Occ_i, beta) + get_Occ_step(np.linspace(0,10,60000), mu_L, mu_R, Occ_i, beta)  + get_Occ_step(np.linspace(10,20,60000), mu_L, mu_R, Occ_i, beta)
    return Occ_i

def Occ_sweep_V(N_sites, mu0, V_arr, e_arr, n_it0, n_it, beta = 1):
    Occ_store = np.zeros([len(V_arr),N_sites,2,N_sites,2], dtype = 'complex128')
    for k in range(0,len(V_arr)):
        mu_L = mu0 + V_arr[k]/2; mu_R = mu0 - V_arr[k]/2
        if k == 0:
            Occ_res = np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128'))
            Occ_i = np.reshape(Occ_res, [N_sites,2,N_sites,2])
            Occ_i = get_Occ_SC(N_sites, e_arr, mu_L, mu_R, Occ_i, n_it0, beta=beta)
        if k!=0:
            Occ_i = get_Occ_SC(N_sites, e_arr, mu_L, mu_R, Occ_i, n_it, beta=beta)
        Occ_store[k] = Occ_i
    return Occ_store

def get_transmissions(N_sites, e_arr_p, mu0, Volt, beta, V_Breit_1, V_Breit_2, n_occ):        
    U_array_Breit0 = get_U_Breit(N_sites, n_occ)
    U_array_Breit0_res = np.reshape(U_array_Breit0, [2*N_sites,2*N_sites])

    Transm_arr_dd = np.zeros(len(e_arr_p), dtype = 'complex128')
    Transm_arr_uu = np.zeros(len(e_arr_p), dtype = 'complex128')
    Transm_arr_du = np.zeros(len(e_arr_p), dtype = 'complex128')
    Transm_arr_ud = np.zeros(len(e_arr_p), dtype = 'complex128')

    for i in range(len(e_arr_p)):
        G_ret_res = GF_ret(e_arr_p[i], U_array_Breit0_res)
        G_adv_res = np.transpose(np.conj(G_ret_res))

        Ti_down = np.trace(np.dot(Gamma_L_res,np.dot(Spin_res_down,np.dot(G_ret_res,np.dot(Gamma_R_res,np.dot(Spin_res_down,G_adv_res))))))
        Ti_up = np.trace(np.dot(Gamma_L_res,np.dot(Spin_res_up,np.dot(G_ret_res,np.dot(Gamma_R_res,np.dot(Spin_res_up,G_adv_res))))))
        Ti_du = np.trace(np.dot(Gamma_L_res,np.dot(Spin_res_up,np.dot(G_ret_res,np.dot(Gamma_R_res,np.dot(Spin_res_down,G_adv_res))))))
        Ti_ud = np.trace(np.dot(Gamma_L_res,np.dot(Spin_res_down,np.dot(G_ret_res,np.dot(Gamma_R_res,np.dot(Spin_res_up,G_adv_res))))))
        
        Transm_arr_dd[i] = Ti_down
        Transm_arr_uu[i] = Ti_up
        Transm_arr_du[i] = Ti_du
        Transm_arr_ud[i] = Ti_ud
        
    Transm_tot = Transm_arr_uu + Transm_arr_du + Transm_arr_ud + Transm_arr_dd
    Spin_pol = (Transm_arr_uu + Transm_arr_du - Transm_arr_ud - Transm_arr_dd)/Transm_tot
    return Transm_tot, Transm_arr_dd, Transm_arr_uu, Transm_arr_du, Transm_arr_ud, Spin_pol

"""Plots etc"""
txz, tyz, t0 = get_tunneling_elements()
tso = get_SO_strength()
inner_helix, N_sites, phi_A = create_molecule(N_hex, pitch)

#Inverse temperature & average chemical potential of the leads
beta = 2.5; mu0 = 0. 
#Energy array over which is integrated to obtain the occupations self-consistently from the lesser Green's function
e_arr = np.linspace(-3*t0,3*t0,5000) 

#Array storing the bias-voltages
V_arr = np.array([0]) 

#Coupling strengths between molecule & left lead (Gamma_down, Gamma_up = 0.75, 0.25 => Down magnetized lead, 0.25, 0.75 => Up magnetized lead)
Gamma_L0_u = 2*t0
Gamma_L0_d = 2*t0
#Coupling strengths between molecule & right lead
Gamma_R0_d = 2*t0
Gamma_R0_u = 2*t0


Spin_proj_down, Spin_proj_up = get_spin_proj(N_sites)
Spin_res_down = np.reshape(Spin_proj_down, newshape = [2*N_sites,2*N_sites])
Spin_res_up = np.reshape(Spin_proj_up, newshape = [2*N_sites,2*N_sites])
Gamma_L_res = np.zeros((2*N_sites, 2*N_sites))
Gamma_L_res[0,0], Gamma_L_res[1,1] = 2*t0, 2*t0
Gamma_R_res = np.zeros((2*N_sites, 2*N_sites))
Gamma_R_res[2*N_sites-2, 2*N_sites-2], Gamma_R_res[2*N_sites-1, 2*N_sites-1] = 2*t0, 2*t0
Sigma_L_res = -1j/2*Gamma_L_res
Sigma_R_res = -1j/2*Gamma_R_res
Sigma_tot_res = Sigma_L_res + Sigma_R_res

ham_res = get_single_ham(N_sites, t0, tso)
Id_res =  np.eye(N_sites*2,N_sites*2,k=0,dtype = 'complex128')
V_Breit_1, V_Breit_2 = get_Breit_ham(N_sites, txz, tyz, phi_A, VB_xz, VB_yz)
V_Breit_1_0, V_Breit_2_0 = get_Breit_ham(N_sites, txz, tyz, phi_A, 0, 0)


Occ_store = Occ_sweep_V(N_sites, mu0, V_arr, e_arr, n_it0 = 20, n_it = 20, beta = beta)
ham_Breit = get_U_Breit(N_sites, Occ_store[0])
Transm_tot, Transm_arr_dd, Transm_arr_uu, Transm_arr_du, Transm_arr_ud, Spin_pol = get_transmissions(N_sites, e_arr, mu0, V_arr[0], beta,  V_Breit_1_0, V_Breit_2_0, Occ_store[0])

np.savetxt('22_10_05_Ttotsinglechannel_nobreit', np.real(Transm_tot))
np.savetxt('22_10_05_Tdownsinglechannel_nobreit', np.real(Transm_arr_dd+Transm_arr_ud))
np.savetxt('22_10_05_Tupsinglechannel_nobreit', np.real(Transm_arr_du+Transm_arr_uu))

plt.figure()
plt.plot(e_arr/tso, np.abs(Transm_tot), color = 'black',label = 'With Breit interaction')
plt.legend()
plt.xlabel('E ($t_{0})$')
plt.ylabel('T')
plt.title("Total transmission of the model without and with Breit interaction")
#plt.savefig("22_09_30_transm_singlechann_Breit_compared.pdf")
plt.show()




#some sanity checks
plt.figure()
plt.imshow(np.real(Id_res));plt.colorbar();plt.title('Identity matrix');plt.show()
plt.figure()
plt.imshow(np.real(ham_res));plt.colorbar();plt.title('Real part of the hamiltonian');plt.show()
plt.figure()
plt.imshow(np.imag(ham_res));plt.colorbar();plt.title('Imaginary part of the hamiltonian');plt.show()
plt.figure()
plt.imshow(np.real(Gamma_L_res));plt.colorbar();plt.title('Couplings to left lead');plt.show()
plt.figure()
plt.imshow(np.real(Gamma_R_res));plt.colorbar();plt.title('Couplings to right lead');plt.show()
plt.figure()
plt.imshow(np.real(np.reshape(ham_Breit, (2*N_sites, 2*N_sites))));plt.colorbar();plt.title('Real part of the Breit Hamiltonian');plt.show()
plt.figure()
plt.imshow(np.imag(np.reshape(ham_Breit, (2*N_sites, 2*N_sites))));plt.colorbar();plt.title('Imaginary part of the Breit Hamiltonian');plt.show()
#the results

plt.figure()
plt.plot(e_arr/t0, np.abs(Transm_tot))
plt.title('Transmission tot')
#plt.ylim(0,2)
plt.show()


plt.figure()
plt.plot(e_arr/t0, np.abs(Transm_arr_ud+Transm_arr_dd), color = 'black', label = 'down')
plt.plot(e_arr/t0, np.abs(Transm_arr_du+Transm_arr_uu), color = 'red', label = 'up')
plt.xlabel('E ($t_{0}$')
plt.ylabel('T')
plt.legend()
plt.title('Transmission of spin up vs spin down electrons')
#plt.savefig("22_09_30_transmup_down_compared_exaggerated.pdf")
plt.show()

plt.figure()
plt.plot(e_arr/t0, np.real(Spin_pol), color = 'black')
plt.title('Spin polarization')
#plt.ylim(0,2)
plt.show()