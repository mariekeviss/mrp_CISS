#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Green's function transport calculations for a tight binding model of 6-helicene 
that includes nearest-neighbor hoppings, Spin-Orbit coupling and the Breit 
interaction. 

@author: mariekevisscher
"""

import numpy as np
import matplotlib.pyplot as plt
from N_order_integrator_only_functions import integrate_n_flipaxes

global Vss, Vsp_sig, Vpp_pi, Vpp_sig, e_sig, e_pi, Vso, N_orbs, e_s

"""Slater-Koster parameters"""
alpha = 0.74e-3
Vss = -7.92
Vsp_sig = 8.08
Vpp_pi = -3.44 #eV
Vpp_sig = 7.09 #eV
e_sig, e_s = -18, -18
e_pi = -10.5
Vso = 6e-3
Norm0 = 15.1

N_orbs = 4

"""Helicene parameters"""
a0 = 1.4e-10
N_hex = 6
pitch = 3.6e-10/a0

#distance between nearest neighbors
R_I = 1
R_M = 2
R_A = np.sqrt(7)

#angles between nearest neighbors
Phi1 = 60*np.pi/180
Phi2 = 19.1*np.pi/180

"""Pauli matrices"""
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]])  


"""Functions involving the geometry of the molecule"""
def loc_vec(phi):
    """Generates the normal vectors of the local coordinate system at a certain site
    Input:
        site (integer): site of the atom
        phi (scalar): angle of the site
    Returns:
        n_vec: array consisting of the 0 vector + nx, ny, nz vectors"""
    n_x = np.array([np.cos(phi), np.sin(phi), 0])
    n_y = np.array([-np.sin(phi), np.cos(phi), 0])
    n_z = np.array([0,0,1])
    
    n_vec = np.array([np.zeros(3), n_x, n_y, n_z])
    return n_vec

def n_vec(vec, R_ij, o):
    """Generates some more vectors that are used in the determination of the tunneling elements
    Input:
        vec (array): normal vector in the local coordinate system
        R_ij (array): points from i to j
        o (integer): indicates which orbital (s, px, py, pz)
    Returns:
        parallel/perpendicular parts of the normal vector"""
    if o == 0:
        n_par = R_ij/np.linalg.norm(R_ij)
        n_perp = np.zeros(3)
    else:
        n_par = np.dot(R_ij, vec)/(np.linalg.norm(R_ij)**2)*R_ij
        n_perp = vec - n_par
    return n_par, n_perp

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
    """Calculates the angle of the nth atom, based on the angles between nearest
    neighbors and which hexagon the atom belongs to.
    Input:
        N_hex : Either integer or an array of integers 
        n_atom: indicates whether the atom belongs to the inner, middle or outer helix
    Returns:
        phit_tot: either a scalar or an array containing the angles of each atom"""
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

def generate_links():
    """Returns an array with the indices of nearest neighbor pairs"""
    N_inner = N_hex + 1
    N_B, N_C = N_hex+1, N_hex
    
    links = []

    for i in range(0,N_inner-1):
        links.append((i,i+1)) #inner helix
    
    for i in range(N_hex):
        #print(i, N_hex+1+i)
        links.append((i, N_inner + i)) #between inner and outer helix
        links.append((N_inner + i, N_inner + N_B + i)) #outer helix
        links.append((N_inner + N_B + i, N_inner + N_B + N_C +i)) #outer helix
        links.append((N_inner + N_B + N_C + i, N_inner + i+1)) #outer helix
    links.append((N_hex, N_hex+N_B)) #link between final atoms
    
    return links

def create_molecule(N_hex, pitch):
    """Generates an array containing the positions of the atoms in the molecule
    and an array containing the angles of the atoms
    Returns:
        helix: the positions of the atoms. The array is 'sorted' by type of atom 
        of the helix, so first all the atoms of type A, then of type B etc."""
    #positions of the atoms
    A_pos = (Atom_pos(np.linspace(0,N_hex-1,N_hex),0,pitch)) #inner helix
    B_pos = (Atom_pos(np.linspace(0,N_hex-1,N_hex),1,pitch)) #outer
    C_pos = (Atom_pos(np.linspace(0,N_hex-1,N_hex),2,pitch)) #outer
    D_pos = (Atom_pos(np.linspace(0,N_hex-1,N_hex),3,pitch)) #outer
    Fin_pos = (Add_final_atoms(N_hex-1,pitch)) #A and B
    N_A = np.shape(A_pos)[1]+1 #this accounts for the fact that Fin_pos also has an atom of type A/B
    N_B = np.shape(B_pos)[1]+1
    
    #combining all the positions in one array
    inner_helix = np.insert(A_pos, np.shape(A_pos)[1], Fin_pos[:,0], axis = 1)
    outer_helix = np.insert(B_pos, np.shape(B_pos)[1], Fin_pos[:,1], axis = 1)
    outer_helix = np.concatenate((outer_helix, C_pos, D_pos), axis = 1)
    helix = np.concatenate((inner_helix, outer_helix), axis = 1)
    
    N_atoms = np.shape(helix)[1]
    
    #obtaining the angle phi for each position
    phi_A = Angle_site(np.linspace(0,N_hex,N_hex+1), 0)
    phi_B = Angle_site(np.linspace(0,N_hex,N_hex+1), 1)
    phi_C = Angle_site(np.linspace(0,N_hex-1,N_hex), 2)
    phi_D = Angle_site(np.linspace(0,N_hex-1,N_hex), 3)
    phi_tot = np.concatenate((phi_A, phi_B, phi_C, phi_D))
    
    return helix, phi_tot, N_atoms, N_A, N_B

"""Functions to create the hamiltonian"""
def get_ham_onsite(N_atoms): #works :)
    """The onsite matrix elements for the s, px, py, pz orbitals
    Returns:
        ham_onsite: (N_atoms x 2 x N_atoms x 2) sized array
        ham_onsite_res: (2*N_atoms x 2*N_atoms) sized array"""
    ham_onsite = np.zeros((N_orbs,2,N_orbs,2), dtype = 'complex128')
    ham_onsite_tot = np.zeros((N_orbs*N_atoms, 2, N_orbs*N_atoms, 2), dtype = 'complex128')
    
    #self energies
    ham_onsite[0, [0,1], 0, [0,1]] = e_s #s-orbital
    ham_onsite[1, [0,1], 1, [0,1]] = e_sig #px-oribtal
    ham_onsite[2, [0,1], 2, [0,1]] = e_sig #py-orbital
    ham_onsite[3, [0,1], 3, [0,1]] = e_pi #pz-orbital
    
    for i in range(N_atoms):
        ham_onsite_tot[N_orbs*i:N_orbs*i+N_orbs, :, N_orbs*i:N_orbs*i+N_orbs, :] = ham_onsite
        
        #projection of the Pauli matrices
        phi = helix_angles[i]
        sx_proj = np.cos(phi)*sx + np.sin(phi)*sy
        sy_proj = -np.sin(phi)*sx + np.cos(phi)*sy
        sz_proj = sz
    
        #spin orbit coupling
        
        ham_onsite_tot[N_orbs*i+1, :, N_orbs*i+2, :] = -1j*sz_proj*Vso
        ham_onsite_tot[N_orbs*i+1, :, N_orbs*i+3, :] = 1j*sy_proj*Vso
        ham_onsite_tot[N_orbs*i+2, :, N_orbs*i+3, :] = -1j*sx_proj*Vso
        ham_onsite_tot[N_orbs*i+2, :, N_orbs*i+1, :] = 1j*sz_proj*Vso
        ham_onsite_tot[N_orbs*i+3, :, N_orbs*i+1, :] = -1j*sy_proj*Vso
        ham_onsite_tot[N_orbs*i+3, :, N_orbs*i+2, :] = 1j*sx_proj*Vso

    ham_onsite_res = np.reshape(ham_onsite_tot, (2*N_atoms*N_orbs, 2*N_atoms*N_orbs))
    return ham_onsite, ham_onsite_res


def get_ham_transfer(N_hex, N_atoms): #works :)
    """The transfer matrix elements.
    Returns:
        ham_T: (N_atoms x 2 x N_atoms x 2) sized array
        ham_T_res: (2*N_atoms x 2*N_atoms) sized array"""
    ham_T = np.zeros((N_orbs*N_atoms, 2, N_orbs*N_atoms, 2), dtype = 'complex128') #3 orbitals per site, 2 possible spin values
    
    for l in helix_links:
        n_1 = loc_vec(helix_angles[l[0]])
        n_2 = loc_vec(helix_angles[l[1]])
        R_12 = helix_pos[:,l[1]] - helix_pos[:, l[0]]
        for i in range(N_orbs):
            #0 = s, 1 = px, 2 = py, 3 = pz
            n_1_par, n_1_perp = n_vec(n_1[i], R_12, i)
            for j in range(N_orbs):        
                n_2_par, n_2_perp = n_vec(n_2[j], -R_12, j)
                if i == 0:
                    if j == 0:
                        hij = np.dot(n_1_par, n_2_par)*Vss
                    else:
                        hij = np.dot(n_1_par, n_2_par)*Vsp_sig 
                else:
                    if j == 0:
                        hij = np.dot(n_1_par, n_2_par)*Vsp_sig 
                    else:
                        hij = np.dot(n_1_par, n_2_par)*Vpp_sig + np.dot(n_1_perp, n_2_perp)*Vpp_pi 
                ham_T[N_orbs*l[0]+i, [0,1], N_orbs*l[1]+j, [0,1]] = hij
                ham_T[N_orbs*l[1]+j, [0,1], N_orbs*l[0]+i, [0,1]] = hij
                        
    ham_T_res = np.reshape(ham_T, (2*N_atoms*N_orbs, 2*N_atoms*N_orbs))
    return ham_T, ham_T_res

def get_Breit_arr_strength(Norm0):
    """Creates an array containing the Breit interactions between s,px, py, pz electrons"""
    Breit_arr = np.zeros((4,4), dtype = 'complex128')
    Breit_arr[0, 2], Breit_arr[0,3] = 2j*np.pi/3, -2j*np.pi/3
    Breit_arr[1,2], Breit_arr[1,3] = -2j*np.pi/5, 1j*np.pi/5
    Breit_arr[2,3] = -1j*np.pi/3
    
    Breit_arr += np.transpose(np.conj(Breit_arr))
    
    return Breit_arr/Norm0

def get_Breit_arr_spin(phi):
    """Creates an array containing the spin part of the Breit interactions between s,px, py, pz electrons"""
    spin_arr = np.zeros((4,2,4,2), dtype = 'complex128')
    spin_arr[0,:,2,:], spin_arr[0,:,3,:] = sz, -np.sin(phi)*sx + np.cos(phi)*sy
    spin_arr[1,:,2,:], spin_arr[1,:,3,:] = sz, -np.sin(phi)*sx + np.cos(phi)*sy
    spin_arr[2,:,3,:] = np.cos(phi)*sx + np.sin(phi)*sy
    spin_arr[2,:,0,:], spin_arr[3,:,0,:] = sz, -np.sin(phi)*sx + np.cos(phi)*sy
    spin_arr[2,:,1,:], spin_arr[3,:,1,:] = sz, -np.sin(phi)*sx + np.cos(phi)*sy
    spin_arr[3,:,2,:] = np.cos(phi)*sx + np.sin(phi)*sy
    
    return spin_arr

def get_V_Breit(N_sites, links, pos, angles, Norm0, alpha):
    """To get the matrix elements used in the breit hamiltonian
    returns:
        2*V_1: array with shape (N_sites, N_sites, 4, 4, 2, 2)
        4*V_2: array with shape (N_sites, N_sites, 4, 4, 2, 2)
        the factors 2 and 4 come from the expression of the Breit interaction"""
    
    #V1[i, j, mu, mu', s, s'] = <i mu s, j nu sigma | B | i mu' s', j nu sigma>
    #V2[i, j, mu, mu', s, s'] = <i mu sigma, j nu s | B | i mu' sigma, j nu s'>
    V_1 = np.zeros((N_sites, N_sites, 4, 4, 2, 2), dtype = 'complex128')
    V_2 = np.zeros((N_sites, N_sites, 4, 4, 2, 2), dtype = 'complex128')
    VB = get_Breit_arr_strength(Norm0)
    for l in links:
        R_01 = np.linalg.norm(pos[:,l[1]] - pos[:,l[0]])
        Vspin_0 = get_Breit_arr_spin(angles[l[0]])
        Vspin_1 = get_Breit_arr_spin(angles[l[1]])
        for i in range(4):
            for j in range(4): 
                V_1[l[0], l[1], i, j, [0,1], [0,1]] = -alpha*VB[i, j]*Vspin_0[i,[0,1],j,[0,1]]/(R_01**2)
                V_1[l[0], l[1], i, j, [0,1], [1,0]] = -alpha*VB[i, j]*Vspin_0[i,[0,1],j,[1,0]]/(R_01**2)
                V_1[l[0], l[1], j, i, [0,1], [0,1]] = -alpha*VB[j, i]*Vspin_0[j,[0,1],i,[0,1]]/(R_01**2)
                V_1[l[0], l[1], j, i, [1,0], [0,1]] = -alpha*VB[j, i]*Vspin_0[j,[1,0],i,[0,1]]/(R_01**2)
                
                V_1[l[1], l[0], i, j, [0,1], [0,1]] = -alpha*VB[i, j]*Vspin_1[i,[0,1],j,[0,1]]/(R_01**2)
                V_1[l[1], l[0], i, j, [0,1], [1,0]] = -alpha*VB[i, j]*Vspin_1[i,[0,1],j,[1,0]]/(R_01**2)
                V_1[l[1], l[0], j, i, [0,1], [0,1]] = -alpha*VB[j, i]*Vspin_1[j,[0,1],i,[0,1]]/(R_01**2)
                V_1[l[1], l[0], j, i, [1,0], [0,1]] = -alpha*VB[j, i]*Vspin_1[j,[1,0],i,[0,1]]/(R_01**2)
                
                V_2[l[0], l[1], i, j, [0,1], [0,1]] = -alpha*VB[i, j]*Vspin_1[i,[0,1],j,[0,1]]/(R_01**2)
                V_2[l[0], l[1], i, j, [0,1], [1,0]] = -alpha*VB[i, j]*Vspin_1[i,[0,1],j,[1,0]]/(R_01**2)
                V_2[l[0], l[1], j, i, [0,1], [0,1]] = -alpha*VB[j, i]*Vspin_1[j,[0,1],i,[0,1]]/(R_01**2)
                V_2[l[0], l[1], j, i, [1,0], [0,1]] = -alpha*VB[j, i]*Vspin_1[j,[1,0],i,[0,1]]/(R_01**2)
                
                V_2[l[1], l[0], i, j, [0,1], [0,1]] = -alpha*VB[i, j]*Vspin_0[i,[0,1],j,[0,1]]/(R_01**2)
                V_2[l[1], l[0], i, j, [0,1], [1,0]] = -alpha*VB[i, j]*Vspin_0[i,[0,1],j,[1,0]]/(R_01**2)
                V_2[l[1], l[0], j, i, [0,1], [0,1]] = -alpha*VB[j, i]*Vspin_0[j,[0,1],i,[0,1]]/(R_01**2)
                V_2[l[1], l[0], j, i, [1,0], [0,1]] = -alpha*VB[j, i]*Vspin_0[j,[1,0],i,[0,1]]/(R_01**2)
                      
    return 2*V_1, 4*V_2

def get_U_Breit(N_sites, occ): #seems to work :)
    """To get the Breit matrix used in the Green's function
    Input:
        occ: occupation numbers. occ should have shape (N_sites, 4, N_sites, 4, 2, 2)
    Returns:
        U1_arr + U2_arr + U3_arr + U4_arr: array with shape (N_sites, 4, 2, N_sites, 4, 2)"""
    U1_occ = occ[np.arange(0, N_sites), :, np.arange(0, N_sites), :, :, :]
    U1_arr_tot0 = np.moveaxis(V_Breit_1*U1_occ, 0,1)
    
    for i in range(5): #i think this is correct
        U1_arr_tot0 = np.sum(U1_arr_tot0, 1)
        
    U1_arr = np.zeros([N_sites,4,2,N_sites,4,2], dtype = 'complex128')
    for i in range(4):
        U1_arr[np.arange(0,N_sites), i, 0, np.arange(0,N_sites), i, 0]= -U1_arr_tot0
        U1_arr[np.arange(0,N_sites), i, 1, np.arange(0,N_sites), i, 1]= -U1_arr_tot0
        
    U2_occ = occ[np.arange(0,N_sites),:,np.arange(0,N_sites),:,:,:][:, np.arange(0,4),np.arange(0,4),:,:][:,:,[0,1],[0,1]]
    U2_occ = np.sum(np.sum(U2_occ, axis = -1), axis = -1)
    U2_arr_tot = np.moveaxis(V_Breit_1,-2,0)
    U2_arr_tot = np.moveaxis(U2_arr_tot, -1, 1)
    U2_arr_tot = np.moveaxis(U2_arr_tot, -1, 2)
    U2_arr_tot = np.moveaxis(U2_arr_tot, -1, 3)
    U2_arr_tot = np.sum(U2_arr_tot*U2_occ, axis = -1)
    U2_arr_tot = np.moveaxis(U2_arr_tot, -1,0)
    U2_arr_tot = np.moveaxis(U2_arr_tot, 1,-1)
    U2_arr_tot = np.moveaxis(U2_arr_tot, 1,2)
    U2_arr = np.zeros([N_sites,4,2,N_sites,4,2], dtype = 'complex128')
    U2_arr[np.arange(0, N_sites), :, :, np.arange(0, N_sites), :, :] = -U2_arr_tot
    U2_arr[np.arange(0, N_sites), :, :, np.arange(0, N_sites), :, :] = -U2_arr_tot
    
    #contributions from V2
    U3_occ = occ[np.arange(0, N_sites), :, np.arange(0,N_sites),:,:,:][:,:,:,[0,1],[0,1]]
    U3_occ = np.sum(U3_occ, axis = -1)
    U3_arr_tot0 = np.moveaxis(V_Breit_2, -2,0)
    U3_arr_tot0 = np.moveaxis(U3_arr_tot0, -1, 1)
    U3_arr_tot0 = U3_arr_tot0*U3_occ
    U3_arr_tot0 = np.swapaxes(U3_arr_tot0, 2,3)
    
    for i in range(3):
        U3_arr_tot0 = np.sum(U3_arr_tot0, axis = -1)
    U3_arr_tot0 = np.moveaxis(U3_arr_tot0, 0, -1)
    U3_arr_tot0 = np.moveaxis(U3_arr_tot0, 0, -2)
    U3_arr = np.zeros((N_sites, 4, 2, N_sites, 4, 2), dtype = 'complex128')
    for i in range(4):
        U3_arr[np.arange(0, N_sites), i, :, np.arange(0, N_sites), i, :] = -U3_arr_tot0
    
    U4_occ = occ[np.arange(0, N_sites), :, np.arange(0, N_sites), :, :, :][:, np.arange(4), np.arange(4), :, :]
    U4_arr_tot0 = np.moveaxis(V_Breit_2, 3,0)
    
    U4_arr_tot0 = U4_arr_tot0*U4_occ
    U4_arr_tot0 = np.moveaxis(U4_arr_tot0, -4, -1)
    for i in range(3):
        U4_arr_tot0 = np.sum(U4_arr_tot0, axis = -1)
    U4_arr_tot0 = np.moveaxis(U4_arr_tot0, 0,1)
    U4_arr = np.zeros((N_sites, 4, 2, N_sites, 4, 2), dtype = 'complex128')
    U4_arr[np.arange(0,N_sites),:,0,np.arange(0,N_sites),:,0] = -U4_arr_tot0
    U4_arr[np.arange(0,N_sites),:,1,np.arange(0,N_sites),:,1] = -U4_arr_tot0
    
    
    return U1_arr + U2_arr + U3_arr + U4_arr

"""Functions to evaluate the Green's functions and obtain the transmissions"""
def get_spin_proj(N_sites):
    """Defining spin-projection matrices for computation of spin-dependent transmission (T_uu, T_ud, ...)"""
    Spin_proj_down = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    Spin_proj_up = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')    
    Spin_proj_up[:,0,:,0] = np.eye(N_sites,N_sites, k=0)
    Spin_proj_down[:,1,:,1] = np.eye(N_sites,N_sites, k=0)
    return Spin_proj_down, Spin_proj_up

def Pauli_NN(N_atoms):
    """Defining arrays of shape (N_atomsxN_atoms) that have the Pauli matrices on the diagonal"""
    sx_NN = np.zeros((N_orbs*N_atoms, 2, N_orbs*N_atoms, 2), dtype = 'complex128')
    sy_NN = np.zeros((N_orbs*N_atoms, 2, N_orbs*N_atoms, 2), dtype = 'complex128')
    sz_NN = np.zeros((N_orbs*N_atoms, 2, N_orbs*N_atoms, 2), dtype = 'complex128')
    for i in range(N_atoms*N_orbs):
        sx_NN[i, :, i, :] = sx
        sy_NN[i, :, i, :] = sy
        sz_NN[i, :, i, :] = sz
    sx_NN, sy_NN, sz_NN = np.reshape(sx_NN, (2*N_atoms*N_orbs, 2*N_atoms*N_orbs)), np.reshape(sy_NN, (2*N_atoms*N_orbs, 2*N_atoms*N_orbs)), np.reshape(sz_NN, (2*N_atoms*N_orbs, 2*N_atoms*N_orbs))
    
    return sx_NN, sy_NN, sz_NN

def GF_ret(E, U_breit_res):
    GF_res = np.linalg.inv(E*Id_res - ham_tot - U_breit_res - Sigma_tot_res)
    
    return GF_res

def FD_dist(E, mu, beta=1):
    return 1/(np.exp((E - mu)*beta) + 1)


def get_GF_tot(N_sites, e_arr, mu_L, mu_R, Occ, beta = 1):
    G_less_store = np.zeros([len(e_arr), N_sites*4, 2, N_sites*4, 2], dtype = 'complex128')
    Occ_res = np.reshape(Occ, (N_sites, 4, 2, N_sites, 4, 2))
    Occ_res = np.moveaxis(Occ_res, 2,-2)
    print(np.shape(Occ_res))
    U_array_Breit0 = get_U_Breit(N_sites, Occ_res)
    U_array_Breit0_res = np.reshape(U_array_Breit0, [2*N_sites*4,2*N_sites*4])

    
    for i in range(0,len(e_arr)):
        E = e_arr[i]
        
        Sigma_less_res = 1j*(Gamma_L_res*FD_dist(E,mu_L,beta) + Gamma_R_res*FD_dist(E,mu_R,beta))
        G_ret_res = GF_ret(E, U_array_Breit0_res)
        G_adv_res = np.conj(np.transpose(G_ret_res))
        
        G_less_res = np.dot(G_ret_res, np.dot(Sigma_less_res, G_adv_res))
        G_less = np.reshape(G_less_res, [N_sites*4,2,N_sites*4,2])
        G_less_store[i] = G_less
    return G_less_store

"""self consistent calculation of the occupation numbers"""
def get_Occ(N_sites, G_less_store, e_arr):
    delta_E = np.abs(e_arr[1] - e_arr[0])
    Occ_arr = np.zeros([4*N_sites, 2, 4*N_sites, 2], dtype = 'complex128')
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
    Occ_store = np.zeros([len(V_arr),N_sites,4,N_sites,4,2,2], dtype = 'complex128')
    for k in range(0,len(V_arr)):
        mu_L = mu0 + V_arr[k]/2; mu_R = mu0 - V_arr[k]/2
        if k == 0:
            Occ_res = np.diag(0.5*np.ones([2*N_sites*4], dtype = 'complex128'))
            Occ_i = np.reshape(Occ_res, [4*N_sites,2,4*N_sites,2])
            Occ_i = get_Occ_SC(N_sites, e_arr, mu_L, mu_R, Occ_i, n_it0, beta=beta)
        if k!=0:
            Occ_i = get_Occ_SC(N_sites, e_arr, mu_L, mu_R, Occ_i, n_it, beta=beta)
 
        Occ_store[k] = np.moveaxis(np.reshape(Occ_i, (N_sites, 4, 2, N_sites, 4, 2)),2,-2)
    return (Occ_store)

def get_transmissions(N_sites, e_arr_p, mu0, Volt, beta, V_Breit_1, V_Breit_2, n_occ):
    Transm_arr_down = np.zeros(len(e_arr_p), dtype = 'complex128')
    Transm_arr_up = np.zeros(len(e_arr_p), dtype = 'complex128')
    Transm_arr_du = np.zeros(len(e_arr_p), dtype = 'complex128')
    Transm_arr_ud = np.zeros(len(e_arr_p), dtype = 'complex128')
    Pol_arr = np.zeros(len(e_arr_p), dtype = 'complex128')
    
    U_array_Breit0 = get_U_Breit(N_atoms, n_occ)
    U_array_Breit0_res = np.reshape(U_array_Breit0, [4*2*N_sites,4*2*N_sites])
    
    for i in range(len(e_arr_p)):
        G_ret_res = GF_ret(e_arr_p[i], U_array_Breit0_res)
        G_adv_res = np.transpose(np.conj(G_ret_res))
        
        Ti_dd = np.trace(np.dot(Gamma_L_res, np.dot(Spin_res_down, np.dot(G_adv_res, np.dot(Gamma_R_res, np.dot(Spin_res_down, G_ret_res))))))
        Ti_ud = np.trace(np.dot(Gamma_L_res, np.dot(Spin_res_down, np.dot(G_adv_res, np.dot(Gamma_R_res, np.dot(Spin_res_up, G_ret_res))))))
        Ti_uu = np.trace(np.dot(Gamma_L_res, np.dot(Spin_res_up, np.dot(G_adv_res, np.dot(Gamma_R_res, np.dot(Spin_res_up, G_ret_res))))))
        Ti_du = np.trace(np.dot(Gamma_L_res, np.dot(Spin_res_up, np.dot(G_adv_res, np.dot(Gamma_R_res, np.dot(Spin_res_down, G_ret_res))))))

        Transm_arr_down[i] = Ti_dd
        Transm_arr_up[i] = Ti_uu
        Transm_arr_du[i] = Ti_du
        Transm_arr_ud[i] = Ti_ud

        Pol_arr[i] = np.trace(np.dot(Gamma_L_res, np.dot(G_adv_res, np.dot(Gamma_R_res, np.dot(sz_NN, G_ret_res)))))/np.trace(np.dot(Gamma_L_res, np.dot(G_adv_res, np.dot(Gamma_R_res, G_ret_res))))
        
    return Transm_arr_down, Transm_arr_up, Transm_arr_du, Transm_arr_ud, Pol_arr

helix_links = generate_links()
helix_pos, helix_angles, N_atoms, N_A, N_B = create_molecule(N_hex, pitch)
sx_NN, sy_NN, sz_NN = Pauli_NN(N_atoms)

#Inverse temperature & average chemical potential of the leads
beta = 2.5; mu0 = 0. 
#Energy array over which is integrated to obtain the occupations self-consistently from the lesser Green's function
e_arr = np.linspace(-5,5,1000) 

#Array storing the bias-voltages
V_arr = np.array([0]) 
#coupling matrices
Spin_proj_down, Spin_proj_up = get_spin_proj(N_orbs*N_atoms)
Spin_res_down = np.reshape(Spin_proj_down, newshape = [2*N_orbs*N_atoms,2*N_orbs*N_atoms])
Spin_res_up = np.reshape(Spin_proj_up, newshape = [2*N_orbs*N_atoms,2*N_orbs*N_atoms])

Gamma_L = np.zeros((N_atoms, N_orbs, 2, N_atoms, N_orbs, 2), dtype = 'complex128')
Gamma_R = np.zeros((N_atoms, N_orbs, 2, N_atoms, N_orbs, 2), dtype = 'complex128')

Gamma_L[0, 3, [0,1], 0, 3, [0,1]] = 1 #pz orbital of first site inner helix
Gamma_L[N_A, 3, [0,1], N_A, 3, [0,1]] = 1 #pz orbital of first site outer helix
Gamma_R[N_A - 1, 3, [0,1], N_A - 1, 3, [0,1]] = 1 #pz orbital of last site inner helix
Gamma_R[N_A + N_B - 1, 3, [0,1], N_A + N_B - 1, 3, [0,1]] = 1 #pz orbital of last site outer helix

Gamma_L_res = np.reshape(Gamma_L, (2*N_orbs*N_atoms, 2*N_orbs*N_atoms))
Gamma_R_res = np.reshape(Gamma_R, (2*N_orbs*N_atoms, 2*N_orbs*N_atoms))

Sigma_L_res = -1j/2*Gamma_L_res
Sigma_R_res = -1j/2*Gamma_R_res
Sigma_tot_res = Sigma_L_res + Sigma_R_res

Id_res =  np.eye(N_atoms*2*N_orbs,N_atoms*2*N_orbs,k=0,dtype = 'complex128')
ham_onsite, ham_onsite_res = get_ham_onsite(N_atoms)

ham_T, ham_T_res = get_ham_transfer(N_hex, N_atoms)
e_arr = np.linspace(-5,5, 1000) 
ham_tot = ham_T_res + ham_onsite_res

V_Breit_1, V_Breit_2 = get_V_Breit(N_atoms, helix_links, helix_pos, helix_angles, Norm0, alpha)
Occ_store = Occ_sweep_V(N_atoms, mu0, V_arr, e_arr, n_it0 = 20, n_it = 20, beta = beta)
U_Breit = get_U_Breit(N_atoms, Occ_store[0])
U_Breit_res = np.reshape(U_Breit, (2*4*N_atoms, 2*4*N_atoms))

Transm_down, Transm_up, Transm_arr_du, Transm_arr_ud, Pol_arr = get_transmissions(N_atoms, e_arr, mu0, V_arr[0], beta, V_Breit_1, V_Breit_2, Occ_store[0])
np.savetxt('22_10_04_Ttot_Breit_morepoints.txt', np.real(Transm_down + Transm_arr_ud + Transm_up + Transm_arr_du))
np.savetxt('22_10_04_Spinpol_Breit_morepoints.txt', np.real(Pol_arr))

"""plots"""
plt.figure(figsize = (6,6))
plt.plot(helix_pos[0,:], helix_pos[1,:],'o')
plt.plot(helix_pos[0,0], helix_pos[1,0],'o', color = 'red', label = 'left lead inner helix')
plt.plot(helix_pos[0,7], helix_pos[1,7],'o', color = 'black', label = 'left lead outer helix')
plt.plot(helix_pos[0,6], helix_pos[1,6],'o', color = 'red', label = 'right lead inner helix')
plt.plot(helix_pos[0,13], helix_pos[1,13],'o', color = 'black', label = 'right lead outer helix')
#plt.plot(helix_pos[0,6], helix_pos[1,6],'o', color = 'red')
plt.legend()
plt.title('Molecule')
plt.show()

plt.figure() #plot of the connected sites
for l in helix_links:
    plt.plot([helix_pos[0,l[0]], helix_pos[0,l[1]]], [helix_pos[1, l[0]], helix_pos[1, l[1]]], label = str(l[0])+','+str(l[1]))
plt.plot(helix_pos[0,0], helix_pos[1,0],'o', color = 'red')
plt.plot(helix_pos[0,7], helix_pos[1,7],'o', color = 'red')
plt.legend()
plt.show()
    

plt.figure()
plt.imshow(np.real(Id_res));plt.colorbar();plt.title('Identity matrix');plt.show()
plt.figure()
plt.imshow(np.real(ham_onsite_res));plt.colorbar();plt.title('Real part of the onsite hamiltonian');plt.show()
plt.figure()
plt.imshow(np.imag(ham_onsite_res));plt.colorbar();plt.title('Imaginary part of the onsite hamiltonian');plt.show()
plt.figure()
plt.imshow(np.real(ham_T_res));plt.colorbar();plt.title('Real part of the transfer hamiltonian');plt.show()
plt.figure()
plt.imshow(np.imag(ham_T_res));plt.colorbar();plt.title('Imaginary part of the transfer hamiltonian');plt.show()
plt.figure()
plt.imshow(np.real(U_Breit_res));plt.colorbar();plt.title('Real part of the Breit hamiltonian');plt.show()
plt.figure()
plt.imshow(np.imag(U_Breit_res));plt.colorbar();plt.title('Imaginary part of the Breit hamiltonian');plt.show()
plt.figure()
plt.imshow(np.real(Gamma_L_res));plt.colorbar();plt.title('Couplings to left lead');plt.show()
plt.figure()
plt.imshow(np.real(Gamma_R_res));plt.colorbar();plt.title('Couplings to right lead');plt.show()

plt.figure()
plt.plot(e_arr, Transm_down + Transm_arr_ud + Transm_up + Transm_arr_du)
plt.title('Transmission')
plt.show()

plt.figure()
plt.plot(e_arr, np.abs(Transm_up + Transm_arr_du))
plt.title('Transmission up')
plt.show()

plt.figure()
plt.plot(e_arr, np.abs(Transm_down + Transm_arr_ud))
plt.title('Transmission down')
plt.show()

plt.figure()
plt.plot(e_arr, np.real(Pol_arr*10e3))
plt.xlabel('energy (eV)')
plt.ylabel('P (*10e-3)')
plt.title("Spin polarization")
plt.show()
