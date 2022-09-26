#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:02:39 2022

@author: mariekevisscher
"""
import numpy as np
import matplotlib.pyplot as plt

global Vss, Vsp_sig, Vpp_pi, Vpp_sig, e_sig, e_pi, Vso, N_orbs, e_s

"""Slater-Koster parameters"""
Vss = -7.92
Vsp_sig = 8.08
Vpp_pi = -3.44 #eV
Vpp_sig = 7.09 #eV
e_sig, e_s = -18, -18
e_pi = -10.5
Vso = 6e-3

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
    """returns an array with the indices of nearest neighbor pairs"""
    N_inner = N_hex + 1
    N_B, N_C, N_D = N_hex+1, N_hex, N_hex
    
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
    """The onsite matrix elements for the s, px, py, pz orbitals"""
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
    """The transfer matrix elements"""
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



"""Functions to evaluate the Green's functions and obtain the transmissions"""
def get_spin_proj(N_sites):
    #Defining spin-projection matrices for computation of spin-dependent transmission (T_uu, T_ud, ...)
    Spin_proj_down = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    Spin_proj_up = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')    
    Spin_proj_up[:,0,:,0] = np.eye(N_sites,N_sites, k=0)
    Spin_proj_down[:,1,:,1] = np.eye(N_sites,N_sites, k=0)
    return Spin_proj_down, Spin_proj_up

def Pauli_NN(N_atoms):
    sx_NN = np.zeros((N_orbs*N_atoms, 2, N_orbs*N_atoms, 2), dtype = 'complex128')
    sy_NN = np.zeros((N_orbs*N_atoms, 2, N_orbs*N_atoms, 2), dtype = 'complex128')
    sz_NN = np.zeros((N_orbs*N_atoms, 2, N_orbs*N_atoms, 2), dtype = 'complex128')
    for i in range(N_atoms*N_orbs):
        sx_NN[i, :, i, :] = sx
        sy_NN[i, :, i, :] = sy
        sz_NN[i, :, i, :] = sz
    sx_NN, sy_NN, sz_NN = np.reshape(sx_NN, (2*N_atoms*N_orbs, 2*N_atoms*N_orbs)), np.reshape(sy_NN, (2*N_atoms*N_orbs, 2*N_atoms*N_orbs)), np.reshape(sz_NN, (2*N_atoms*N_orbs, 2*N_atoms*N_orbs))
    
    return sx_NN, sy_NN, sz_NN

def GF_ret(E):
    GF_res = np.linalg.inv(E*Id_res - ham_tot - Sigma_tot_res)
    return GF_res  

def get_transmissions(e_arr_p):
    Transm_arr_down = np.zeros(len(e_arr_p), dtype = 'complex128')
    Transm_arr_up = np.zeros(len(e_arr_p), dtype = 'complex128')
    Transm_arr_du = np.zeros(len(e_arr_p), dtype = 'complex128')
    Transm_arr_ud = np.zeros(len(e_arr_p), dtype = 'complex128')
    Pol_arr = np.zeros(len(e_arr_p), dtype = 'complex128')
    
    for i in range(len(e_arr_p)):
        G_ret_res = GF_ret(e_arr_p[i])
        G_adv_res = np.transpose(np.conj(G_ret_res))
        
        Ti_dd = np.trace(np.dot(Gamma_L_res, np.dot(Spin_res_down, np.dot(G_adv_res, np.dot(Gamma_R_res, np.dot(Spin_res_down, G_ret_res))))))
        Ti_ud = np.trace(np.dot(Gamma_L_res, np.dot(Spin_res_down, np.dot(G_adv_res, np.dot(Gamma_R_res, np.dot(Spin_res_up, G_ret_res))))))
        Ti_uu = np.trace(np.dot(Gamma_L_res, np.dot(Spin_res_up, np.dot(G_adv_res, np.dot(Gamma_R_res, np.dot(Spin_res_up, G_ret_res))))))
        Ti_du = np.trace(np.dot(Gamma_L_res, np.dot(Spin_res_up, np.dot(G_adv_res, np.dot(Gamma_R_res, np.dot(Spin_res_down, G_ret_res))))))

        Transm_arr_down[i] = Ti_dd
        Transm_arr_up[i] = Ti_uu
        Transm_arr_du[i] = Ti_du
        Transm_arr_ud[i] = Ti_ud
        #Pol_arr[i] = (Ti_uu + Ti_du - Ti_ud - Ti_dd)/(Ti_uu + Ti_du + Ti_ud + Ti_dd)

        Pol_arr[i] = np.trace(np.dot(Gamma_L_res, np.dot(G_adv_res, np.dot(Gamma_R_res, np.dot(sz_NN, G_ret_res)))))/np.trace(np.dot(Gamma_L_res, np.dot(G_adv_res, np.dot(Gamma_R_res, G_ret_res))))
        
    return Transm_arr_down, Transm_arr_up, Transm_arr_du, Transm_arr_ud, Pol_arr

helix_links = generate_links()
helix_pos, helix_angles, N_atoms, N_A, N_B = create_molecule(N_hex, pitch)
sx_NN, sy_NN, sz_NN = Pauli_NN(N_atoms)


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
e_arr = np.linspace(-5,5, 500) 
ham_tot = ham_T_res + ham_onsite_res

Transm_down, Transm_up, Transm_arr_du, Transm_arr_ud, Pol_arr = get_transmissions(e_arr)

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
    #plt.plot(, helix_pos[1, l[1]], 'o', label = str(l[0]))
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
plt.imshow(np.real(Gamma_L_res));plt.colorbar();plt.title('Couplings to left lead');plt.show()
plt.figure()
plt.imshow(np.real(Gamma_R_res));plt.colorbar();plt.title('Couplings to right lead');plt.show()

plt.figure()
plt.plot(e_arr, np.abs(Transm_down+ Transm_arr_ud))
plt.title('Transmission down')
plt.show()

plt.figure()
plt.plot(e_arr, np.abs(Transm_up + Transm_arr_du))
plt.title('Transmission up')
plt.show()

plt.figure()
plt.plot(e_arr, np.real(Pol_arr*10e3))
plt.xlabel('energy (eV)')
plt.ylabel('P (*10e-3)')
plt.title("Spin polarization")
plt.show()
