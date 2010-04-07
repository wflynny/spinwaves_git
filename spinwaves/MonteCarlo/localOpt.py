'''
Created on Jul 16, 2009

@author: tsarvey

Updated on Jul 30, 2009
@author: wflynn
'''
import copy
#from ctypes import c_float, c_int
from numpy import cos, sin, arctan, arccos, pi,arcsin
#from CSim import passAtoms, passMatrices, loadLib
from simple import readFile
import spinwaves.spinwavecalc.readfiles as rf
import numpy as np
import sympy as sp
from scipy.sparse import bsr_matrix
from scipy.optimize import fmin_l_bfgs_b
from timeit import time

def gen_Jij(atom_list,jmats):
    """ Creates a scipy bsr sparse array of the Jij interaction matrices"""
    N_atoms = len(atom_list)
    jij_values = []
    jij_columns = []
    jij_rowIndex = []
    zeroval = np.zeros((3,3))
    
    # Counts total number of interactions: needed for row indexing
    num_inters = 0
    # Scan through atom_list
    
    nbrs_ints = []  
    for i in range(N_atoms):
        nbrs_ints = atom_list[i].interactions
        nbrs_ints.sort()

        # Now we have a sorted list of (nbr,intr) tuples from lowest neighbor to highest neighbor
        # Scan through interactions
        if len(nbrs_ints)>0:
            for j in range(len(nbrs_ints)):
                nbr = nbrs_ints[j][0]
                intr = nbrs_ints[j][1]
    
                #Get an interaction matrix
                curr_mat = jmats[intr].tolist()
                curr_mat = np.array(curr_mat, dtype=np.float64)
    
                # Values   = current matrix
                # Columns  = the current neighbor
                # RowIndex = total number of interactions 
                jij_values.append(curr_mat)
                jij_columns.append(nbr)
                if j == 0:
                    jij_rowIndex.append(num_inters)
                
                # Increase number of total interactions
                num_inters = num_inters + 1
        else:
            jij_values.append(zeroval)
            jij_columns.append(0)
            jij_rowIndex.append(num_inters)
            num_inters = num_inters + 1
    # Add dummy index to rowIndex
    jij_rowIndex.append(len(jij_values))

    # Convert to numpy arrays
    jij_values = np.array(jij_values)
    jij_columns = np.array(jij_columns)
    jij_rowIndex = np.array(jij_rowIndex)
    
    print jij_values
    print jij_values.shape[1:]
    print N_atoms + 1
    print len(jij_rowIndex)
    print jij_columns
    print jij_rowIndex
    
    # Create Sparse Array
    jij = bsr_matrix( (jij_values,jij_columns,jij_rowIndex), shape=(3*N_atoms,3*N_atoms) ).todense()

    return jij

def gen_anisotropy(atom_list):
    """ From an atom_list, this method returns a numpy array of the atoms' anisotropy terms"""
    N_atoms = len(atom_list)
    anis_vect = []
    for i in range(N_atoms):
        anis_vect.append(atom_list[i].anisotropy[0])
        anis_vect.append(atom_list[i].anisotropy[1])
        anis_vect.append(atom_list[i].anisotropy[2])
    anis_vect = np.array(anis_vect)
    return anis_vect

def opt_aux(atom_list, jmats, spins, tol = 1.0e-10):
    """This function separates the functionality of the optimizer from the
    files.  This method assumes that jmats is in the correct order. 
    ie. jnums looks like [0,1,2,3...]"""
    N_atoms = len(atom_list)
    # Generate the Jij and anisotropy arrays
    Jij = gen_Jij(atom_list,jmats)
    anis = gen_anisotropy(atom_list)
    # Get the spin magnitudes from the atoms in atom list
    spin_mags = []
    for atom in atom_list:
        spin_mags.append(atom.spinMag)
    spin_mags = np.array(spin_mags)
    
    print "got here!"
    
    # hamiltonian method
    def hamiltonian(p, Jij = None, spinMags = None, anis = None):
        """ Computes the hamiltonian given a list a thetas and phis"""
        # Thetas are the first half, phis the second half
        theta = p[:len(p)//2]
        phi = p[len(p)//2:]

        # Sx,Sy,Sz
        Sx = spinMags*sin(theta)*cos(phi)
        Sy = spinMags*sin(theta)*sin(phi)
        Sz = spinMags*cos(theta)
#        print 'local opt spins'
#        print Sx[0], Sy[0], Sz[0]

        # Array of spin vectors for each atom. Reshape it. Calculate hamiltonian with it and return the hamiltonian. 
        Sij = np.array([Sx,Sy,Sz])
        Sij = Sij.T.reshape(1,3*len(p)//2)[0].T
        
        SijT = Sij.T
        #res1 = SijT * Sij
        res1 = SijT*Jij
        Hij = np.dot(res1,Sij).flat[0]
        Ham = - Hij - np.dot(anis, Sij**2)

        return Ham 
    
    # derivative of the hamiltonian
    def deriv(p, Jij = None, spinMags = None, anis = None):
        """ Computes the derivative of the hamiltonian with respect to each theta and then each phi"""
        # Thetas are the first half, phis the second half
        half = len(p)/2
        theta = p[:half]
        phi = p[half:]        

        # Sx,Sy,Sz
        Sx = spinMags*sin(theta)*cos(phi)
        Sy = spinMags*sin(theta)*sin(phi)
        Sz = spinMags*cos(theta)

        # dSx/dtheta,dSy/dtheta,dSz/dtheta
        Sxt = spinMags*cos(theta)*cos(phi)
        Syt = spinMags*cos(theta)*sin(phi)
        Szt = -spinMags*sin(theta)
        # dSx/dphi,dSy/dphi,dSz/dphi
        Sxp = -spinMags*sin(theta)*sin(phi)
        Syp = spinMags*sin(theta)*cos(phi)
        Szp = 0*cos(theta)
        
        # Makes an array of the derivatives with respect to theta, then another with respect to phi
        # (dSx1/dtheta1,dSy1/dtheta1,dSz1/dtheta1
        # (                                      dSx2/dtheta2,dSy2/dtheta2,dSz2/dtheta2
        # (                                                                             ...
        # Similarly for phis
        # Then we multiply this by Jij which yields a half by 3*half array. We then multiply this by
        # the Sij vector which finally yields a half by 1 array. Thus
        # dSijt * Jij * Sij -> [dE/dtheta1, dE/dtheta2, dE/dtheta3, ...]
        # and similarly for phi
        dSijt = np.zeros((half,3*half))
        dSijp = np.zeros((half,3*half))
        dSijt[range(half),range(0,3*half,3)]=Sxt
        dSijt[range(half),range(1,3*half,3)]=Syt
        dSijt[range(half),range(2,3*half,3)]=Szt
        dSijp[range(half),range(0,3*half,3)]=Sxp
        dSijp[range(half),range(1,3*half,3)]=Syp
        dSijp[range(half),range(2,3*half,3)]=Szp
        
        # Standard Sij spin vector we've been using
        Sij = np.array([Sx,Sy,Sz])
        Sij = Sij.T.reshape(1,3*len(p)//2)[0].T

        # Calculate a hamiltonian for both theta and phi
        res1t = np.dot(dSijt * Jij, Sij)
        res2t = np.dot(Sij.T, Jij * dSijt.T)
        Hijt = res1t + res2t
        Hamt = - Hijt - np.matrix(np.dot(2*anis.T*Sij,dSijt.T))
        Hamt = Hamt.T

        res1p = np.dot(dSijp * Jij, Sij)
        res2p = np.dot(Sij.T, Jij * dSijp.T)
        Hijp = res1p + res2p
        Hamp = - Hijp - np.matrix(np.dot(2*anis.T*Sij,dSijp.T))
        Hamp = Hamp.T
        
        # Concatenate the two and return the result
        result = np.concatenate((np.array(Hamt),np.array(Hamp)))
        return result.T

    # populate initial p list
    # returns a numpy array of all the thetas followed by all the phis
    thetas = []
    phis = []
    if len(spins)!= N_atoms: raise Exception('poop')
    for i in range(N_atoms):
        sx = spins[i][0]
        sy = spins[i][1]
        sz = spins[i][2]
        s  = atom_list[i].spinMag

        theta = arccos(sz/s)
        phi   = np.arctan2(sy,sx)
        
        thetas.append(theta)
        phis.append(phi)
        
#        print 'initial spins'
#        print s*sin(theta)*cos(phi), s*sin(theta)*sin(phi), s*cos(theta)
    p0 = np.array(thetas+phis)

    # define the limits parameter list
    limits = []
    for i in range(len(p0)):
        if i < len(p0)//2:#theta
            limits.append((0,pi))
        else:#phi
            limits.append((-pi,pi))
    
    print "about to call function"
    
    st = time.time()
    # call minimizing function
    m = fmin_l_bfgs_b(hamiltonian, p0, fprime = deriv, args = (Jij, spin_mags, anis), pgtol=tol, bounds = limits)
    print time.time()-st, "seconds"
    print "function done"
    # grab returned parameters
    # thetas are the first half of the list, phis are the second half
    pout=m[0]
    theta=pout[0:len(pout)/2]
    phi=pout[len(pout)/2::]
    # recreate sx,sy,sz's
    sx=spin_mags*sin(theta)*cos(phi)
    sy=spin_mags*sin(theta)*sin(phi)
    sz=spin_mags*cos(theta)
    
    print "local optimization done"

    return np.array([sx,sy,sz]).T   


if __name__ == '__main__':
    #print optimizeSpins('C:\\export.txt', 'C:\\spins.txt', 'C:\\spins2.txt')
    interfile = 'C:/Documents and Settings/wflynn/Desktop/yang_montecarlo.txt'
    spinfile  = 'c:/Documents and Settings/wflynn/Desktop/spins.txt'
    atoms, mats = readFile(interfile)
    #interfile='c:/montecarlo_ferro.txt'
    #spinfile='c:/Spins_ferro.txt'
    #readfile=interfile
    tol = 1.0e-8
