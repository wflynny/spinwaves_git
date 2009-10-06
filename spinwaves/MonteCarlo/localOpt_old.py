'''
Created on Jul 16, 2009

@author: tsarvey

Updated on Jul 30, 2009
@author: wflynn
'''
import copy
#from ctypes import c_float, c_int
from numpy import cos, sin, arctan, arccos, pi,arcsin
from spinwaves.utilities.mpfit import mpfit
#from CSim import passAtoms, passMatrices, loadLib
#from simple import readFile
import spinwaves.spinwavecalc.readfiles as rf
import numpy as np
import sympy as sp
from scipy.sparse import bsr_matrix
from scipy.optimize import fmin_l_bfgs_b


#Populate C atom list
def optimizeSpins(interactionFile, inFile, outFile):
    """This will use the local optimizer, mpfit, to perfect the values of ground
    state spins.  'inFile is a spin file with the same format that the 
    montecarlo simulation creates.  'outFile' is the path of the file to output,
    with locally optimized spins.  It is the same format as inFile."""
    
    atoms, jMatrices = readFile(interactionFile)
    atomListPointer, nbr_pointers, mat_pointers = passAtoms(atoms)
    matPointer = passMatrices(jMatrices)
    c_lib = loadLib()
    
    def hamiltonian(p, fjac=None, y=None, err=None):
        """p is the parameters and will be formatted as:
        [theta1, phi1, theta2, phi2, ...] where theta and phi specify the
        angular coordinates of the spin."""
        #copy new spin values to C spinwave list
        for i in range(len(atoms)):
            theta = p[2*i]
            phi = p[(2*i) + 1]
            r = atoms[i].spinMag
            
            Sx = r*cos(theta)*sin(phi)
            Sy = r*sin(theta)*sin(phi)
            Sz = r*cos(phi)
            
            c_lib.setSpin(atomListPointer, c_int(i), c_float(Sx), c_float(Sy), c_float(Sz))
        
        status = 0
        result = c_lib.totalE(atomListPointer, len(atoms), matPointer)
        print float(result)
        return [status, [result]]
    
    #populate initial p list
    p0 = []
    for atom in atoms:
        x = atom.s[0]
        y = atom.s[1]
        z = atom.s[2]
        r = atom.spinMag
        theta = arctan(y/x)#-pi/2 to pi/2
        #range will now be from -pi to pi
        if(x < 0 and y < 0):
            theta -= pi
        elif(x<0):
            theta += pi
        phi = arccos(z/r)#0 to pi
        p0.append(theta)
        p0.append(phi)
        
    parbase={'value':0., 'limited':[0,0], 'limits':[0.,0.]}
    parinfo=[]
    for i in range(len(p0)):
        parinfo.append(copy.deepcopy(parbase))
    for i in range(len(p0)):
        parinfo[i]['value']=p0[i]
        if(i/2 == 0):#even index means theta
            parinfo[i]['limits'][0]= -pi
            parinfo[i]['limits'][1] = pi
            parinfo[i]['limited'][0] = 1
            parinfo[i]['limited'][1] = 1
        else:#phi
            parinfo[i]['limits'][0]= 0
            parinfo[i]['limits'][1] = pi
            parinfo[i]['limited'][0] = 1
            parinfo[i]['limited'][1] = 1
        
    
    m = mpfit(hamiltonian, p0, parinfo = parinfo)
    return m.params



def gen_Jij(atom_list,jmats):
    """ Creates a scipy bsr sparse array of the Jij interaction matrices"""
    N_atoms = len(atom_list)
    jij_values = []
    jij_columns = []
    jij_rowIndex = []   
    
    # Counts total number of interactions: needed for row indexing
    num_inters = 0
    # Scan through atom_list
    for i in range(N_atoms):
        ints_list = atom_list[i].interactions
        nbrs_list = atom_list[i].neighbors
        nbrs_ints = [ (nbrs_list[x],ints_list[x]) for x in range(len(nbrs_list)) ]
        nbrs_ints.sort()

        # Now we have a sorted list of (nbr,intr) tuples from lowest neighbor to highest neighbor
        # Scan through interactions
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
    # Add dummy index to rowIndex
    jij_rowIndex.append(len(jij_values))

    # Convert to numpy arrays
    jij_values = np.array(jij_values)
    jij_columns = np.array(jij_columns)
    jij_rowIndex = np.array(jij_rowIndex)
    
    # Create Sparse Array
    jij = bsr_matrix( (jij_values,jij_columns,jij_rowIndex), shape=(3*N_atoms,3*N_atoms) ).todense()
    
    return jij

def gen_spinVector(atom_list):
    """ From an atom_list, this method returns a numpy array of the atoms' spinvectors"""
    N_atoms = len(atom_list)
    vect_list = []
    for i in range(N_atoms):
        spinvect = atom_list[i].spin
        vect_list.append(spinvect[0])
        vect_list.append(spinvect[1])
        vect_list.append(spinvect[2])
    return np.array(vect_list)

def gen_anisotropy(atom_list):
    """ From an atom_list, this method returns a numpy array of the atoms' anisotropy terms"""
    N_atoms = len(atom_list)
    anis_vect = []
    for i in range(N_atoms):
        anis_vect.append(atom_list[i].Dx)
        anis_vect.append(atom_list[i].Dy)
        anis_vect.append(atom_list[i].Dz)
    anis_vect = np.array(anis_vect)
    return anis_vect

def calculate_Ham(sij, anis, jij, dsij = None):
    """ 
    Method the local optimizer uses
    Takes:
    - spin vector of the form [Sx1,Sy1,Sz1,Sx2,Sy2,Sz2,...]
    - anisotropy vector of the form [[Dx1,Dy1,Dz1],[Dx2,Dy2,Dz2],...]
    - jij sparse matrix of interaction matrices
    - dsij derivative of the spin vector. see the deriv() method in local optimizer
    Returns:
    - Hamilitonian Expression
    """
    
    sijT =sij.T
    
    # Normal mode
    # sij.T * jij * sij
    if dsij == None:
        res1 = sijT * jij
        Hij = np.dot(res1,sij).flat[0]
        Ham = - Hij - np.dot(anis, sij**2)
    # Derivative mode
    # dsij * jij * sij
    else:
        res1 = dsij * jij
        Hij = np.dot(res1,sij)
        Ham = - Hij - np.matrix(np.dot((dsij**2),anis)).T

    return Ham

def local_optimizer(interfile, spinfile):
    """
    Given an interaction file and a spin file, this optimizer will uses readfiles.readfiles to populate 
    an atom list as well as return the interaction matrices. Given that information, it will recalculate
    the hamiltonian and attempt to minimize it as a function of theta and phi. The annealing will get the
    spins close to the global minimum and this method makes sure the spins are aligned. This routine uses
    only python instead of C. 
    """

    # Get atoms list, jmats from the interaction and spin files
    atom_list, jnums, jmats,N_atoms_uc=rf.readFiles(interfile,spinfile,allAtoms=True)
    
    return opt_aux(atom_list, jmats)

def opt_aux(atom_list, jmats):
    """This method separates the functionality of the local optimization from
    the files.  jmats is assumed to be in order of index here, ie. jnums is of
    the form [0,1,2,3...]
    atomlist is a list of atoms as defined in spinwavecalc.readfiles"""
    N_atoms = len(atom_list)
    # Generate the Jij and anisotropy arrays
    Jij = gen_Jij(atom_list,jmats)
    anis = gen_anisotropy(atom_list)
    # Get the spin magnitudes from the atoms in atom list
    spin_mags = []
    for i in range(len(atom_list)):
        spin_mags.append(atom_list[i].spinMagnitude)
    spin_mags = np.array(spin_mags)
    
    # hamiltonian method
    def hamiltonian(p, Jij = None, spins = None, anis = None):
        """ Computes the hamiltonian given a list a thetas and phis"""
        # Thetas are the first half, phis the second half
        theta = p[:len(p)//2]
        phi = p[len(p)//2:]

        # Sx,Sy,Sz
        Sx = spins*cos(theta)*cos(phi)
        Sy = spins*cos(theta)*sin(phi)
        Sz = spins*sin(theta)

        # Array of spin vectors for each atom. Reshape it. Calculate hamiltonian with it and return the hamiltonian. 
        Sij = np.array([Sx,Sy,Sz])
        Sij = Sij.T.reshape(1,3*len(p)//2).T
        result = calculate_Ham(Sij, anis, Jij)

        return result
    
    # derivative of the hamiltonian
    def deriv(p, Jij = None, spins = None, anis = None):
        """ Computes the derivative of the hamiltonian with respect to each theta and then each phi"""
        # Thetas are the first half, phis the second half
        half = len(p)/2
        theta = p[:half]
        phi = p[half:]        

        # Sx,Sy,Sz
        Sx = spins*cos(theta)*cos(phi)
        Sy = spins*cos(theta)*sin(phi)
        Sz = spins*sin(theta)

        # dSx/dtheta,dSy/dtheta,dSz/dtheta
        Sxt = -spins*sin(theta)*cos(phi)
        Syt = -spins*sin(theta)*sin(phi)
        Szt = spins*cos(theta)
        # dSx/dphi,dSy/dphi,dSz/dphi
        Sxp = -spins*cos(theta)*sin(phi)
        Syp = spins*cos(theta)*cos(phi)
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
        Sij = Sij.T.reshape(1,3*len(p)//2).T
        # Calculate a hamiltonian for both theta and phi
        rest = calculate_Ham(Sij,anis,Jij,dsij = dSijt)
        resp = calculate_Ham(Sij,anis,Jij,dsij = dSijp)
        # Concatenate the two and return the result
        result = np.concatenate((np.array(rest),np.array(resp)))

        return result.T

    # populate initial p list
    # returns a numpy array of all the thetas followed by all the phis
    thetas = []
    phis = []
    for i in range(N_atoms):
        sx = atom_list[i].spin[0]
        sy = atom_list[i].spin[1]
        sz = atom_list[i].spin[2]
        s  = atom_list[i].spinMagnitude
        
        theta = arcsin(sz/s)
        phi   = np.arctan2(sy,sx)
        
        thetas.append(theta)
        phis.append(phi)
    p0 = np.array(thetas+phis)
    
    # define the limits parameter list
    limits = []
    for i in range(len(p0)):
        if i < len(p0)//2:#theta
            limits.append((-pi,pi))
        else:#phi
            limits.append((0,pi))
    
    # call minimizing function
    m = fmin_l_bfgs_b(hamiltonian, p0, fprime = deriv, args = (Jij, spin_mags, anis), bounds = limits)

    # grab returned parameters
    # thetas are the first half of the list, phis are the second half
    pout=m[0]
    theta=pout[0:len(pout)/2]
    phi=pout[len(pout)/2::]
    # recreate sx,sy,sz's
    sx=spin_mags*cos(theta)*cos(phi)
    sy=spin_mags*cos(theta)*sin(phi)
    sz=spin_mags*sin(theta)
    
    # return spin vector array
    return np.array([sx,sy,sz]).T


def optimization_driver(interfile, spinfile):
    """ runs the local optimization routine given just an interactionfile and spinfile. this should be called by the GUI """
    atom_list, jnums, jmats, N_atoms_uc = rf.readFiles(interfile,spinfile,allAtoms=True)
    N_atoms = len(atom_list)
    jij = gen_Jij(atom_list,jmats)
    anis = gen_anisotropy(atom_list)
    sij = gen_spinVector(atom_list)
    Ham = calculate_Ham(sij, anis, jij)
    
    return local_optimizer(interfile, spinfile)
    


if __name__ == '__main__':
    #print optimizeSpins('C:\\export.txt', 'C:\\spins.txt', 'C:\\spins2.txt')
    #interfile = 'c:/test_montecarlo.txt'
    #spinfile  = 'c:/test_Spins.txt'
    interfile='c:/montecarlo_ferro.txt'
    spinfile='c:/Spins_ferro.txt'

    atom_list, jnums, jmats,N_atoms_uc=rf.readFiles(interfile,spinfile,allAtoms=True)
    N_atoms = len(atom_list)

    jij = gen_Jij(atom_list, jmats)
    anis = gen_anisotropy(atom_list)
    sij = gen_spinVector(atom_list)
    Ham = calculate_Ham(sij, anis, jij)
    print Ham
    
    print local_optimizer(interfile, spinfile)
    
    
    