from __future__ import division
import sys

import sympy as sp
import numpy as np
import sympy.matrices as spm
from sympy import I,pi,var,exp,oo,sqrt
from sympy.physics.paulialgebra import delta
from sympy.core.cache import *
from timeit import default_timer as clock
import matplotlib
matplotlib.use('WXAgg')
import pylab
from matplotlib._pylab_helpers import Gcf
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from numpy import ma
from spinwaves.cross_section.util.list_manipulation import *
from spinwaves.cross_section.util.subin import sub_in
from spinwaves.cross_section.util.printing import *
from spinwaves.spinwavecalc.readfiles import atom, readFiles
from spinwaves.spinwavecalc.spinwave_calc_file import calculate_dispersion, calc_eigs_direct
#from periodictable import elements
sys.path.append('C:/tripleaxisproject-local/ tripleaxisproject/trunk/eclipse/src')
from rescalculator.lattice_calculator import Lattice, Orientation
from multiprocessing import Pipe
from copy import deepcopy

# Computes the inner product with a metric tensor
def inner_prod(vect1,vect2,ten = spm.Matrix([[1,0,0],
                                             [0,1,0],
                                             [0,0,1]])):
    # For column vectors -  make sure vectors match eachother as well as # of rows in tensor
    if vect1.shape == vect2.shape == (3,1) == (ten.lines,1): 
        return (vect1.T * ten * vect2)[0]
    # For row vectors -  make sure vectors match eachother as well as # of cols in tensor
    elif vect1.shape == vect2.shape == (1,3) == (1,ten.cols): 
        return (vect1 * ten * vect2.T)[0]
    # Everything else
    else: 
        return None

#------------ CROSS SECTION CALC METHODS ---------------------------------------

# Lists of the b and b dagger operators
def generate_b_bd_operators(atom_list):
    """Generates b and b dagger operators"""
    b_list = []; bd_list = []
    N = len(atom_list)
    for i in range(N):
        b = sp.Symbol('b%i'%(i,), commutative = False)
        bd = sp.Symbol('bd%i'%(i,), commutative = False)

        b_list.append(b); bd_list.append(bd)
    print "Operators Generated: b, bd"
    return (b_list,bd_list)

# Generates the a and a dagger operators
def generate_a_ad_operators(atom_list, k, b_list, bd_list):
    """Generates a and a dagger operators"""
    a_list = []; ad_list = []
    N = len(atom_list)
    t = sp.Symbol('t', real = True)
    q = sp.Symbol('q', real = True)
    L = sp.Symbol('L', real = True)
    wq = sp.Symbol('wq', real = True)

    for i in range(N):
        a_list.append(exp(I*(q*L - wq*t)) * b_list[i])
        ad_list.append(exp(-I*(q*L - wq*t)) * bd_list[i])

    a = sp.Pow(sp.sqrt(N),-1) * sum(a_list)
    ad = sp.Pow(sp.sqrt(N),-1) * sum(ad_list)

    print "Operators Generated: a, ad"
    return (a,ad)

# Generates the Sp and Sm operators
def generate_Sp_Sm_operators(atom_list, a, ad):
    """Generates S+ and S- operators"""
    S = sp.Symbol('S', commutative = True)

    Sp = sp.sqrt(2*S) * a
    Sm = sp.sqrt(2*S) * ad

    print "Operators Generated: Sp, Sm"
    return (Sp,Sm)

def generate_Sa_Sb_Sn_operators(atom_list, Sp, Sm):
    """Generates Sa, Sb, Sn operators"""
    S = sp.Symbol('S', commutative = True)

    Sa = ((1/2)*(Sp+Sm)).expand()
    Sb = ((1/2)*(1/I)*(Sp-Sm)).expand()
    Sn = (S - sp.Pow(2*S,-1) * Sm.expand() * Sp.expand()).expand()

    print "Operators Generated: Sa, Sb, Sn"
    return (Sa, Sb, Sn)

# Generates the Sx, Sy and Sz operators
def generate_Sx_Sy_Sz_operators(atom_list, Sa, Sb, Sn):
    """Generates Sx, Sy and Sz operators"""
    Sx_list = []; Sy_list = []; Sz_list = []
    N = len(atom_list)
    S = sp.Symbol('S', commutative = True)
    loc_vect = spm.Matrix([Sa,Sb,Sn])
    loc_vect = loc_vect.reshape(3,1)

    for i in range(N):
        rotmat = sp.Matrix(atom_list[i].spinRmatrix)
        glo_vect = rotmat * loc_vect

        Sx = sp.powsimp(glo_vect[0].expand())
        Sy = sp.powsimp(glo_vect[1].expand())
        Sz = sp.powsimp(glo_vect[2].expand())

        Sx_list.append(Sx)
        Sy_list.append(Sy)
        Sz_list.append(Sz)
        
    #Unit vector markers
    kapxhat = sp.Symbol('kapxhat',real = True)#spm.Matrix([1,0,0])#
    kapyhat = sp.Symbol('kapyhat',real = True)#spm.Matrix([0,1,0])#
    kapzhat = sp.Symbol('kapzhat',real = True)#spm.Matrix([0,0,1])#
    
    Sx_list.append(kapxhat)
    Sy_list.append(kapyhat)
    Sz_list.append(kapzhat)
    print "Operators Generated: Sx, Sy, Sz"
    return (Sx_list,Sy_list,Sz_list)

# Generate Hamiltonian
def generate_Hamiltonian(atom_list, b_list, bd_list):
    """Generates the Hamiltonian operator"""
    # Ham = Ham0 + sum over q of hbar*omega_q * bdq * bq
    # Ham0 = - S^2 N sum over rho of J(rho)
    # hbar * omega_q = 2 S {cJ(0)-cJ(q)}
    # sum over rho of J(rho) = Sum J(l-lp) from lp 0 to N l fixed
    # cJ(0) = sum over rho of J(rho)
    # cJ(q) = cJ(0)*exp(I*q*(l-lp))
    N = len(atom_list)
    S = sp.Symbol('S', commutative = True)

    J = sp.Function('J')
    q = sp.Symbol('q', commutative = True)
    l = sp.Symbol('l', commutative = True)
    lp = sp.Symbol('lp', commutative = True)
    rho = sp.Symbol('p', commutative = True)
    rho = l - lp

    # Define Curly J function
    def cJ(N,q):
        temp = []
        for i in range(N):
            temp.append(J(0-i) * sp.exp(I * q * (0-i)))
        return sum(temp)

    # Define hbar*omega_q function
    def hbwq(N,q):
        return 2*S * (cJ(N,0) - cJ(N,q))

    Ham0 = -S**2 * N * cJ(N,0)

    # Sum over hbar*omega_q for all q
    temp2 = []
    for i in range(N):
        temp2.append(hbwq(N,i) * bd_list[i] * b_list[i])
    Ham_sum = sum(temp2)
    Ham = Ham0 + Ham_sum

    print "Generated: Hamiltonian"
    return Ham

# Define a method that generates the possible combinations of operators
def generate_possible_combinations(atom_list, alist):
    """This method returns the possible operator combinations from a list of operators"""
    # For a combination to be returned, the product must have an equal number of b
    # and b dagger operators. If not, they are rejected.
    op_list = []
    alista = []
    N = len(atom_list)
    t = sp.Symbol('t', commutative = True)
    L = sp.Symbol('L', real = True)
    q = sp.Symbol('q', real = True)
    qp = sp.Symbol('qp', real = True)
    wq = sp.Symbol('wq', real = True)
    wqp = sp.Symbol('wqp', real = True)

    alista = [[subelement.subs(t, 0) for subelement in element] for element in alist]
    for ele in alista:
        for sub in ele:
            sub = sub.subs(L,0)

    for i in range(len(alist)):
        for j in range(len(alist)):
            vect1 = alist[i][-1]
            vect2 = alist[j][-1]
            if cmp(vect1, vect2) == 0: delta = 1
            else: delta = 0

            allzerolist = [alista[i][0].subs(L,0) for k in range(len(alista[i])-1)]+[delta-vect1*vect2]
            otherlist = [alist[j][k].subs(q,qp).subs(wq,wqp) for k in range(len(alist[j])-1)]+[1]
            append_list = list_mult(allzerolist,otherlist)
            op_list.append(append_list)
    print "Generated: Possible Operator Combinations"
    return op_list

# 
def holstein(atom_list, arg):
    new = []
    N = len(atom_list)
    S = sp.Symbol('S', commutative = True)
    for k in range(len(arg)):
        temp = []
        orig = len(arg[k])
        for i in range(N):
            Snew = atom_list[i].spinMagnitude
            
            S2coeff = coeff(arg[k][i], S**2)
            Scoeff = coeff(arg[k][i], S)
            if S2coeff != None and Scoeff != None:
                temp.append((S2coeff*S**2 + Scoeff*S).subs(S,Snew))
            elif S2coeff != None and Scoeff == None:
                temp.append((S2coeff*S**2).subs(S,Snew))
            elif S2coeff == None and Scoeff != None:
                temp.append((Scoeff*S).subs(S,Snew))
        if temp != []:
            if temp[0] != 0:
                temp.append(arg[k][-1])
                new.append(temp)
    print "Applied: Holstein"
    return new

def reduce_options(atom_list, arg):
    """
    Further reduces possible operator combinations by removing combinations if
    they are the negative of another combination or they are not time dependent
    (i.e. elastic scattering)
    """
    new = []
    N = len(atom_list)
    t = sp.Symbol('t')
    for element in arg:
        if str(element[0]).find('t') > 0:
            new.append(element)

    for elementa in new:
        if elementa == 0:
            new.remove(elementa)
            break
        for elementb in new:
            if elementa[0].expand(deep = False) == (-1*elementb[0]).expand(deep = False):
                new.remove(elementa)
                new.remove(elementb)
                break
    print 'Applied: Possible Operator Reduction'
    return new

# Apply Commutation Relation
def apply_commutation(atom_list, arg):
    """Applies the commutation relation of [b_i, bd_j] = kronecker delta _ ij"""
    # [bi,bdj] = delta_ij
    # Thus commutator = 0 (THEY COMMUTE) for i != j
    # Thus commutator = 1 for i == j
        # Then just put '+1' after commutation
    # NOTE: This method will take bd*b*bd*b to bd*(bd*b+1)*b so
    # I have replace bd_b called first but implement it inside this method too.
    N = len(atom_list)
    if type(arg) == type([]):
        for k in range(len(arg)):
            for i in range(N):
                for j in range(N):
                    bj = sp.Symbol('b%i'%(j,), commutative = False)
                    bdj = sp.Symbol('bd%i'%(j,), commutative = False)
                    nj = sp.Symbol('n%i'%(j,), commutative = False)

                    for g in range(N):
                        bg = sp.Symbol('b%i'%(g,), commutative = False)
                        bdg = sp.Symbol('bd%i'%(g,), commutative = False)
                        
                        arg[k][i] = arg[k][i].subs(bg*bj,0)
                        arg[k][i] = arg[k][i].subs(bdg*bdj,0)
                        
                        if j == g:
                            arg[k][i] = arg[k][i].subs(bj*bdg, bdg*bj+1)
                        else:
                            arg[k][i] = arg[k][i].subs(bj*bdg, bdg*bj)
#
#                        if g == j:
##                            arg[k][i] = (arg[k][i].subs(bdg*bj, nj))
##                            arg[k][i] = (arg[k][i].subs(bj*bdg, (bdg*bj+1)))
#                            arg[k][i] = (arg[k][i].subs(bdg*bj, nj))
#                        elif g != j:
#                            arg[k][i] = (arg[k][i].subs(bg*bdj, 0))
#                            arg[k][i] = (arg[k][i].subs(bj*bdg, 0))
#                            arg[k][i] = (arg[k][i].subs(bdg*bj, 0))
#                            arg[k][i] = (arg[k][i].subs(bdj*bg, 0))

        print "Applied: Commutation"
        return arg

# Replaces expressions arranged by apply_commutation
def replace_bdb(atom_list, arg):
    """Replaces bdqbq with nq"""
    # Replaces bdq*bq' with nq when q = q'
    N = len(atom_list)
    for k in range(len(arg)):
        for i in range(N):
            for j in range(N):
                bj = sp.Symbol('b%i'%(j,), commutative = False)
                bdj = sp.Symbol('bd%i'%(j,), commutative = False)
                nj = sp.Symbol('n%i'%(j,), real = True)

                for g in range(N):
                    bg = sp.Symbol('b%i'%(g,), commutative = False)
                    bdg = sp.Symbol('bd%i'%(g,), commutative = False)

                    if j == g:
                        arg[k][i] = (arg[k][i].subs(bdg*bj, nj))

                    elif j != g:
                        arg[k][i] = (arg[k][i].subs((bdj*bg), 0))
                        arg[k][i] = (arg[k][i].subs((bdg*bj), 0))
                        

                    arg[k][i] = (arg[k][i].subs((bdj*bdg), 0))
                    arg[k][i] = (arg[k][i].subs((bj*bg), 0))
                    arg[k][i] = (arg[k][i].subs((bdg*nj), 0))
                    arg[k][i] = (arg[k][i].subs((bg*nj), 0))
            print '1', arg[k][i]
            q = sp.Symbol('q', real = True)
            qp = sp.Symbol('qp', real = True)
            wq = sp.Symbol('wq', real = True)
            wqp = sp.Symbol('wqp', real = True)
            arg[k][i] = arg[k][i].subs(qp,q).subs(wqp,wq)
            print '2', arg[k][i]
    print "Applied: bdq*bq Replacement"
    return arg


def eq(a,b,tol=2e-1):
        c = (abs(a-b) < tol)
        if c:
            return 0
        else:
            if a>b: return abs(a-b)
            else: return abs(b-a)
#def eval_cross_section(N, N_uc, atom_list, jmats, cross, qvals, temp, direction, lmin, lmax):
def generate_cross_section(interactionfile, spinfile, lattice, arg, 
                       tau_list, h_list, k_list, l_list, w_list, 
                       temperature, steps, eief, efixed = 14.7):
    """
    Calculates the cross_section given the following parameters:
    interactionfile, spinfile - files to get atom data
    lattice     - Lattice object from tripleaxisproject
    arg         - reduced list of operator combinations
    tau_list    - list of tau position
    w_list      - list of w's probed
    temp        - temperature
    kmin        - minimum value of k to scan
    kmax        - maximum value of k to scan
    steps       - number of steps between kmin, kmax
    eief        - True if fixed Ef
    efixed      - value of the fixed energy, either Ei or Ef
    """

    # Read files, get atom_list and such
    atom_list, jnums, jmats,N_atoms_uc=readFiles(interactionfile,spinfile)
    

    # Get Hsave to calculate its eigenvalues
    N_atoms = len(atom_list)
    Hsave,poly,heigs = calculate_dispersion(atom_list,N_atoms_uc,N_atoms,jmats,showEigs=True)
    atom_list=atom_list[:N_atoms_uc]
    N_atoms = len(atom_list)
    N = N_atoms
    
    print "Calculated: Dispersion Relation"

    # Generate kappa's from (h,k,l)
    #kaprange = []
    #kapvect = []
    #if len(h_list) == len(k_list) == len(l_list):
        #for i in range(len(h_list)):
            #kappa = lattice.modvec(h_list[i],k_list[i],l_list[i], 'latticestar')
            #kaprange.append(kappa[0])
            #kapvect.append(np.array([h_list[i],k_list[i],l_list[i]]))
##            kapvect.append(np.array([h_list[i]/kappa,k_list[i]/kappa,l_list[i]/kappa]))
    #else:
        #raise Exception('h,k,l not same lengths')
    # Generate q's from kappa and tau
    kaprange=lattice.modvec(h_list,k_list,l_list, 'latticestar')
    nkpts=len(kaprange)
    kapvect=np.empty((nkpts,3),'Float64')
    kapvect[:,0]=h_list
    kapvect[:,1]=k_list
    kapvect[:,2]=l_list
#    print kapvect.shape
#    print kaprange.shape
    kapunit = kapvect.copy()
    kapunit[:,0]=kapvect[:,0]/kaprange
    kapunit[:,1]=kapvect[:,1]/kaprange
    kapunit[:,2]=kapvect[:,2]/kaprange
    #plusq=kappa-tau
    plusq=[]
    minusq=[]
    qlist=[]
    ones_list=np.ones((1,nkpts),'Float64')
    #wtlist=np.ones((1,nkpts*2),'Float64').flatten()
    wtlist=np.hstack([w_list,w_list])
    #weven=np.array(range(0,nkpts*2,2))
    #wodd=np.array(range(1,nkpts*2,2))
    #wtlist[wodd]=w_list
    #wtlist[weven]=w_list    
    qtlist=[]
    
    
    
    for tau in tau_list:
        taui=np.ones((nkpts,3),'Float64')
        taui[:,0]=ones_list*tau[0]
        taui[:,1]=ones_list*tau[1]
        taui[:,2]=ones_list*tau[2]
        kappa_minus_tau=kapvect-taui
        tau_minus_kappa=taui - kapvect
               
        qlist.append(np.vstack([kappa_minus_tau,tau_minus_kappa]))
    #calculate kfki
    nqpts=nkpts*2
    kfki=calc_kfki(w_list,eief,efixed)


    eig_list=[]
#    print qlist
    eigs = Hsave.eigenvals().keys()
    for q in qlist:
        #eigs = calc_eigs_direct(Hsave,q[:,0],q[:,1],q[:,2])
        #eigs = Hsave.eigenvals().keys()
        print eigs
        eig_list.append(eigs)
    print "Calculated: Eigenvalues"
    
 #   print len(qlist)
 #   print len(eig_list[0])
#    sys.exit()
    # Grab Form Factors
#----Commented out 9/14/09 by Tom becuase of magnetic_ff error--------------
    #ff_list = []
    #s = sp.Symbol('s')
    #for i in range(N_atoms):
        #el = elements[atom_list[i].atomicNum]
        #val = atom_list[i].valence
        #if val != None:
            #Mq = el.magnetic_ff[val].M_Q(kaprange)
        #else:
            #Mq = el.magnetic_ff[0].M_Q(kaprange)
        #ff_list.append(Mq)
    #print "Calculated: Form Factors"
#--------------------------------------------------------------------------

    # Other Constants
    gamr0 = 2*0.2695e-12 #sp.Symbol('gamma', commutative = True)
    hbar = sp.S(1.0) # 1.05457148*10**(-34) #sp.Symbol('hbar', commutative = True)
    g = 2.#sp.Symbol('g', commutative = True)
    # Kappa vector
    kap = sp.Symbol('kappa', real = True)#spm.Matrix([sp.Symbol('kapx',real = True),sp.Symbol('kapy',real = True),sp.Symbol('kapz',real = True)])
    t = sp.Symbol('t', real = True)
    w = sp.Symbol('w', real = True)
    W = sp.Symbol('W', real = True)
    tau = sp.Symbol('tau', real = True)
    Q = sp.Symbol('q', real = True)
    L = sp.Symbol('L', real = True)
    lifetime=sp.Symbol('V',real=True)
    boltz = 8.617343e-2
    
    # Wilds for sub_in method
    A = sp.Wild('A',exclude = [0,t])
    B = sp.Wild('B',exclude = [0,t])
    C = sp.Wild('C')
    D = sp.Wild('D')
    K = sp.Wild('K')
    
    # Grabs the unit vectors from the back of the lists. 
    unit_vect = []
    kapxhat = sp.Symbol('kapxhat',real=True)
    kapyhat = sp.Symbol('kapyhat',real=True)
    kapzhat = sp.Symbol('kapzhat',real=True)
    for i in range(len(arg)):
        unit_vect.append(arg[i].pop())
    #print unit_vect
    unit_vect = sum(unit_vect)
    print unit_vect
    
    csection=0
    for i in range(len(arg)):
        for j in range(len(arg[i])):
            csection = (csection + arg[i][j]*unit_vect)
    
    for i in range(N):
        ni = sp.Symbol('n%i'%(i,), real = True)
        np1i = sp.Symbol('np1%i'%(i,), Real = True)
        csection.subs(ni+1,np1i)

    csection = csection.expand()
    
    print csection
    
    csection = csection.subs(kapxhat*kapyhat,0)
    csection = csection.subs(kapxhat*kapzhat,0)
    csection = csection.subs(kapyhat*kapzhat,0)
    csection = csection.subs(kapzhat*kapyhat,0)
    csection = csection.subs(kapzhat*kapxhat,0)
    csection = csection.subs(kapyhat*kapxhat,0)
    
    print csection
    
    csection = (csection * exp(-I*w*t) * exp(I*kap*L)).expand()
    csection = sp.powsimp(csection, deep=True)
    print 'intermediate'
    print csection
    csection = sp.powsimp(csection)
    csection = sub_in(csection,exp(I*t*A + I*t*B + I*C + I*D + I*K),sp.DiracDelta(A*t + B*t + C + D + K))
    csection = sub_in(csection,sp.DiracDelta(A*t + B*t + C*L + D*L ),sp.DiracDelta(A*hbar + B*hbar)*sp.simplify(sp.DiracDelta(C + D  - tau)))  #This is correct
    #csection = sub_in(csection,sp.DiracDelta(A*t + B*t + C*L + D*L ),sp.simplify(lifetime*sp.DiracDelta(C + D  - tau)*
                                                                                 #sp.Pow((A-B)**2+lifetime**2,-1)))

    ## First the exponentials are turned into delta functions:
    #for i in range(len(arg)):
        #for j in range(N):
##            print '1', arg[i][j]
            #arg[i][j] = sp.powsimp(arg[i][j])#sp.powsimp(arg[i][j], deep = True, combine = 'all')
            #arg[i][j] = (arg[i][j] * exp(-I*w*t) * exp(I*kap*L)).expand()# * exp(I*inner_prod(spm.Matrix(atom_list[j].pos).T,kap))
            #arg[i][j] = sp.powsimp(arg[i][j])#sp.powsimp(arg[i][j], deep = True, combine = 'all')
##            print '2', arg[i][j]
            #arg[i][j] = sub_in(arg[i][j],exp(I*t*A + I*t*B + I*C + I*D + I*K),sp.DiracDelta(A*t + B*t + C + D + K))#*sp.DiracDelta(C))
##            print '3', arg[i][j]
            #arg[i][j] = sub_in(arg[i][j],sp.DiracDelta(A*t + B*t + C*L + D*L + K*L),sp.DiracDelta(A*hbar + B*hbar)*sp.simplify(sp.DiracDelta(C + D + K + tau)))
##            print '4', arg[i][j]
            #arg[i][j] = sub_in(arg[i][j],sp.DiracDelta(-A - B)*sp.DiracDelta(C),sp.DiracDelta(A + B)*sp.DiracDelta(C))
##            arg[i][j] = arg[i][j].subs(-w - wq, w + wq)
##            print '5', arg[i][j]
    

    # Subs the actual values for kx,ky,kz, omega_q and n_q into the operator combos
    # The result is basically a nested list:
    #
    #    csdata     = [ op combos ]
    #    op combos  = [ one combo per atom ]
    #    1 per atom = [ evaluated exp ]
    #
    csection = sub_in(csection,sp.DiracDelta(-A - B)*sp.DiracDelta(C),sp.DiracDelta(A + B)*sp.DiracDelta(C))
    print "Applied: Delta Function Conversion"
    
    

    print "end part 1"
    #print csection
    return (N_atoms_uc, csection, kaprange, qlist, tau_list, eig_list, kapvect, wtlist)
    
def eval_cross_section(N_atoms_uc, csection, kaprange, qlist, tau_list, eig_list, kapvect, wtlist):    
    print "begin part 2"
    
    print 'kap',len(kaprange)
    print 'tau',len(tau_list)
    print 'eig',len(eig_list)
    print 'wt ',len(wtlist)
    
    gamr0 = 2*0.2695e-12 #sp.Symbol('gamma', commutative = True)
    hbar = sp.S(1.0) # 1.05457148*10**(-34) #sp.Symbol('hbar', commutative = True)
    g = 2.#sp.Symbol('g', commutative = True)
    # Kappa vector
    kap = sp.Symbol('kappa', real = True)#spm.Matrix([sp.Symbol('kapx',real = True),sp.Symbol('kapy',real = True),sp.Symbol('kapz',real = True)])
    t = sp.Symbol('t', real = True)
    w = sp.Symbol('w', real = True)
    W = sp.Symbol('W', real = True)
    tau = sp.Symbol('tau', real = True)
    Q = sp.Symbol('q', real = True)
    L = sp.Symbol('L', real = True)
    lifetime=sp.Symbol('V',real=True)
    boltz = 8.617343e-2    
    kapxhat = sp.Symbol('kapxhat',real=True)
    kapyhat = sp.Symbol('kapyhat',real=True)
    kapzhat = sp.Symbol('kapzhat',real=True)    

    A = sp.Wild('A',exclude = [0,t])
    B = sp.Wild('B',exclude = [0,t])
    C = sp.Wild('C')
    D = sp.Wild('D')
    K = sp.Wild('K')

    kapunit = kapvect.copy()
    kapunit[:,0]=kapvect[:,0]/kaprange
    kapunit[:,1]=kapvect[:,1]/kaprange
    kapunit[:,2]=kapvect[:,2]/kaprange    
    
    nkpts = len(kaprange)
    nqpts = len(qlist)
    print 'kpts',nkpts
    print 'qpts',nqpts
    
    csdata=[]
    wq = sp.Symbol('wq', real = True)

#    for tauindex in range(len(tau_list)):
#        temp1=[]
#        for kapindex in range(len(kaprange)):
#            temp2=[]
#            for qindex in range(nqpts):
#                temp3=[]
#                for eigindex in range(len(eig_list[qindex])):
#                    tempcsec = deepcopy(csection)
#                    for num in range(N_atoms_uc):
#                        nq = sp.Symbol('n%i'%(num,), real=True)
#                        n=sp.S(0.0)
#                        tempcsec=tempcsec.subs(nq,n)
#
#                    if kapindex < nkpts:
#                        val = kapvect[kapindex]-tau_list[tauindex]-qlist[qindex][0]
#                        if eq(val[0],0)==0 and eq(val[1],0)==0 and eq(val[2],0)==0:
#                            tempcsec = tempcsec.subs(sp.DiracDelta(kap-tau-Q),sp.S(1))
#                            tempcsec = tempcsec.subs(sp.DiracDelta(-kap+tau+Q),sp.S(1))
#                        else:
#                            tempcsec = tempcsec.subs(sp.DiracDelta(kap-tau-Q),sp.S(0))
#                            tempcsec = tempcsec.subs(sp.DiracDelta(-kap+tau+Q),sp.S(0))
#                        tempcsec = tempcsec.subs(kapxhat,kapunit[kapindex,0])
#                        tempcsec = tempcsec.subs(kapyhat,kapunit[kapindex,1])
#                        tempcsec = tempcsec.subs(kapzhat,kapunit[kapindex,2])
#                    elif kapindex > nkpts:
#                        val = kapvect[kapindex-nkpts]-tau_list[tauindex]-qlist[qindex][0]
#                        if eq(val[0],0)==0 and eq(val[1],0)==0 and eq(val[2],0)==0:
#                            tempcsec = tempcsec.subs(sp.DiracDelta(kap-tau-Q),sp.S(1))
#                            tempcsec = tempcsec.subs(sp.DiracDelta(-kap+tau+Q),sp.S(1))
#                        else:
#                            tempcsec = tempcsec.subs(sp.DiracDelta(kap-tau-Q),sp.S(0))
#                            tempcsec = tempcsec.subs(sp.DiracDelta(-kap+tau+Q),sp.S(0))
#                        tempcsec = tempcsec.subs(kapxhat,kapunit[kapindex,0])
#                        tempcsec = tempcsec.subs(kapyhat,kapunit[kapindex,1])
#                        tempcsec = tempcsec.subs(kapzhat,kapunit[kapindex,2])
    for tindex in range(len(tau_list)):
        temp1=[]
        for qindex in range(nqpts):
            temp2=[]
            for eindex in range(len(eig_list[qindex])):
                #temp = eval_it(k,g,i,nkpts,csection)
                temp = deepcopy(csection)

                for num in range(N_atoms_uc):
                    nq = sp.Symbol('n%i'%(num,), real = True)
                    #n = sp.Pow( sp.exp(-np.abs(eig_list[k][g][i])/boltz/temperature) - 1 ,-1)
                    n=sp.S(0.0)
                    temp = temp.subs(nq,n)
                    
                #if g==0:
                if qindex < nkpts:
                    value = kapvect[qindex]-tau_list[tindex] - qlist[tindex][qindex]
                    if eq(value[0],0) == 0 and eq(value[1],0) == 0 and eq(value[2],0) == 0:
                        temp = temp.subs(sp.DiracDelta(kap-tau-Q),sp.S(1))
                        temp = temp.subs(sp.DiracDelta(-kap+tau+Q),sp.S(1)) # recall that the delta function is symmetric
                    else:
                        temp = temp.subs(sp.DiracDelta(kap-tau-Q),sp.S(0))
                        temp = temp.subs(sp.DiracDelta(-kap+tau+Q),sp.S(0))
                    temp = temp.subs(kapxhat,kapunit[qindex,0])
                    temp = temp.subs(kapyhat,kapunit[qindex,1])
                    temp = temp.subs(kapzhat,kapunit[qindex,2])
                #value =kapvect[g//2]- tau_list[k] + qlist[k][g] 
                #if g%2!=0:
                elif qindex >= nkpts:
                    value = kapvect[qindex-nkpts]-tau_list[tindex] - qlist[tindex][qindex]
                    if eq(value[0],0) == 0 and eq(value[1],0) == 0 and eq(value[2],0) == 0:
                        temp = temp.subs(sp.DiracDelta(kap-tau+Q),sp.S(1))
                        temp = temp.subs(sp.DiracDelta(-kap+tau-Q),sp.S(1))
                    else:
                        temp = temp.subs(sp.DiracDelta(kap-tau+Q),sp.S(0))
                        temp = temp.subs(sp.DiracDelta(-kap+tau-Q),sp.S(0))
                    temp = temp.subs(kapxhat,kapunit[qindex-nkpts,0])
                    temp = temp.subs(kapyhat,kapunit[qindex-nkpts,1])
                    temp = temp.subs(kapzhat,kapunit[qindex-nkpts,2])                    
                value = tau_list[tindex]      
                if eq(value[0],0) == 0 and eq(value[1],0) == 0 and eq(value[2],0) == 0:
                    temp = temp.subs(sp.DiracDelta(kap-tau-Q),sp.S(1))
                    temp = temp.subs(sp.DiracDelta(-kap+tau+Q),sp.S(1)) #
                    temp = temp.subs(sp.DiracDelta(kap-tau+Q),sp.S(1))
                    temp = temp.subs(sp.DiracDelta(-kap+tau-Q),sp.S(1))
                    
                #temp = temp.subs(kapxhat,kapunit[g//2,0])
                #temp = temp.subs(kapyhat,kapunit[g//2,1])
                #temp = temp.subs(kapzhat,kapunit[g//2,2])
                
                
                #value = eig_list[tindex][qindex][eindex] - wtlist[qindex] 
                #print '1',temp
                temp = temp.subs(wq,eig_list[qindex][eindex])

                temp=temp.subs(w,wtlist[qindex])
                if 0:
                    if eq(value,0) == 0:
                        temp = temp.subs(sp.DiracDelta(wq - w), sp.S(1))
                    else: 
                        temp = temp.subs(sp.DiracDelta(wq - w), sp.S(0))
                if 0:
                    if eq(eig_list[qindex][eindex], wtlist[qindex]) == 0:
                        G=sp.Wild('G', exclude = [Q,kap,tau,w])
                        temp=sub_in(temp, sp.DiracDelta(G - A*w), sp.S(1))
                    elif eq(eig_list[qindex][eindex], -wtlist[qindex]) == 0:
                        G=sp.Wild('G', exclude = [Q,kap,tau,w])
                        temp=sub_in(temp, sp.DiracDelta(G - A*w), sp.S(1))
                    else:
                        temp = temp.subs(w,wtlist[qindex])

#                print '4',temp
                temp2.append(temp)
            temp1.append(temp2)
        csdata.append(temp1)


    #print csdata

    # Front constants and stuff for the cross-section
    front_constant = 1.0#(gamr0)**2#/(2*pi*hbar)
    front_func = (1./2.)*g#*F(k)
    debye_waller= 1. #exp(-2*W)
    
    cstemp=[]
    for g in range(nqpts):
        for k in range(len(tau_list)):
            cstemp.append(csdata[k][g][0].subs(lifetime,sp.S(.1)))
            
    print cstemp
    sys.exit()
    
    qtlist=np.array(qlist,'Float64').reshape((nqpts*len(tau_list),3))[:,0]
    xi=qtlist
    yi=wtlist
    zi=np.array(cstemp,'Float64')
    Z=matplotlib.mlab.griddata(xi,yi,zi,xi,yi)
    zmin, zmax = np.min(Z), np.max(Z)
    locator = ticker.MaxNLocator(10) # if you want no more than 10 contours
    locator.create_dummy_axis()
    locator.set_bounds(zmin, zmax)
    levs = locator()
    levs[0]=1.0
    print zmin, zmax
    #zm=ma.masked_where(Z<=0,Z)
    zm=Z
    print zm
    print levs
    plt.contourf(xi,yi,Z, levs)#, norm=matplotlib.colors.LogNorm(levs[0],levs[len(levs)-1]))
    l_f = ticker.LogFormatter(10, labelOnlyBase=False) 
    cbar = plt.colorbar(ticks = levs, format = l_f)
    plt.show()

    #csrange = []
    #for g in range(nqpts):
        #temp1=[]
        #for k in range(len(tau_list)):
            #for lrange in range(len(eig_list[k][g])):
                #dif = csdata[k][g][lrange].subs(lifetime,sp.S(.1))
                #dif = dif*front_func**2*front_constant*debye_waller#*kfki[g]# * ff_list[0][q]
                #temp1.append(dif)
        #csrange.append(sum(temp1))
    #print "Calculated: Cross-section"
    #csrange=np.real(csrange)
    #csrange=np.array(csrange,'Float64')
    #print csrange.shape

    #zi=csrange
    #X,Y = np.meshgrid(xi,yi)
    #Z=matplotlib.mlab.griddata(xi,yi,zi,xi,yi)
##    Z=np.zeros((len(xi),len(yi))) 
##    print Z.shape
##    print xi.shape
##    print yi.shape
   ## Z[range(len(xi)),range(len(yi))]=zi
    
    #if 0:
        #pylab.plot(xi,zi,'s')
        #pylab.show()
    #if 0:
        #pylab.plot(yi,zi,'s')
        #pylab.show()
    
    #zmin, zmax = np.min(Z), np.max(Z)
    #locator = ticker.MaxNLocator(10) # if you want no more than 10 contours
    #locator.create_dummy_axis()
    #locator.set_bounds(zmin, zmax)
    #levs = locator()
    #levs[0]=1.0
    #print zmin, zmax
    ##zm=ma.masked_where(Z<=0,Z)
    #zm=Z
    #print zm
    #print levs
    #pylab.contourf(xi,yi,zm, levs)#, norm=matplotlib.colors.LogNorm(levs[0],levs[len(levs)-1]))
    #l_f = ticker.LogFormatter(10, labelOnlyBase=False) 
    #cbar = pylab.colorbar(ticks = levs, format = l_f) 
    #if 0:
        #pylab.show()
    #print 'hi'
##    pylab.contourf(qrange,wrange,csrange)
##    pylab.show()

    
def calc_kfki(w,eief,efixed):
    #eief==True if fixed Ef

    
    if eief==True:
        #fixed ef
        w_f=efixed*np.ones((len(w),1),'Float64')
        w_i=w+w_f
    else:
        #fixed ei
        w_i=efixed*np.ones((len(w),1),'Float64')
        w_f=w_i-w
        
    kfki=np.sqrt(w_f/w_i)
    return kfki
    
def run_cross_section(interactionfile, spinfile):
    start = clock()

    # Generate Inputs
    atom_list, jnums, jmats,N_atoms_uc=readFiles(interactionfile,spinfile)
    atom1 = atom(pos = [0.00,0,0], neighbors = [1],   interactions = [0], int_cell = [0], 
                 atomicNum = 26, valence = 3)
    atom2 = atom(pos = [0.25,0,0], neighbors = [0,2], interactions = [0], int_cell = [0], 
                 atomicNum = 26, valence = 3)
    atom3 = atom(pos = [0.50,0,0], neighbors = [1,3], interactions = [0], int_cell = [0], 
                 atomicNum = 26, valence = 3)
    atom4 = atom(pos = [0.75,0,0], neighbors = [2],   interactions = [0], int_cell = [0], 
                 atomicNum = 26, valence = 3)
#    atom_list,N_atoms_uc = ([atom1, atom2, atom3, atom4],1)
    
    atom_list=atom_list[:N_atoms_uc]
    N_atoms = len(atom_list)
    
    print N_atoms
    if 1:
        kx = sp.Symbol('kx', real = True)
        ky = sp.Symbol('ky', real = True)
        kz = sp.Symbol('kz', real = True)
        k = spm.Matrix([kx,ky,kz])
    
    (b,bd) = generate_b_bd_operators(atom_list)
#    list_print(b)
    (a,ad) = generate_a_ad_operators(atom_list, k, b, bd)
#    list_print(a)
    (Sp,Sm) = generate_Sp_Sm_operators(atom_list, a, ad)
#    list_print(Sp)
    (Sa,Sb,Sn) = generate_Sa_Sb_Sn_operators(atom_list, Sp, Sm)
#    list_print(Sa)
    (Sx,Sy,Sz) = generate_Sx_Sy_Sz_operators(atom_list, Sa, Sb, Sn)
#    list_print(Sx)
    print ''
    
    #Ham = generate_Hamiltonian(N_atoms, atom_list, b, bd)
    ops = generate_possible_combinations(atom_list, [Sx,Sy,Sz])
#    list_print(ops)
    ops = holstein(atom_list, ops)
#    list_print(ops)
    ops = apply_commutation(atom_list, ops)
#    list_print(ops)
    ops = replace_bdb(atom_list, ops)
#    list_print(ops)

    ops = reduce_options(atom_list, ops)
    list_print(ops)

    # Old Method
    #cross_sect = generate_cross_section(atom_list, ops, 1, real, recip)
    #print '\n', cross_sect
    
    if 1:
        aa = bb = cc = np.array([2.0*np.pi], 'Float64')
        alpha = beta = gamma = np.array([np.pi/2.0], 'Float64')
        vect1 = np.array([[1,0,0]])
        vect2 = np.array([[0,0,1]])
        lattice = Lattice(aa, bb, cc, alpha, beta, gamma, Orientation(vect1, vect2))
        data={}
        data['kx']=1.
        data['ky']=0.
        data['kz']=0.
        direction=data

        temperature = 1.0
        kmin = 0
        kmax = 2*sp.pi
        steps = 25

        tau_list = []
        for i in range(1):
            tau_list.append(np.array([0,0,0], 'Float64'))

        h_list = np.linspace(0.1,12.0,200)
        k_list = np.zeros(h_list.shape)
        l_list = np.zeros(h_list.shape)
        w_list = np.linspace(0.1,10.0,200)

        efixed = 14.7 #meV
        eief = True
        N_atoms_uc,csection,kaprange,qlist,tau_list,eig_list,kapvect,wtlist = generate_cross_section(interactionfile, spinfile, lattice, ops, 
                                                                                                     tau_list, h_list, k_list, l_list, w_list,
                                                                                                     temperature, steps, eief, efixed)
        print csection

        return N_atoms_uc,csection,kaprange,qlist,tau_list,eig_list,kapvect,wtlist
    end = clock()
    print "\nFinished %i atoms in %.2f seconds" %(N_atoms,end-start)
    
def run_eval_cross_section(N_atoms_uc,csection,kaprange,qlist,tau_list,eig_list,kapvect,wtlist):
    eval_cross_section(N_atoms_uc,csection,kaprange,qlist,tau_list,eig_list,kapvect,wtlist)


#---------------- MAIN --------------------------------------------------------- 


if __name__=='__main__':
    
    interfile = 'c:/Users/Bill/Documents/montecarlo.txt'#'c:/montecarlo.txt'
    spinfile = 'c:/Users/Bill/Documents/spins.txt'#'c:/Spins.txt'
    
    N_atoms_uc,csection,kaprange,qlist,tau_list,eig_list,kapvect,wtlist = run_cross_section(interfile,spinfile)
#    left_conn, right_conn = Pipe()
#    p = Process(target = create_latex, args = (right_conn, csection, "Cross-Section"))
#    p.start()
#    eig_frame = LaTeXDisplayFrame(self.parent, p.pid, left_conn.recv(), 'Cross-Section')
#    self.process_list.append(p)
#    p.join()
#    p.terminate()
    run_eval_cross_section(N_atoms_uc,csection,kaprange,qlist,tau_list,eig_list,kapvect,wtlist)


    ### THINGS LEFT TO DO
    # - optimize for N_atoms > 2
    # - Fix definition of F function in cross_section