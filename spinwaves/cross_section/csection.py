from __future__ import division
import sys

import sympy as sp
import numpy as np
import sympy.matrices as spm
from scipy.integrate import dblquad
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
from periodictable import elements
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
    kapxhat = sp.Symbol('kapxhat',real = True)
    kapyhat = sp.Symbol('kapyhat',real = True)
    kapzhat = sp.Symbol('kapzhat',real = True)
    
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
    t = sp.Symbol('t', real = True)
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
            print 'here',append_list
            op_list.append(append_list)
    print "Generated: Possible Operator Combinations"
    return op_list

# 
def holstein(atom_list, arg):
    new = []
    N = len(atom_list)
    S = sp.Symbol('S', real = True)
    t = sp.Symbol('t', real = True)
    for k in range(len(arg)):
        temp = []
        for i in range(N):
            Snew = atom_list[i].spinMagnitude

            #gets rid of time independent terms
            #works since arg[k][i] is an Add instance so args breaks it up at '+'s
            pieces = arg[k][i].args
            for piece in pieces:
                if not piece.has(t):
                    arg[k][i] = arg[k][i] - piece

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

            q = sp.Symbol('q', real = True)
            qp = sp.Symbol('qp', real = True)
            wq = sp.Symbol('wq', real = True)
            wqp = sp.Symbol('wqp', real = True)
            arg[k][i] = arg[k][i].subs(qp,q).subs(wqp,wq)

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
                       tau_list, h_list, k_list, l_list, w_list):
    """
    Calculates the cross_section given the following parameters:
    interactionfile, spinfile - files to get atom data
    lattice     - Lattice object from tripleaxisproject
    arg         - reduced list of operator combinations
    tau_list    - list of tau position
    w_list      - list of w's probed
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
#    wtlist=np.hstack([w_list,w_list])
    wtlist=w_list
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
    #kfki=calc_kfki(w_list,eief,efixed)


    eig_list=[]
#    print qlist
    eigs = Hsave.eigenvals().keys()
    for q in qlist:
        #eigs = calc_eigs_direct(Hsave,q[:,0],q[:,1],q[:,2])
        #eigs = Hsave.eigenvals().keys()
        print eigs
        eig_list.append(eigs)
    eig_list = np.array(eig_list)
    print "Calculated: Eigenvalues"
    
    
 #   print len(qlist)
 #   print len(eig_list[0])
#    sys.exit()

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
    lifetime=0.5#sp.Symbol('V',real=True)
    boltz = 8.617343e-2

    # Form Factor

    ff_list = []
    for i in range(N_atoms_uc):
        el = elements[atom_list[i].atomicNum]
        val = atom_list[i].valence
        if val != None:
            Mq = el.magnetic_ff[val].M_Q(kaprange)
        else:
            Mq = el.magnetic_ff[0].M_Q(kaprange)
        ff_list = Mq #ff_list.append(Mq)
    print "Calculated: Form Factors"
  

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
            
    print csection
    
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
    
    csection = (csection * exp(-I*w*t) * exp(I*kap*L)).expand(deep=False)
    csection = sp.powsimp(csection, deep=True)
    print 'beginning'
    print csection
    csection = sp.powsimp(csection)
    csection = sub_in(csection,exp(I*t*A + I*t*B + I*C + I*D + I*K),sp.DiracDelta(A*t + B*t + C + D + K))
    print 'intermediate'
    print csection
#    csection = sub_in(csection,sp.DiracDelta(A*t + B*t + C*L + D*L ),(1./hbar)*sp.DiracDelta(A + B)*sp.simplify(sp.DiracDelta(C + D  - tau)))  #This is correct
    csection = sub_in(csection,sp.DiracDelta(A*t + B*t + C*L + D*L ),sp.Pow(pi,-1)*(lifetime*0.5)*sp.Pow((A+B)**2+(lifetime*0.5)**2,-1)*sp.simplify(sp.DiracDelta(C + D  - tau)))
    print 'ending'
    print csection
    

    csection = sub_in(csection,sp.DiracDelta(-A - B),sp.DiracDelta(A + B))
    csection = sub_in(csection,(-A - B)**2,(A + B)**2)
    csection = csection.subs(sp.DiracDelta(Q+tau-kap),sp.DiracDelta(kap-Q-tau))
    csection = csection.subs(sp.DiracDelta(tau-kap-Q),sp.DiracDelta(kap+Q-tau))
    print "Applied: Delta Function Conversion"

    print "end part 1"
    #print csection
    return (N_atoms_uc, csection, kaprange, tau_list, eig_list, kapvect, wtlist, ff_list)
    
def eval_cross_section(N_atoms_uc, csection, kaprange, tau_list, eig_list, kapvect, wtlist,
                       fflist, temperature, eief = True, efixed = 14.7):
    """
    N_atoms_uc - number of atoms in unit cell
    csection - analytic cross-section expression
    kaprange - kappa modulus
    kapvect - list of kappa vectors
    tau_list - list of taus
    eig_list - list of eigenvalues
    wtlist - list of omegas
    fflist - magnetic form factor list
    temperature - temperature
    eief - True => E_initial = efixed, False => E_final = efixed
    efixed - fixed energy; either E_final or E_initial, subject to eief
    """
    
    
    print "begin part 2"

    # Kappa vector
    kap = sp.Symbol('kappa', real = True)
    t = sp.Symbol('t', real = True)
    w = sp.Symbol('w', real = True)
    tau = sp.Symbol('tau', real = True)
    Q = sp.Symbol('q', real = True)
    wq = sp.Symbol('wq', real = True)
    kapxhat = sp.Symbol('kapxhat',real=True)
    kapyhat = sp.Symbol('kapyhat',real=True)
    kapzhat = sp.Symbol('kapzhat',real=True)

    kapunit = kapvect.copy()
    kapunit[:,0]=kapvect[:,0]/kaprange
    kapunit[:,1]=kapvect[:,1]/kaprange
    kapunit[:,2]=kapvect[:,2]/kaprange    

    # Front constants and stuff for the cross-section
    boltz = 8.617343e-2
    gamr0 = 1.913*2.818
    hbar = 1#6.582*10**-13 
    g = 2.0
    temperature = temperature

    debye_waller = 1.0 #exp(-2*W)
    front_constant = ((gamr0**2)*debye_waller/(2*pi*hbar)).evalf() #(gamr0)**2#/(2*pi*hbar)
    print front_constant
    
    # kappa, omegas, tau, eigs
    temp1 = []
    for kapi in range(len(kapvect)):
        temp2 =[]
        for wi in range(len(wtlist)):
            temp3=[]
            for taui in range(len(tau_list)):
                print 'k,w,t',kapi,wi,taui
                ws=[]

                #qplus = kapvect[kapi] + tau_list[taui]
                #qmins = tau_list[taui] - kapvect[kapi]

                csectempp = deepcopy(csection)
                csectempm = deepcopy(csection)                    

                #qvalp = kapvect[kapi] - tau_list[taui] + qmins
                #qvalm = kapvect[kapi] - tau_list[taui] - qplus

                csectempp = csectempp.subs(sp.DiracDelta(kap - Q - tau),sp.S(1))
                csectempp = csectempp.subs(sp.DiracDelta(kap + Q - tau),sp.S(0))
                csectempm = csectempm.subs(sp.DiracDelta(kap - Q - tau),sp.S(0))
                csectempm = csectempm.subs(sp.DiracDelta(kap + Q - tau),sp.S(1))
#                print 'm'
#                print csectempm
#                print 'p'
#                print csectempp

                csectempp = csectempp.subs(kapxhat,kapunit[kapi,0])
                csectempp = csectempp.subs(kapyhat,kapunit[kapi,1])
                csectempp = csectempp.subs(kapzhat,kapunit[kapi,2])
                csectempm = csectempm.subs(kapxhat,kapunit[kapi,0])
                csectempm = csectempm.subs(kapyhat,kapunit[kapi,1])
                csectempm = csectempm.subs(kapzhat,kapunit[kapi,2])

                for eigi in range(len(eig_list[taui])):
                    eigcsecp=deepcopy(csectempp)
                    eigcsecm=deepcopy(csectempm)

#                    print 'a'
#                    print eigcsecp
#                    print eigcsecm

                    eigtemp = deepcopy(eig_list[0][eigi])

#                    print 'eig', eigtemp

                    spinmag = sp.Symbol('S', real = True)
                    kx = sp.Symbol('kx', real = True)
                    ky = sp.Symbol('ky', real = True)
                    kz = sp.Symbol('kz', real = True)

#                    print 'kappa'
#                    print kapvect[kapi][0]
#                    print kapvect[kapi][1]
#                    print kapvect[kapi][2]
                    eigtemp = eigtemp.subs(spinmag, sp.S(1.0))
                    eigtemp = eigtemp.subs(kx, kapvect[kapi][0])
                    eigtemp = eigtemp.subs(ky, kapvect[kapi][1])
                    eigtemp = eigtemp.subs(kz, kapvect[kapi][2])
                    eigtemp = sp.abs(eigtemp.evalf(chop=True))
#                    sp.Pow( sp.exp(-np.abs(eig_list[0][eigi])/boltz) - 1 ,-1) #\temperature term taken out

#                    print 'eig'
#                    print eigtemp.evalf()

#                    print 'a'
#                    print eigcsecp
#                    print eigcsecm


                    nval = sp.Pow(sp.exp(sp.abs(eigtemp)/(boltz*temperature))-1,-1).evalf()
                    for i in range(N_atoms_uc):
                        nq = sp.Symbol('n%i'%(i,), real = True)
                        eigcsecp = eigcsecp.subs(nq,nval)
                        eigcsecm = eigcsecm.subs(nq,nval) 

#                    print 'b'
#                    print eigcsecp
#                    print eigcsecm

                    wvalp = eigtemp - wtlist[wi]
                    wvalm = eigtemp + wtlist[wi]
#                    print 'w vals'
#                    print wvalp
#                    print wvalm

                    eigcsecp = eigcsecp.subs((w-wq),wvalp)
                    eigcsecp = eigcsecp.subs((w+wq),wvalm)
                    eigcsecp = sp.re(eigcsecp.evalf(chop = True))
                    eigcsecm = eigcsecm.subs((w-wq),wvalp)
                    eigcsecm = eigcsecm.subs((w+wq),wvalm)
                    eigcsecm = sp.re(eigcsecm.evalf(chop = True))

#                    if np.abs(wvalp) < 0.1: 
#                        eigcsecp = eigcsecp.subs(sp.DiracDelta(w - wq), sp.S(1))
#                        eigcsecm = eigcsecm.subs(sp.DiracDelta(w - wq), sp.S(0))
#                    else:
#                        eigcsecp = eigcsecp.subs(sp.DiracDelta(w - wq), sp.S(0))
#                        eigcsecm = eigcsecm.subs(sp.DiracDelta(w - wq), sp.S(0))
#                    if np.abs(wvalm) < 0.1:
#                        eigcsecp = eigcsecp.subs(sp.DiracDelta(w + wq), sp.S(0))
#                        eigcsecm = eigcsecm.subs(sp.DiracDelta(w + wq), sp.S(1)) 
#                    else:
#                        eigcsecp = eigcsecp.subs(sp.DiracDelta(w + wq), sp.S(0))
#                        eigcsecm = eigcsecm.subs(sp.DiracDelta(wq + w), sp.S(0))                                        
#
#                    print 'p'
#                    print eigcsecp
#                    print 'm'
#                    print eigcsecm

                    # eief == True => ei=efixed
                    # eief == False => ef=efixed
                    kpk = 0.
                    if eief == True:
                        ei = efixed
                        ef = ei - eigtemp
                        kpk = ef/ei
                    else:
                        ef = efixed
                        ei = ef + eigtemp
                        kpk = ef/ei


                    ws.append(kpk*(eigcsecp+eigcsecm))
                
                temp3.append(sum(ws))
            temp2.append(np.sum(np.array(temp3)))
        temp1.append(temp2)
    temp1 = np.array(temp1)

    csdata = np.array(sp.re(temp1.T))
    print csdata.shape

    #Multiply data by front constants
    #csdata = front_constant*csdata
    
    #Multiply by Form Factor
    print fflist.shape
    csdata = g*fflist*csdata

    return kapvect, wtlist, csdata

def eval_single_cross_section(N_atoms_uc, atom_list, theta, phi, rad, csection, kaprange, tau, eig_list, wt,
                       temperature, eief = True, efixed = 14.7):
    """
    N_atoms_uc - number of atoms in unit cell
    csection - analytic cross-section expression
    kaprange - kappa modulus
    kap - kappa vector
    tau - tau
    eig_list - list of eigenvalues
    wt - omega
    temperature - temperature
    eief - True => E_initial = efixed, False => E_final = efixed
    efixed - fixed energy; either E_final or E_initial, subject to eief
    """

    print "begin part 2"

    kx = rad*np.sin(theta)*np.cos(phi)
    ky = rad*np.sin(theta)*np.sin(phi)
    kz = rad*np.cos(theta)
    kap = np.array([kx,ky,kz])
    
    kapmod = np.sqrt(kx*kx+ky*ky+kz*kz)

    t = sp.Symbol('t', real = True)
    w = sp.Symbol('w', real = True)
    Q = sp.Symbol('q', real = True)
    wq = sp.Symbol('wq', real = True)
    kapxhat = sp.Symbol('kapxhat',real=True)
    kapyhat = sp.Symbol('kapyhat',real=True)
    kapzhat = sp.Symbol('kapzhat',real=True)

    kapunit = kap.copy()
    kapunit[0]=kap[0]/kapmod
    kapunit[1]=kap[1]/kapmod
    kapunit[2]=kap[2]/kapmod    

    # Front constants and stuff for the cross-section
    boltz = 8.617343e-2
    gamr0 = 1.913*2.818
    hbar = 1#6.582*10**-13 
    g = 2.0
    temperature = temperature

    debye_waller = 1.0 #exp(-2*W)
    front_constant = ((gamr0**2)*debye_waller/(2*pi*hbar)).evalf() #(gamr0)**2#/(2*pi*hbar)
    print front_constant

    ws=[]

    csectempp = deepcopy(csection)
    csectempm = deepcopy(csection)                    

    csectempp = csectempp.subs(sp.DiracDelta(kap - Q - tau),sp.S(1))
    csectempp = csectempp.subs(sp.DiracDelta(kap + Q - tau),sp.S(0))
    csectempm = csectempm.subs(sp.DiracDelta(kap - Q - tau),sp.S(0))
    csectempm = csectempm.subs(sp.DiracDelta(kap + Q - tau),sp.S(1))

    csectempp = csectempp.subs(kapxhat,kapunit[0])
    csectempp = csectempp.subs(kapyhat,kapunit[1])
    csectempp = csectempp.subs(kapzhat,kapunit[2])
    csectempm = csectempm.subs(kapxhat,kapunit[0])
    csectempm = csectempm.subs(kapyhat,kapunit[1])
    csectempm = csectempm.subs(kapzhat,kapunit[2])

    for eigi in range(len(eig_list)):
        eigcsecp=deepcopy(csectempp)
        eigcsecm=deepcopy(csectempm)

        eigtemp = deepcopy(eig_list[0][eigi])

        spinmag = sp.Symbol('S', real = True)
        kx = sp.Symbol('kx', real = True)
        ky = sp.Symbol('ky', real = True)
        kz = sp.Symbol('kz', real = True)

        eigtemp = eigtemp.subs(spinmag, sp.S(1.0))
        eigtemp = eigtemp.subs(kx, kap[0])
        eigtemp = eigtemp.subs(ky, kap[1])
        eigtemp = eigtemp.subs(kz, kap[2])
        eigtemp = sp.abs(eigtemp.evalf(chop=True))

        nval = sp.Pow(sp.exp(sp.abs(eigtemp)/(boltz*temperature))-1,-1).evalf()
        for i in range(N_atoms_uc):
            nq = sp.Symbol('n%i'%(i,), real = True)
            eigcsecp = eigcsecp.subs(nq,nval)
            eigcsecm = eigcsecm.subs(nq,nval) 

        wvalp = eigtemp - wt
        wvalm = eigtemp + wt

        eigcsecp = eigcsecp.subs((w-wq),wvalp)
        eigcsecp = eigcsecp.subs((w+wq),wvalm)
        eigcsecp = sp.re(eigcsecp.evalf(chop = True))
        eigcsecm = eigcsecm.subs((w-wq),wvalp)
        eigcsecm = eigcsecm.subs((w+wq),wvalm)
        eigcsecm = sp.re(eigcsecm.evalf(chop = True))

        # eief == True => ei=efixed
        # eief == False => ef=efixed
        kpk = 0.
        if eief == True:
            ei = efixed
            ef = ei - eigtemp
            kpk = ef/ei
        else:
            ef = efixed
            ei = ef + eigtemp
            kpk = ef/ei

        ws.append(kpk*(eigcsecp+eigcsecm))
    
    csdata = sp.re(sum(ws))

    #Multiply data by front constants
    #csdata = front_constant*csdata

    # Form Factor

    ff = 0
    for i in range(N_atoms_uc):
        el = elements[atom_list[i].atomicNum]
        val = atom_list[i].valence
        if val != None:
            Mq = el.magnetic_ff[val].M_Q(N_atoms_uc)
        else:
            Mq = el.magnetic_ff[0].M_Q(N_atoms_uc)
        ff = Mq #ff_list.append(Mq)

    #Multiply by Form Factor
    csdata = fflist*csdata
    
    result = integrate_over_sphere(csdata)

    return result

def spherical_averaging(N_atoms_uc, atom_list, rad, csection, kaprange, tau, eig_list, wt,
                       temperature, eief = True, efixed = 14.7):
    """
    N_atoms_uc - number of atoms in unit cell
    csection - analytic cross-section expression
    kaprange - kappa modulus
    kap - kappa vector
    tau - tau
    eig_list - list of eigenvalues
    wt - omega
    temperature - temperature
    eief - True => E_initial = efixed, False => E_final = efixed
    efixed - fixed energy; either E_final or E_initial, subject to eief
    """


    # theta = y, phi = x
    # y comes first (i.e. f(y,x,(args))
    def cross_section_calc(theta, phi, rad, N_atoms_uc, atom_list, csection, kaprange, tau, eig_list, wt,
                       temperature, eief = True, efixed = 14.7):

        # Front constants and stuff for the cross-section
        boltz = 8.617343e-2
        gamr0 = 1.913*2.818
        hbar = 1#6.582*10**-13 
        g = 2.0
        temperature = temperature
    
        w = sp.Symbol('w', real = True)
        Q = sp.Symbol('q', real = True)
        wq = sp.Symbol('wq', real = True)
        kapxhat = sp.Symbol('kapxhat',real=True)
        kapyhat = sp.Symbol('kapyhat',real=True)
        kapzhat = sp.Symbol('kapzhat',real=True)
        
        debye_waller = 1.0 #exp(-2*W)
        front_constant = ((gamr0**2)*debye_waller/(2*pi*hbar)).evalf() #(gamr0)**2#/(2*pi*hbar)
        print front_constant
    
        spinmag = sp.Symbol('S', real = True)
        kx = sp.Symbol('kx', real = True)
        ky = sp.Symbol('ky', real = True)
        kz = sp.Symbol('kz', real = True)


        kx = rad*np.sin(theta)*np.cos(phi)
        ky = rad*np.sin(theta)*np.sin(phi)
        kz = rad*np.cos(theta)
        kap = np.array([kx,ky,kz])
        
        kapmod = np.sqrt(kx*kx+ky*ky+kz*kz)
    
        kapunit = kap.copy()
        kapunit[0]=kap[0]/kapmod
        kapunit[1]=kap[1]/kapmod
        kapunit[2]=kap[2]/kapmod    
    
        ws=[]
    
        csectempp = deepcopy(csection)
        csectempm = deepcopy(csection)                    
    
        csectempp = csectempp.subs(sp.DiracDelta(kap - Q - tau),sp.S(1))
        csectempp = csectempp.subs(sp.DiracDelta(kap + Q - tau),sp.S(0))
        csectempm = csectempm.subs(sp.DiracDelta(kap - Q - tau),sp.S(0))
        csectempm = csectempm.subs(sp.DiracDelta(kap + Q - tau),sp.S(1))
    
        csectempp = csectempp.subs(kapxhat,kapunit[0])
        csectempp = csectempp.subs(kapyhat,kapunit[1])
        csectempp = csectempp.subs(kapzhat,kapunit[2])
        csectempm = csectempm.subs(kapxhat,kapunit[0])
        csectempm = csectempm.subs(kapyhat,kapunit[1])
        csectempm = csectempm.subs(kapzhat,kapunit[2])
    
        for eigi in range(len(eig_list)):
            eigcsecp=deepcopy(csectempp)
            eigcsecm=deepcopy(csectempm)
    
            eigtemp = deepcopy(eig_list[0][eigi])
    
            eigtemp = eigtemp.subs(spinmag, sp.S(1.0))
            eigtemp = eigtemp.subs(kx, kap[0])
            eigtemp = eigtemp.subs(ky, kap[1])
            eigtemp = eigtemp.subs(kz, kap[2])
            eigtemp = sp.abs(eigtemp.evalf(chop=True))
    
            nval = sp.Pow(sp.exp(sp.abs(eigtemp)/(boltz*temperature))-1,-1).evalf()
            for i in range(N_atoms_uc):
                nq = sp.Symbol('n%i'%(i,), real = True)
                eigcsecp = eigcsecp.subs(nq,nval)
                eigcsecm = eigcsecm.subs(nq,nval) 
    
            wvalp = eigtemp - wt
            wvalm = eigtemp + wt
    
            eigcsecp = eigcsecp.subs((w-wq),wvalp)
            eigcsecp = eigcsecp.subs((w+wq),wvalm)
            eigcsecp = sp.re(eigcsecp.evalf(chop = True))
            eigcsecm = eigcsecm.subs((w-wq),wvalp)
            eigcsecm = eigcsecm.subs((w+wq),wvalm)
            eigcsecm = sp.re(eigcsecm.evalf(chop = True))
    
            # eief == True => ei=efixed
            # eief == False => ef=efixed
            kpk = 0.
            if eief == True:
                ei = efixed
                ef = ei - eigtemp
                kpk = ef/ei
            else:
                ef = efixed
                ei = ef + eigtemp
                kpk = ef/ei
    
            ws.append(kpk*(eigcsecp+eigcsecm))
        
        csdata = sp.re(sum(ws))
    
        #Multiply data by front constants
        #csdata = front_constant*csdata
    
        # Form Factor
    
        ff = 0
        for i in range(N_atoms_uc):
            el = elements[atom_list[i].atomicNum]
            val = atom_list[i].valence
            if val != None:
                Mq = el.magnetic_ff[val].M_Q(N_atoms_uc)
            else:
                Mq = el.magnetic_ff[0].M_Q(N_atoms_uc)
            ff = Mq #ff_list.append(Mq)
    
        #Multiply by Form Factor
        csdata = g*ff*csdata
        
        return csdata*np.sin(theta)*rad**2

    plimL = 0.0 
    plimU = 2*np.pi
    tlimL = lambda x: 0.0
    tlimU = lambda x: np.pi
    
    return dblquad(cross_section_calc, plimL, plimU, tlimL, tlimU, 
                   args=(rad, N_atoms_uc, atom_list, csection, kaprange, 
                         tau, eig_list, wt, temperature, eief, efixed))


def plot_cross_section(xi, wtlist, csdata):
    xi = xi # kapvect[:,0]
    yi = wtlist
    zi = np.array(csdata,'Float64')
    
    zmin, zmax = np.min(zi), np.max(zi)
    if 1:
        locator = ticker.MaxNLocator(10) # if you want no more than 10 contours
        locator.create_dummy_axis()
        locator.set_bounds(zmin, zmax)
        levs = locator()
        levs[0]=1.0
    #print zmin, zmax
    plt.contourf(xi,yi,zi, levs)
  
    l_f = ticker.LogFormatter(10, labelOnlyBase=False)
    cbar = plt.colorbar(ticks = levs, format = l_f)

    plt.show()    
    
def run_cross_section(interactionfile, spinfile):
    start = clock()

    # Generate Inputs
    atom_list, jnums, jmats,N_atoms_uc=readFiles(interactionfile,spinfile)
    
    atom_list=atom_list[:N_atoms_uc]
    N_atoms = len(atom_list)

    kx = sp.Symbol('kx', real = True)
    ky = sp.Symbol('ky', real = True)
    kz = sp.Symbol('kz', real = True)
    k = spm.Matrix([kx,ky,kz])
    
    (b,bd) = generate_b_bd_operators(atom_list)
    (a,ad) = generate_a_ad_operators(atom_list, k, b, bd)
    (Sp,Sm) = generate_Sp_Sm_operators(atom_list, a, ad)
    (Sa,Sb,Sn) = generate_Sa_Sb_Sn_operators(atom_list, Sp, Sm)
    (Sx,Sy,Sz) = generate_Sx_Sy_Sz_operators(atom_list, Sa, Sb, Sn)
    print ''
    
    #Ham = generate_Hamiltonian(N_atoms, atom_list, b, bd)
    ops = generate_possible_combinations(atom_list, [Sx,Sy,Sz])
    ops = holstein(atom_list, ops)
    ops = apply_commutation(atom_list, ops)
    ops = replace_bdb(atom_list, ops)

    ops = reduce_options(atom_list, ops)
    list_print(ops)
    
    print "prelims complete. generating cross-section","\n"

    aa = bb = cc = np.array([2.0*np.pi], 'Float64')
    alpha = beta = gamma = np.array([np.pi/2.0], 'Float64')
    vect1 = np.array([[1,0,0]])
    vect2 = np.array([[0,0,1]])
    lattice = Lattice(aa, bb, cc, alpha, beta, gamma, Orientation(vect1, vect2))
    
    tau_list = []
    for i in range(1):
        tau_list.append(np.array([0,0,0], 'Float64'))

    h_list = np.linspace(0.1,6.3,50)
    k_list = np.zeros(h_list.shape)
    l_list = np.zeros(h_list.shape)
    
    w_list = np.linspace(-10,10,50)

    (N_atoms_uc,csection,kaprange,
     tau_list,eig_list,kapvect,wtlist,fflist) = generate_cross_section(interactionfile, spinfile, lattice, ops, 
                                                                tau_list, h_list, k_list, l_list, w_list)
    print csection

    return N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wtlist,fflist
    end = clock()
    print "\nFinished %i atoms in %.2f seconds" %(N_atoms,end-start)
    
def run_eval_cross_section(N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wtlist,fflist):
    temperature = 0.0001
    steps = 25
    efixed = 14.7 #meV
    eief = True
    
    kapvect, wtlist, csdata = eval_cross_section(N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wtlist,fflist,temperature,eief=eief,efixed=efixed)
    
    return kapvect,wtlist,csdata

#---------------- MAIN --------------------------------------------------------- 


if __name__=='__main__':



    interfile = r'c:/montecarlo.txt'#'c:/Users/Bill/Documents/montecarlo.txt'#'C:/eig_test_montecarlo.txt'
    spinfile = r'c:/spins.txt'#'c:/Users/Bill/Documents/spins.txt'#'C:/eig_test_Spins.txt'
    
    N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wtlist,fflist = run_cross_section(interfile,spinfile)
#    left_conn, right_conn = Pipe()
#    p = Process(target = create_latex, args = (right_conn, csection, "Cross-Section"))
#    p.start()
#    eig_frame = LaTeXDisplayFrame(self.parent, p.pid, left_conn.recv(), 'Cross-Section')
#    self.process_list.append(p)
#    p.join()
#    p.terminate()

    #kapvect,wtlist,csdata=run_eval_cross_section(N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wtlist,fflist)
    #plot_cross_section(kapvect[:,0],wtlist,csdata)
    
