"""
To run from this file, first make sure all dependencies are there. For the current SVN release, you can comment out
*Pyro.core
*Pyro.errors
*Pyro.core.initClient()

Next, assuming you have an interaction file and a spins file, use the method
    run_cross_section.
This will return an expression for the cross_section amongst a plethora of lists, symbols, etc. used for evaluation. 

Then use the method
    run_eval_pointwise
which takes most of the stuff returned from run_cross_section. This will return a list of q (x values), omega (y values)
and cross-section values (z-values). These can be plotted using a contour plot.

Lastly, and optionally, you can use the method plot_cross_section to plot the results. You may need to tweak this for your
needs. You can also save the arrays of data you have generated to .npy files using the save_cs_files method. This is highly
recommended when using spherical averaging. 

The system is broken up into the two different "drivers" so that you can calculate the cross-section expression and then
do whatever you want with it in multiple-processes. Just note that mostly everything besides the cross-section expression 
generation is highly dependent on that expression. 

"""

import sys
import os
#sys.path.append(r"/home/npatel/python_code/spinwaves_git")
import sympy as sp
import sympy.matrices as spm
from sympy import I,pi,exp,oo,sqrt,abs,S,Pow,re,Symbol,Wild
from sympy.physics.paulialgebra import delta
from sympy.core.cache import clear_cache
import numpy as np
from numpy import arctan2,sin,cos,array
from numpy import newaxis as nax
from scipy.integrate import simps, trapz, fixed_quad

import matplotlib
matplotlib.use('WXAgg')
import pylab
from matplotlib._pylab_helpers import Gcf
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from spinwaves.cross_section.util.subin import sub_in
from spinwaves.spinwavecalc.readfiles import atom, readFiles
from spinwaves.spinwavecalc.spinwave_calc_file import calculate_dispersion, calc_eigs_direct
from rescalculator.lattice_calculator import Lattice, Orientation
from periodictable import elements

from multiprocessing import Pipe, Process
from copy import deepcopy,copy
from timeit import default_timer as clock

#from cluster.william_mapper import Mapper
#
#import Pyro.core
#import Pyro.errors
#Pyro.core.initClient()
#
#import psyco
#psyco.log()
#psyco.full(memory=1000000)
#psyco.profile(0.05, memory=100)
#psyco.profile(0.2)

#------------ GLOBAL VARIABLES ---------------------------------------
"""
 ALL GLOBACL VARIABLES ARE CAPTIALIZED
 ALL GLOBAL SYMBOLS WILL BE IN CAPS, FOLLOWED BY _SYM.
 ALL OTHER SYMBOLS ARE USUALLY CONSTRUCTED WITH 'sym%i'%(num,)
"""

# define global variables
T_SYM = Symbol('t', real = True) # t
L_SYM = Symbol('L', real = True) # L
Q_SYM = Symbol('q', real = True) # q
QP_SYM = Symbol('qp', real = True) # qp
W_SYM = Symbol('w', real = True)
WQ_SYM = Symbol('wq', real = True) # wq
WQP_SYM = Symbol('wqp', real = True) #wqp
WT_SYM = Symbol('wt', real = True)
S_SYM = Symbol('S', commutative = True) # S
NQ_SYM = Symbol('nq', real = True) #nq = n0, n1, n2, ...

KAP_SYM = Symbol('kappa', real = True) # kap
TAU_SYM = Symbol('tau', real = True) # tau
KX_SYM = Symbol('kx', real = True)
KY_SYM = Symbol('ky', real = True)
KZ_SYM = Symbol('kz', real = True)
DD_KTPQ_SYM = Symbol('DDKTPQ', real = True)
DD_KTMQ_SYM = Symbol('DDKTMQ', real = True)

THETA_SYM = sp.Symbol('theta', real = True)
PHI_SYM = sp.Symbol('phi', real = True)
EIG_SYM = sp.Symbol('eig', real = True)
RAD_SYM = sp.Symbol('rad', real = True)
FF_SYM = sp.Symbol('ff', real = True)

KAPXHAT_SYM = Symbol('kapxhat',real=True)
KAPYHAT_SYM = Symbol('kapyhat',real=True)
KAPZHAT_SYM = Symbol('kapzhat',real=True)

# Wilds for sub_in method
A_WILD = Wild('A',exclude = [0,T_SYM])
B_WILD = Wild('B',exclude = [0,T_SYM])
C_WILD = Wild('C')
D_WILD = Wild('D')
K_WILD = Wild('K')
    
GAMMA_R0_VALUE = 1.913*2.818
HBAR_VALUE = 1.0#6.582e-13 
G_VALUE = 2.
LIFETIME_VALUE = 0.5
BOLTZ_VALUE = 8.617343e-2
DEBYE_WALLER_VALUE = 1.0 #exp(-2*W)

#------------ HELPER METHODS ---------------------------------------------------

def list_print(lista):
    print 'printing...'
    for element in lista:
        print element
    print ''

def list_mult(lista, listb):
    "Defines a way to multiply two lists of the same length"
    if len(lista) != len(listb):
        print "lists not same length"
        return []
    else:
        temp = []
        for i in range(len(lista)):
            if isinstance(lista[i], int):
                temp.append(sp.powsimp((lista[i] * listb[i]).expand(deep=False)))
            elif isinstance(listb[i], int):
                temp.append(sp.powsimp((lista[i] * listb[i]).expand(deep=False)))
            else:
                temp.append(sp.powsimp((lista[i] * listb[i]).expand(deep=False)))
        return temp

def coeff(expr, term):
    "Returns the coefficient of a term given an expression"
    if isinstance(expr, int):
        return 0
    expr = sp.collect(expr, term)
    symbols = list(term.atoms(sp.Symbol))
    w = Wild("coeff", exclude = symbols)
    m = expr.match(w * term + sp.Wild("rest"))
    m2 = expr.match(w * term)
    res = False
    if m2:
        res = m2[w] * term == expr
    if m and not res:
        return m[w]
    #added the next two lines
    elif m2:
        return m2[w]
    
def coeff_bins(expr,bins):
    # chop up expr at '+'
    expr_list = sp.make_list(expr, sp.Add)
    retbins = np.zeros(len(bins),dtype=object)
    # get rid of None expressions
    if expr is None or isinstance(expr,int):
        return retbins
    #scan through expr
    for subexpr in expr_list:
        #see if it contains a bin element
        for i in range(len(bins)):
            curr_coeff = subexpr.as_coefficient(bins[i])
            if curr_coeff:
                retbins[i] += curr_coeff
    return retbins

#------------ CROSS SECTION CALC METHODS ---------------------------------------

# Lists of the b and b dagger operators
def generate_b_bd_operators(atom_list):
    """Generates b and b dagger operators"""
    b_list = []; bd_list = []
    N = len(atom_list)
    for i in range(N):
        b = Symbol('b%i'%(i,), commutative = False)
        bd = Symbol('bd%i'%(i,), commutative = False)
        b_list.append(b); bd_list.append(bd)

    print "Operators Generated: b, bd"
    return (b_list,bd_list)

# Generates the a and a dagger operators
def generate_a_ad_operators(atom_list, k, b_list, bd_list):
    """Generates a and a dagger operators"""
    a_list = []; ad_list = []
    a0_list = []; ad0_list = []
    N = len(atom_list)

    for i in range(N):
        a_list.append(exp(I*(QP_SYM*L_SYM - WQP_SYM*T_SYM)) * b_list[i])
        ad_list.append(exp(-I*(QP_SYM*L_SYM - WQP_SYM*T_SYM)) * bd_list[i])
        a0_list.append(exp(0) * b_list[i])
        ad0_list.append(exp(0) * bd_list[i])

    a = Pow(sp.sqrt(N),-1) * sum(a_list)
    ad = Pow(sp.sqrt(N),-1) * sum(ad_list)
    a0 = Pow(sp.sqrt(N),-1) * sum(a0_list)
    ad0 = Pow(sp.sqrt(N),-1) * sum(ad0_list)

    print "Operators Generated: a, ad"
    return (a, ad, a0, ad0)

# Generates the Sp and Sm operators
def generate_Sp_Sm_operators(atom_list, a, ad, a0, ad0):
    """Generates S+ and S- operators"""

    Sp = sqrt(2*S_SYM) * a
    Sm = sqrt(2*S_SYM) * ad
    Sp0 = sqrt(2*S_SYM) * a0
    Sm0 = sqrt(2*S_SYM) * ad0

    print "Operators Generated: Sp, Sm"
    return (Sp, Sm, Sp0, Sm0)

def generate_Sa_Sb_Sn_operators(atom_list, Sp, Sm, Sp0, Sm0):
    """Generates Sa, Sb, Sn operators"""

    Sa = ((1./2.)*(Sp+Sm)).expand()
    Sb = ((1./2.)*(1./I)*(Sp-Sm)).expand()
    Sn = (S_SYM - Pow(2*S_SYM,-1) * Sm.expand() * Sp.expand()).expand()
    Sa0 = ((1./2.)*(Sp0+Sm0)).expand()
    Sb0 = ((1./2.)*(1./I)*(Sp0-Sm0)).expand()
    Sn0 = (S_SYM - Pow(2*S_SYM,-1) * Sm0.expand() * Sp0.expand()).expand()
    
    print "Operators Generated: Sa, Sb, Sn"
    return (Sa, Sb, Sn, Sa0, Sb0, Sn0)

# Generates the Sx, Sy and Sz operators
def generate_Sx_Sy_Sz_operators(atom_list, Sa, Sb, Sn, Sa0, Sb0, Sn0):
    """Generates Sx, Sy and Sz operators"""
    Sx_list = []; Sy_list = []; Sz_list = []
    Sx0_list = []; Sy0_list = []; Sz0_list = []
    N = len(atom_list)

    loc_vect = spm.Matrix([Sa,Sb,Sn])
    loc_vect = loc_vect.reshape(3,1)
    loc_vect0 = spm.Matrix([Sa0,Sb0,Sn0])
    loc_vect0 = loc_vect0.reshape(3,1)

    for i in range(N):
        rotmat = sp.Matrix(atom_list[i].spinRmatrix)
        glo_vect = rotmat * loc_vect
        glo_vect0 = rotmat * loc_vect0

        Sx = sp.powsimp(glo_vect[0].expand())
        Sy = sp.powsimp(glo_vect[1].expand())
        Sz = sp.powsimp(glo_vect[2].expand())
        Sx_list.append(Sx)
        Sy_list.append(Sy)
        Sz_list.append(Sz)

        Sx0 = sp.powsimp(glo_vect0[0].expand())
        Sy0 = sp.powsimp(glo_vect0[1].expand())
        Sz0 = sp.powsimp(glo_vect0[2].expand())
        Sx0_list.append(Sx0)
        Sy0_list.append(Sy0)
        Sz0_list.append(Sz0)
          
    Sx_list.append(KAPXHAT_SYM)
    Sy_list.append(KAPYHAT_SYM)
    Sz_list.append(KAPZHAT_SYM)
    Sx0_list.append(KAPXHAT_SYM)
    Sy0_list.append(KAPYHAT_SYM)
    Sz0_list.append(KAPZHAT_SYM)
    
    print "Operators Generated: Sx, Sy, Sz"
    return (Sx_list,Sy_list,Sz_list,Sx0_list,Sy0_list,Sz0_list)

# Define a method that generates the possible combinations of operators
#def generate_possible_combinations(atom_list, alist):
def generate_possible_combinations(atom_list, op_list, op_list0):
    """This method returns the possible operator combinations from a list of operators"""
    # For a combination to be returned, the product must have an equal number of b
    # and b dagger operators. If not, they are rejected.
    res_list = []
    N = len(atom_list)
    
    op_list = np.array(op_list)
    op_list0 = np.array(op_list0)

    for i in range(len(op_list0)):
        vecti = op_list0[i,-1]
        for j in range(len(op_list)):
            vectj = op_list[j,-1]
            if cmp(vecti,vectj) == 0: delta = 1
            else: delta = 0

            res_list.append((op_list0[i,:-1]*op_list[j,:-1]).tolist() + [delta - vecti*vectj])
    
    res_list = map(sp.expand, res_list)
    
    print "Generated: Possible Operator Combinations"
    return res_list
 
def holstein(atom_list, arg):
    N = len(atom_list)
    arg = np.array(arg)

    for k in range(len(arg)):
        for i in range(N):
            Snew = atom_list[i].spinMagnitude
            
            #gets rid of time independent terms
            #works since arg[k][i] is an Add instance so args breaks it up at '+'s
            pieces = arg[k][i].args
            for piece in pieces:
                if not piece.has(T_SYM):
                    arg[k][i] = arg[k][i] - piece             

            coeffs = coeff_bins(arg[k][i],[S_SYM**2,S_SYM])
            S2coeff,Scoeff = coeffs[0],coeffs[1]
            if S2coeff and Scoeff:
                arg[k][i] = (S2coeff*Snew**2 + Scoeff*Snew)
            elif S2coeff and not Scoeff:
                arg[k][i] = (S2coeff*Snew**2)
            elif not S2coeff and Scoeff:
                arg[k][i] = (Scoeff*Snew)

    #removes all rows with zeros for each element
#    arg = arg[arg[:,:-1].any(axis=1)]
    
    print "Applied: Holstein"
    return arg.tolist()

def reduce_options(atom_list, arg):
    """
    Further reduces possible operator combinations by removing combinations if
    they are the negative of another combination or they are not time dependent
    (i.e. elastic scattering)
    """
#    new = []
#    N = len(atom_list)
#    for element in arg:
#        if str(element[0]).find('t') > 0:
#            new.append(element)
    new = arg

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
    """Replaces bdq*bq with nq when q = q'"""
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

            arg[k][i] = arg[k][i].subs(QP_SYM,Q_SYM).subs(WQP_SYM,WQ_SYM)

    print "Applied: bdq*bq Replacement"
    return arg

def generate_cross_section(interactionfile, spinfile, lattice, arg, 
                       tau_list, h_list, k_list, l_list, w_list, temp, eig_eps = 0.01):
    """
    Calculates the cross_section given the following parameters:
    interactionfile, spinfile - files to get atom data
    lattice         - Lattice object from tripleaxisproject
    arg             - reduced list of operator combinations
    tau_list        - list of tau position
    h_,k_,l_lists   - lists of scan positions in (h,k,l) space.
                    - create kappa vector with these.
    w_list          - list of w's probed
    temp            - temperature
    """

    # Read files, get atom_list and such
    atom_list, jnums, jmats,N_atoms_uc=readFiles(interactionfile,spinfile)

    # Get Hsave to calculate its eigenvalues
    N_atoms = len(atom_list)
    Hsave = calculate_dispersion(atom_list,N_atoms_uc,N_atoms,jmats,showEigs=False)
    print Hsave
    atom_list=atom_list[:N_atoms_uc]
    N_atoms = len(atom_list)
    N = N_atoms
    
    print "Calculated: Dispersion Relation"

    # Generate kappas from (h,k,l)
    kaprange=lattice.modvec(h_list,k_list,l_list, 'latticestar')
    nkpts=len(kaprange)
    kapvect=np.empty((nkpts,3),'Float64')
    kapvect[:,0]=h_list
    kapvect[:,1]=k_list
    kapvect[:,2]=l_list

    # Grabs the unit vectors from the back of the lists. 
    kapunit = kapvect.copy()
    kapunit[:,0]=kapvect[:,0]/kaprange
    kapunit[:,1]=kapvect[:,1]/kaprange
    kapunit[:,2]=kapvect[:,2]/kaprange

    unit_vect = []
    for i in range(len(arg)):
        unit_vect.append(arg[i].pop())
    #print unit_vect
    unit_vect = sum(unit_vect)

    # Generate qs from kappas and taus
    qlist=[]
    ones_list=np.ones((1,nkpts),'Float64')
    
    for tau in tau_list:
        taui=np.ones((nkpts,3),'Float64')
        taui[:,0]=ones_list*tau[0]
        taui[:,1]=ones_list*tau[1]
        taui[:,2]=ones_list*tau[2]
        kappa_minus_tau=kapvect-taui
        tau_minus_kappa=taui - kapvect
               
        qlist.append(np.vstack([kappa_minus_tau,tau_minus_kappa]))

    # Eigenvalues and omegas
    wtlist=w_list
    eig_list=[]

    eig_list = calc_eigs_numerically(Hsave,h_list,k_list,l_list)
    eig_list = np.abs(eig_list)
    eig_list = np.where(eig_list < eig_eps, eig_eps, eig_list)

    print eig_list.shape 

#    eigs = Hsave.eigenvals().keys()
#    for q in qlist:
#        eig_list.append(eigs)
#    eig_list = np.array(eig_list)
    
    print "Calculated: Eigenvalues"

    # Form Factor
    ff_list = []
    for i in range(N_atoms_uc):
        el = elements[atom_list[i].atomicNum]
        val = atom_list[i].valence
        if val != None:
            Mq = el.magnetic_ff[val].M_Q(kaprange)
        else:
            Mq = el.magnetic_ff[0].M_Q(kaprange)
        ff_list = Mq 
    print "Calculated: Form Factors"
    
    # Generate most general form of csection
    csection=0
    for i in range(len(arg)):
        for j in range(len(arg[i])):
            csection = (csection + arg[i][j]*unit_vect)
                
    for i in range(N):
        ni = sp.Symbol('n%i'%(i,), real = True)
        np1i = sp.Symbol('np1%i'%(i,), Real = True)
        csection.subs(ni+1,np1i)

    # start refining the cross-section
    csection = csection.expand()

    # note: xhat*yhat = xhat*zhat = yhat*zhat = 0
    csection = csection.subs(KAPXHAT_SYM*KAPYHAT_SYM,0)
    csection = csection.subs(KAPXHAT_SYM*KAPZHAT_SYM,0)
    csection = csection.subs(KAPYHAT_SYM*KAPZHAT_SYM,0)
    csection = csection.subs(KAPZHAT_SYM*KAPYHAT_SYM,0)
    csection = csection.subs(KAPZHAT_SYM*KAPXHAT_SYM,0)
    csection = csection.subs(KAPYHAT_SYM*KAPXHAT_SYM,0)
        
    #  multiply by exp(-iwt)
    # then we do the heart of the substituting which is taking 
    # things of the form integral_{-inf}^{+inf} exp(-i(ql-wt))exp(-iwt) -> delta(k+-q-t)delta(w-wt)
    # we just use a symbol for the delta(k+-q-t) since we don't actually evaluate them
    # we also use a lorentzian instead of a delta function for delta(w-wt) with lifetime (1/2), adjustable at top of file)
    csection = (csection * exp(-I * W_SYM * T_SYM) * exp(I * KAP_SYM * L_SYM)).expand(deep=False)
    csection = sp.powsimp(csection, deep=True)
    print 'beginning'
#    print csection
    csection = sp.powsimp(csection)
    csection = sub_in(csection,exp(I*T_SYM*A_WILD + I*T_SYM*B_WILD + I*C_WILD + I*D_WILD + I*K_WILD),sp.DiracDelta(A_WILD*T_SYM + B_WILD*T_SYM + C_WILD + D_WILD + K_WILD))
    print 'intermediate'
#    print csection
#    csection = sub_in(csection,sp.DiracDelta(A*t + B*t + C*L + D*L ),(1./hbar)*sp.DiracDelta(A + B)*sp.simplify(sp.DiracDelta(C + D  - tau)))  #This is correct
    csection = sub_in(csection,sp.DiracDelta(A_WILD*T_SYM + B_WILD*T_SYM + C_WILD*L_SYM + D_WILD*L_SYM ),sp.Pow(pi,-1)*(LIFETIME_VALUE*0.5)*sp.Pow((A_WILD+B_WILD)**2+(LIFETIME_VALUE*0.5)**2,-1)*sp.simplify(sp.DiracDelta(C_WILD + D_WILD  - TAU_SYM)))
    print 'ending'
#    print csection
    
    # Do some associative clean up to make it easier for later substitutions
    csection = sub_in(csection,sp.DiracDelta(-A_WILD - B_WILD),sp.DiracDelta(A_WILD + B_WILD))
    csection = sub_in(csection,(-A_WILD - B_WILD)**2,(A_WILD + B_WILD)**2)
    csection = csection.subs(sp.DiracDelta(Q_SYM + TAU_SYM - KAP_SYM),sp.DiracDelta(KAP_SYM - Q_SYM - TAU_SYM))
    csection = csection.subs(sp.DiracDelta(TAU_SYM - KAP_SYM - Q_SYM),sp.DiracDelta(KAP_SYM + Q_SYM - TAU_SYM))

    csection = csection.subs(sp.DiracDelta(KAP_SYM - Q_SYM - TAU_SYM),DD_KTMQ_SYM)
    csection = csection.subs(sp.DiracDelta(KAP_SYM + Q_SYM - TAU_SYM),DD_KTPQ_SYM)
    print "Applied: Delta Function Conversion"
    
    csection = csection.evalf(chop=True)
    csection = sp.re(csection)
    
    # SUB IN SYMBOL OR EXPRESSION
    for i in range(N_atoms_uc):
        ni = sp.Symbol('n%i'%(i,), real = True)
        print ni
        nq = Pow(sp.exp(abs(WQ_SYM)/(BOLTZ_VALUE*temperature))-1,-1)
        csection = csection.subs(ni,nq)
    
    print csection
    
    print "Generated: Analytic Cross-Section Expression"
    return (N_atoms_uc, csection, kaprange, tau_list, eig_list, kapvect, wtlist, ff_list)
    
def eval_cross_section(N_atoms_uc, csection, kaprange, tau_list, eig_list, kapvect, wtlist,
                       fflist, temperature, eief = True, efixed = 14.7):
    """
    Note: This method seems to be slower and returns less precise results then using the single-cross-section
    method inside some for loops. Thus, use of this method is not recommended anymore.
    
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
    
    print "Begin Numerical Evaluation of Cross-Section"

    kapunit = kapvect.copy()
    kapunit[:,0]=kapvect[:,0]/kaprange
    kapunit[:,1]=kapvect[:,1]/kaprange
    kapunit[:,2]=kapvect[:,2]/kaprange

    temperature = temperature
    front_constant = ((GAMMA_R0_VALUE**2)*DEBYE_WALLER_VALUE/(2*pi*HBAR_VALUE)).evalf() #(gamr0)**2#/(2*pi*hbar)
    print front_constant
    
    # kappa, omegas, tau, eigs
    temp1 = []
    for kapi in range(len(kapvect)):
        temp2 =[]
        for wi in range(len(wtlist)):
            temp3=[]
            for taui in range(len(tau_list)):
#                print 'k,w,t',kapi,wi,taui
                ws=[]

                #qplus = kapvect[kapi] + tau_list[taui]
                #qmins = tau_list[taui] - kapvect[kapi]

                csectempp = copy(csection)
                csectempm = copy(csection)                    

                #qvalp = kapvect[kapi] - tau_list[taui] + qmins
                #qvalm = kapvect[kapi] - tau_list[taui] - qplus

                csectempp = csectempp.subs(DD_KTMQ_SYM,sp.S(1))
                csectempp = csectempp.subs(DD_KTPQ_SYM,sp.S(0))
                csectempm = csectempm.subs(DD_KTMQ_SYM,sp.S(0))
                csectempm = csectempm.subs(DD_KTPQ_SYM,sp.S(1))
#                csectempp = csectempp.subs(sp.DiracDelta(KAP_SYM - Q_SYM - TAU_SYM),sp.S(1))
#                csectempp = csectempp.subs(sp.DiracDelta(KAP_SYM + Q_SYM - TAU_SYM),sp.S(0))
#                csectempm = csectempm.subs(sp.DiracDelta(KAP_SYM - Q_SYM - TAU_SYM),sp.S(0))
#                csectempm = csectempm.subs(sp.DiracDelta(KAP_SYM + Q_SYM - TAU_SYM),sp.S(1))
#                print 'm'
#                print csectempm
#                print 'p'
#                print csectempp

                csectempp = csectempp.subs(KAPXHAT_SYM,kapunit[kapi,0])
                csectempp = csectempp.subs(KAPYHAT_SYM,kapunit[kapi,1])
                csectempp = csectempp.subs(KAPZHAT_SYM,kapunit[kapi,2])
                csectempm = csectempm.subs(KAPXHAT_SYM,kapunit[kapi,0])
                csectempm = csectempm.subs(KAPYHAT_SYM,kapunit[kapi,1])
                csectempm = csectempm.subs(KAPZHAT_SYM,kapunit[kapi,2])

                for eigi in range(len(eig_list[taui])):
                    eigcsecp=copy(csectempp)
                    eigcsecm=copy(csectempm)

#                    print 'a'
#                    print eigcsecp
#                    print eigcsecm

                    eigtemp = copy(eig_list[0][eigi])

#                    print 'eig', eigtemp

#                    print 'kappa'
#                    print kapvect[kapi][0]
#                    print kapvect[kapi][1]
#                    print kapvect[kapi][2]
                    eigtemp = eigtemp.subs(S_SYM, sp.S(1.0))
                    eigtemp = eigtemp.subs(KX_SYM, kapvect[kapi][0])
                    eigtemp = eigtemp.subs(KY_SYM, kapvect[kapi][1])
                    eigtemp = eigtemp.subs(KZ_SYM, kapvect[kapi][2])
                    eigtemp = sp.abs(eigtemp.evalf(chop=True))
#                    sp.Pow( sp.exp(-np.abs(eig_list[0][eigi])/boltz) - 1 ,-1) #\temperature term taken out

#                    print 'eig'
#                    print eigtemp.evalf()

#                    print 'a'
#                    print eigcsecp
#                    print eigcsecm


                    nval = sp.Pow(sp.exp(sp.abs(eigtemp)/(BOLTZ_VALUE*temperature))-1,-1).evalf()
#                    nval = sp.S(0)
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

                    eigcsecp = eigcsecp.subs((W_SYM - WQ_SYM),wvalp)
                    eigcsecp = eigcsecp.subs((W_SYM + WQ_SYM),wvalm)
                    eigcsecp = sp.re(eigcsecp.evalf(chop = True))
                    eigcsecm = eigcsecm.subs((W_SYM - WQ_SYM),wvalp)
                    eigcsecm = eigcsecm.subs((W_SYM + WQ_SYM),wvalm)
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
    csdata = csdata*(0.5*G_VALUE*fflist)**2

    return kapvect, wtlist, csdata

#def chop(expr,tol=1e-8):
#    for item in expr.atoms():
#        if item < tol:
#            expr = expr.subs(item,0)
#    return expr

#@profile
def single_cross_section_calc(theta, phi, rad, atom_list, csection, tau, eigval, wt,
                       temperature, ffval, eief = True, efixed = 14.7, eps = 0.01):
    """
    This method uses the core of eval_cross_section. It takes a single value for wt, tau. Instead of kx,ky,kz lists,
    it takes a single of the following: theta, phi and radius. With them, it calculates kx, ky, kz and kappa. 
    
    Inputs:
    theta, phi, rad         - single values of each. used to create kx,ky,kz and kappa vector. 
    N_atoms_uc, atomlist,   - taken from readfiles(interactions,spins)
    csection                - takes the cross-section expression generated in generate_cross_section
    tau                     - single value for tau
    eig_list                - list of eigenvalues. calculated in generate_cross_section using spinwavecalcfile methods
    wt                      - single value of omega
    temperature             - temperature
    ffval                   - form factor term, taken from list of form factors calculated in generate_cross_section
    eief, efixed            - determines whether we want Ef/Ei. if true: ef/ei ef=ei-eigenval and ei=14.7 meV
    eps                     - this value determines the size of the epsilon neighborhood around 0 with which is used
                              to exclude values of omega which will return an infinite thermal averaged n_q value. 
    
    This method is faster than the eval_cross_section method. Use this one.
    
    """

    temperature = temperature

    # calculate front constant
    front_constant = ((GAMMA_R0_VALUE**2)*DEBYE_WALLER_VALUE/(2*pi*HBAR_VALUE)).evalf()

    # calculate kx,ky,kz and kappa
    kx = rad*sp.sin(theta)*sp.cos(phi)
    ky = rad*sp.sin(theta)*sp.sin(phi)
    kz = rad*sp.cos(theta)
    kap = array([kx,ky,kz])   

    # subs in kappa unit vectors
    csection = csection.subs(KAPXHAT_SYM,kx/rad)
    csection = csection.subs(KAPYHAT_SYM,ky/rad)
    csection = csection.subs(KAPZHAT_SYM,kz/rad)

    #make two copies of the cross-section, one + and one -
    csectempp = copy(csection)  #good
    csectempm = copy(csection)  #good     
    
    # get rid of the appropriate delta(k-t+-q) factor
    csectempp = csectempp.subs(DD_KTMQ_SYM,sp.S(1))
    csectempp = csectempp.subs(DD_KTPQ_SYM,sp.S(0))
    csectempm = csectempm.subs(DD_KTMQ_SYM,sp.S(0))
    csectempm = csectempm.subs(DD_KTPQ_SYM,sp.S(1))

    # get the kp/k factor
    # eief == True => ei=efixed
    # eief == False => ef=efixed

    # sum over all the different eigenvalues
    csdata = csectempp+csectempm

    cs_func = sp.lambdify((THETA_SYM, PHI_SYM, W_SYM, WQ_SYM), csdata, modules="sympy")
    csdata = cs_func(theta, phi, wt, eigval)
    
    if eief:
        ei = efixed
#            ef = ei - eigi
        ef = ei - eigval
    else:
        ef = efixed
#            ef = ei + eigi
        ei = ef + eigval
    kpk = ef/ei
    
    #Multiply data by front constants
    csdata = front_constant*kpk*csdata
    
    # Multiply by form factor
    csdata = (0.5*G_VALUE*ffval)**2*csdata
#    csdata = (0.5*G_VALUE)**2*csdata
    
    print "completed first time"
    print csdata
    return csdata


def spherical_averaging(rad, wt, tau, ffval, N_atoms_uc, atom_list, csection, eig_list,
                       temperature, eief = True, efixed = 14.7,thetas=None,phis=None,):
    """
    This method takes a single radius, omega and tau and calculates the spherical average at that point
    using the single_cross_section_calc method. 
    
    N_atoms_uc - number of atoms in unit cell
    csection - analytic cross-section expression
    kap - kappa vector
    tau - tau
    eig_list - list of eigenvalues
    wt - omega
    ffval - form factor value
    temperature - temperature
    eief - True => E_initial = efixed, False => E_final = efixed
    efixed - fixed energy; either E_final or E_initial, subject to eief
    """
    
    args=(rad, N_atoms_uc, atom_list, csection, tau, eig_list, wt, temperature, ffval, eief, efixed)

#    theta_test=np.pi/2.0
#    phi_test = np.pi/4.0

    if thetas == None:
        thetas = np.linspace(0,np.pi,25)
    if phis == None:
        phis = np.linspace(0,2*np.pi,25)
    cs_vals = []
    partial_res=[]
    
#    # Form Factor
#    ff = 0
#    for i in range(N_atoms_uc):
#        el = elements[atom_list[i].atomicNum]
#        val = atom_list[i].valence
#        if val != None:
#            Mq = el.magnetic_ff[val].M_Q(rad)
#        else:
#            Mq = el.magnetic_ff[0].M_Q(rad)
#        ff = Mq

    start1 = clock()
    val_func = lambda t,p: single_cross_section_calc(t,p,*args)*np.sin(t)*rad**2
    cs_vals = np.array([[val_func(t,p) for p in phis] for t in thetas])
    
#    t,p = sp.symbols('tp')
#    val_func1 = sp.lambdify((t,p), single_cross_section_calc(t,p,*args)*np.sin(t)*rad**2, modules = "numpy")
#    cs_vals1 = val_func1(phis[:,np.newaxis],thetas[np.newaxis,:])
    
#    for t in thetas:
#        temp_cs = []
#        for p in phis:
#            val = single_cross_section_calc(t,p,*args)*ff*np.sin(t)*rad**2
#            temp_cs.append(val)
#        cs_vals.append(temp_cs)
#    cs_vals = np.array(cs_vals)
    end1 = clock()
    calc_time = end1-start1

    start2 = clock()
    partial_res = np.array([simps(cs_vals[i],phis) for i in range(len(thetas))])
#    for i in range(len(thetas)):
#        single_res = simps(cs[i],phis)
#        partial_res.append(single_res)
    total_res = simps(partial_res,thetas)
    end2 = clock()
    inte_time = end2-start2
    
    print 'result', total_res
    print 'calc time', calc_time
    print 'inte time', inte_time
    return total_res

def plot_cross_section(xi, wtlist, csdata, colorbarFlag = True, minval = 0, maxval = 25):
    """
    Used for plotting. Needs to be more robust before release.
    Also, watch the limits for the intensity contours. Something is tricky with it and
    doesn't want to show values that are close to 0 as different from 0. 
    
    """
    xi = xi # kapvect[:,0]
    yi = wtlist
    zi = np.array(csdata,'Float64')
    
    zmin, zmax = np.min(zi), np.max(zi)
    if zmin < minval:
        print 'plotting clipped minimal extrema'
        zi = np.where(zi < minval, minval, zi)
    if zmax > maxval:
        print 'plotting clipped maximal extrema'
        zi = np.where(zi > maxval, maxval, zi)
    print zmin, zmax

    if colorbarFlag:
        locator = ticker.MaxNLocator(25)
        locator.create_dummy_axis()
        locator.set_bounds(minval, maxval)
        levs = locator()
        levs[0]=0.5
        plt.contourf(xi,yi,zi, levs)
        
        l_f = ticker.LogFormatter(10, labelOnlyBase=False)
        cbar = plt.colorbar(ticks = levs, format = l_f)
    else: 
        plt.contourf(xi,yi,zi)
        cbar = plt.colorbar()
    
    plt.show()
    
def save_cs_files(xarr,yarr,zarr,others):
    "Takes an x, y, and z array and saves them"

    file_pathname = os.path.abspath('')
    
    np.save(os.path.join(file_pathname,r'csx'),xarr)
    np.save(os.path.join(file_pathname,r'csy'),yarr)
    np.save(os.path.join(file_pathname,r'csz'),zarr)
    
    i=1
    for oth in others:
        np.save(os.path.join(file_pathname,r'cs_other_arr%i'%i),oth)
        i=i+1
    
    print "Files Saved"



def lambdify_expr(expr,syms,vals=None):
    """
    expr =  expr to lambdify
    syms =  symbolic arguments
    vals =  mixed tuple of symbols and values to substitute in for initial expression simplification
    """
    
    func = sp.lambdify(syms, expr, modules = "sympy")
    func2 = lambda syms: expr
    
    if vals:
        syms = list(syms)
        for i in range(len(syms)):
            if syms[i] not in vals:
                syms.insert(i,0)
                del syms[i+1]
        syms = tuple(filter(lambda x: x!=0,syms))
        
        func = sp.lambdify(syms,func(*vals),modules="sympy")
    return func
    
def calc_eigs_numerically(mat,h,k,l,S=1):
    """
    Give it a matrix, and the (h,k,l) values to substitute into that matrix, each in a separate list.
    S is automatically evaluated as one, but can be changed. h,k,l lists must be the same length.
    """
    #get rid of these
    S_SYM = sp.Symbol('S')
    KX_SYM = sp.Symbol('kx')
    KY_SYM = sp.Symbol('ky')
    KZ_SYM = sp.Symbol('kz')        

    #lambdification functionality
    syms = (S_SYM,KX_SYM,KY_SYM,KZ_SYM)
    matsym = mat.tolist()
    func = sp.lambdify(syms,matsym,modules=["sympy"])
    
    eigarr = []
    Slist = S*np.ones(h.shape)
    
    # reduce symbolic matrix to numerical matrix and calculate the eigenvalues
    for i in range(len(h)):
        eigmat = np.array(func(Slist[i],h[i],k[i],l[i]))
        
        # Convert numpy array to sympy matrix and lambdify it to
        # exchange sympy.I with numpy's 1j. Then convert it back to 
        # a numpy array and append it to the list of eigs. 
        eigmat = sp.Matrix(eigmat)
        I2jfunc = sp.lambdify((sp.I),eigmat,modules="numpy")
        eigmat = np.array(I2jfunc(1j))

        eigs,vects = np.linalg.eig(eigmat)
        eigarr.append(eigs)
    return np.array(eigarr)

def run_cross_section(interactionfile, spinfile, tau_list, temperature, 
                      direction=[1,0,0], hkl_interval=[1e-3,2*np.pi,1000], omega_interval=[0,5,1000]):
    """
    We use this method to generate the expression for the cross-section given just the interaction and spins file.
    *** Use this first to calculate cross-section before using any other methods as they all need the csection expression
    this generates.***
    
    I have the infrastructure in place for this "steps" and bounds for the scan like the dispersion calc has:
    direction       - of the form [h,k,l] where h,k,l are either 0 or 1. For example, a scan along h would be
                    [1,0,0].
    hkl_interval    - of the form [start,end,#of steps].
    omega_interval  - of the form [start,end,#of steps].
    """
    start = clock()

    # Generate Inputs
    atom_list, jnums, jmats,N_atoms_uc=readFiles(interactionfile,spinfile)
    
    atom_list=atom_list[:N_atoms_uc]
    N_atoms = len(atom_list)

    k = spm.Matrix([KX_SYM,KY_SYM,KZ_SYM])
    
    (b,bd) = generate_b_bd_operators(atom_list)
    (a,ad,a0,ad0) = generate_a_ad_operators(atom_list, k, b, bd)
    (Sp,Sm,Sp0,Sm0) = generate_Sp_Sm_operators(atom_list, a, ad, a0, ad0)
    (Sa,Sb,Sn,Sa0,Sb0,Sn0) = generate_Sa_Sb_Sn_operators(atom_list, Sp, Sm, Sp0, Sm0)
    (Sx,Sy,Sz,Sx0,Sy0,Sz0) = generate_Sx_Sy_Sz_operators(atom_list, Sa, Sb, Sn, Sa0, Sb0, Sn0)
    print ''
    
    #Ham = generate_Hamiltonian(N_atoms, atom_list, b, bd)
    ops = generate_possible_combinations(atom_list, [Sx,Sy,Sz], [Sx0,Sy0,Sz0])
    ops = holstein(atom_list, ops)
    ops = apply_commutation(atom_list, ops)
    ops = replace_bdb(atom_list, ops)
    ops = reduce_options(atom_list, ops)

#    list_print(ops)
    
    print "prelims complete. generating cross-section","\n"
    
   

    aa = bb = cc = np.array([2.0*np.pi], 'Float64')
    alpha = beta = gamma = np.array([np.pi/2.0], 'Float64')
    vect1 = np.array([[1,0,0]])
    vect2 = np.array([[0,0,1]])
    lattice = Lattice(aa, bb, cc, alpha, beta, gamma, Orientation(vect1, vect2))
    
    tau_list = tau_list

    if direction != None and hkl_interval != None:
        if direction[0]:
            h_list = np.linspace(hkl_interval[0],hkl_interval[1],hkl_interval[2])
        else: h_list = np.zeros((hkl_interval[2],))
        if direction[1]:
            k_list = np.linspace(hkl_interval[0],hkl_interval[1],hkl_interval[2])
        else: k_list = np.zeros((hkl_interval[2],))
        if direction[2]:
            l_list = np.linspace(hkl_interval[0],hkl_interval[1],hkl_interval[2])
        else: l_list = np.zeros((hkl_interval[2],))
    
    if omega_interval:
        w_list = np.linspace(omega_interval[0],omega_interval[1],omega_interval[2])

    (N_atoms_uc,csection,kaprange,
     tau_list,eig_list,kapvect,wtlist,fflist) = generate_cross_section(interactionfile, spinfile, lattice, ops, 
                                                                tau_list, h_list, k_list, l_list, w_list, temperature)
  
    end = clock()
    print ' TIME TIME TIME ', end-start
    
    
    return N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wtlist,fflist
#    end = clock()
#    print "\nFinished %i atoms in %.2f seconds" %(N_atoms,end-start)
    
def run_eval_cross_section(N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wtlist,fflist):
    """
    This method runs the eval_cross_section method given values calculated from run_cross_section. 
    I would say don't use this since eval_cross_section is slower than the other method and not as precise. 
    """
    
    temperature = 0.0001
    steps = 25
    efixed = 14.7 #meV
    eief = True
    
    kapvect, wtlist, csdata = eval_cross_section(N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wtlist,fflist,temperature,eief=eief,efixed=efixed)
    
    return kapvect,wtlist,csdata

def run_eval_pointwise(N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wtlist,fflist,temp):
    """
    This method uses the single_cross_section_calc method to evaluate the csection expression. We have the method
    wrapped in some for loops and it appears faster and more precise than run_eval_cross_section. 
    
    *** USE THIS AFTER generate_cross_section ***
    
    Takes values that are returned from run_cross_section. Returns values to be used to plot results - basically: x,y,z
    where x.shape = (n,0), y.shape = (m,0) and z.shape = (n,m)
    """
    
    h_list = kapvect[:,0]
    k_list = kapvect[:,1]
    l_list = kapvect[:,2]
    w_list = wtlist
    temperature = temp
    
    rad_list = np.array(kaprange)
    theta_list = np.array(np.arccos(l_list/rad_list))
    phi_list = np.array(arctan2(k_list,h_list))
    
    tau_list = np.array(tau_list)
    w_list = np.array(w_list)
    
    print "Computing Numerical Evaluation of Cross-section"

#    THETA,PHI,RAD,FF_SYM = sp.Symbol('THETA'),sp.Symbol('PHI'),sp.Symbol('RAD'),sp.Symbol('ff')

    print 'Generating Expression'
    
    csection_func_expr = single_cross_section_calc(THETA_SYM ,PHI_SYM, RAD_SYM, atom_list, csection, TAU_SYM, EIG_SYM,
                                                          WT_SYM, temperature, FF_SYM)
    
    print 'CCC',csection_func_expr
    print 'Evaluating Expression'
    
    csection_func = sp.lambdify((THETA_SYM ,PHI_SYM ,RAD_SYM ,TAU_SYM, EIG_SYM, WT_SYM, FF_SYM),csection_func_expr, modules="numpy")
    
    print 'Generating Array'
    
    values = csection_func(theta_list[:,nax,nax,nax],phi_list[:,nax,nax,nax],rad_list[:,nax,nax,nax],
                           tau_list[nax,nax,:,nax],eig_list[:,:,nax,nax],wtlist[nax,nax,nax,:],fflist[:,nax,nax,nax]).sum(axis=1).sum(axis=1)
    
    print 'Complete'
    print values.shape
    
    """
    final_vals=[]
    for wti in wt_list:
        inner_vals=[]
        for i in range(len(rad_list)):
            inner_vals2=[]
            for tau in tau_list:

                val = single_cross_section_calc(theta_list[i], phi_list[i], rad_list[i], N_atoms_uc, atom_list, csection, tau, eig_list, wti,
                                                 temperature, fflist[i], eief = True, efixed = 14.7)
                inner_vals2.append(val)
            inner_vals.append(sum(inner_vals2))
        final_vals.append(inner_vals)
    final_vals = array(final_vals)
    
    return h_list, w_list, final_vals
    """
    
    return h_list, w_list, values.T

def run_spherical_averaging(N_atoms_uc,atom_list,rad,csection,kapvect,tau_list,eig_list,wt_list,fflisttemperature):
    """
    This method runs the spherical averaging method to calculate the spherically averaged scattering cross_section. 
    
    Currently, results are iffy and slow. We are working on compatiability with compufans to speed things up. We could
    also use pyro to take advantage of multiple cores on a single machine. 
    
    """
#    theta_list=[]
#    phi_list=[]
    #mapper = Pyro.core.getProxyForURI("PYRONAME://:Mapper.dispatcher")
    rad_list=[]
    for kappa in kapvect:
        kx,ky,kz = kappa[0],kappa[1],kappa[2]
        r = np.sqrt(kx*kx+ky*ky+kz*kz)
#        theta_list.append(np.arccos(kz/r))
#        phi_list.append(np.arctan2(ky,kx))
        rad_list.append(r)
    
    aa = bb = cc = np.array([2.0*np.pi], 'Float64')
    alpha = beta = gamma = np.array([np.pi/2.0], 'Float64')
    vect1 = np.array([[1,0,0]])
    vect2 = np.array([[0,0,1]])
    lattice = Lattice(aa, bb, cc, alpha, beta, gamma, Orientation(vect1, vect2))
    
    same_args = (N_atoms_uc, atom_list, csection, eig_list, temperature)
    res_array = []
    rand_wt_list = np.append(wt_list[::2],wt_list[1::2])
#    rand_wt_list = wt_list.copy()
#    np.random.shuffle(rand_wt_list)
    xvals = np.array(rad_list)
#    xvals = np.array(xvals)
    yvals = []
    
    
####### WORK ON GETTING THIS EXPReSSION INTO THE SPHERICAL AVERAGING METHOD    
    
    THETA,PHI,RAD,FF_SYM = sp.Symbol('THETA'),sp.Symbol('PHI'),sp.Symbol('RAD'),sp.Symbol('ff')

    csection_func_expr = single_cross_section_calc(THETA,PHI,RAD,N_atoms_uc,atom_list, csection, TAU_SYM, eig_list,
                                                          WT_SYM, temperature,FF_SYM)
    
    csection_func = sp.lambdify((THETA,PHI,RAD,TAU_SYM,WT_SYM,FF_SYM),csection_func_expr, modules="numpy")
    
    values = csection_func(theta_list[:,nax,nax],phi_list[:,nax,nax],rad_list[:,nax,nax],
                           tau_list[nax,:,nax],wtlist[nax,nax,:],fflist[:,nax,nax]).sum(axis=1)


    r,w,t = sp.symbols('rwt')
    
    func = sp.lambdify((r,w,t,same_args), spherical_averaging(r,w,t,1.0,*same_args))
    func(rad_list,wt_list,tau_list,same_args)
    
#    for wt in wt_list:
#    #for wt in rand_wt_list:
#        yvals.append(wt)
#        
#        partial_res_array=[]
#        for rad in rad_list:
#            zval = 0
#            ffval = fflist[rad_list.index(rad)]
#            for tau in tau_list:
#                val = spherical_averaging(rad,wt,tau,ffval,*same_args)
#                print 'actual result',val
#                zval = zval + val
#            partial_res_array.append(zval)
#            
#        res_array.append(partial_res_array)
#        
#    res_array = np.array(res_array)
    print res_array
    return xvals,yvals,res_array.T

def correction_driver(interactionfile, spinfile, tau_list, temperature, 
                      direction=[1,0,0], hkl_interval=[1e-3,2*np.pi,1000], omega_interval=[0,5,1000]):
    import csection_calc as csc
    
    atom_list, jnums, jmats,N_atoms_uc=readFiles(interactionfile,spinfile)
    
    atom_list=atom_list[:N_atoms_uc]
    N_atoms = len(atom_list)

    k = spm.Matrix([KX_SYM,KY_SYM,KZ_SYM])
    
    (b,bd) = generate_b_bd_operators(atom_list)
    (a,ad,a0,ad0) = generate_a_ad_operators(atom_list, k, b, bd)
    (Sp,Sm,Sp0,Sm0) = generate_Sp_Sm_operators(atom_list, a, ad, a0, ad0)
    (Sa,Sb,Sn,Sa0,Sb0,Sn0) = generate_Sa_Sb_Sn_operators(atom_list, Sp, Sm, Sp0, Sm0)
    (Sx,Sy,Sz,Sx0,Sy0,Sz0) = generate_Sx_Sy_Sz_operators(atom_list, Sa, Sb, Sn, Sa0, Sb0, Sn0)

    ops = generate_possible_combinations(atom_list, [Sx,Sy,Sz], [Sx0,Sy0,Sz0])
    ops = holstein(atom_list, ops)
    ops = apply_commutation(atom_list, ops)
    ops = replace_bdb(atom_list, ops)
    ops1 = reduce_options(atom_list, ops)
    ops2 = csc.reduce_options(atom_list, ops)

    ops1 = np.array(ops1)
    ops2 = np.array(ops2)
    print ops1-ops2
    print ops1
    print ops2

    

#---------------- MAIN --------------------------------------------------------- 

## Methodized version of MAIN 
#def cs_driver():
#    file_pathname = os.path.abspath('')
#    interfile = os.path.join(file_pathname,r'montecarlo.txt')
#    spinfile = os.path.join(file_pathname,r'spins.txt')
#
#    h_list = np.linspace(0.001,3.14,15)
#    k_list = np.zeros(h_list.shape)
#    l_list = np.zeros(h_list.shape)
#    w_list = np.linspace(0,5,15)
#    tau = np.array([0,0,0])
#    wt = np.array(1.0)
#    rad = 1.0
#
#    atom_list, jnums, jmats,N_atoms_uc=readFiles(interfile,spinfile)
#    N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wt_list,fflist = run_cross_section(interfile,spinfile)
#    
#    x,y,z=run_spherical_averaging(N_atoms_uc,atom_list,rad,csection,kapvect,tau_list,eig_list,wt_list,temperature)
#    np.save(os.path.join(file_pathname,r'myfilex.txt'),x)
#    np.save(os.path.join(file_pathname,r'myfiley.txt'),y)
#    np.save(os.path.join(file_pathname,r'myfilez.txt'),z)

if __name__=='__main__':
#def pd():
    #from spinwaves.cross_section.csection_calc import spherical_averaging as sph
    
    ST = clock()
    
    file_pathname = os.path.abspath('')
    print file_pathname
    if 0: # YANG
        spinfile=r'C:/Documents and Settings/wflynn/Desktop/spins.txt'
        interfile=r'C:/Documents and Settings/wflynn/Desktop/yang_montecarlo.txt'
    if 0: # SQUARE
        spinfile=r'C:/Documents and Settings/wflynn/Desktop/spinwave_test_spins.txt'#'C:/eig_test_Spins.txt'
        interfile=r'C:/Documents and Settings/wflynn/Desktop/spinwave_test_montecarlo.txt'
    if 1: # CHAIN
        spinfile=r'C:/Documents and Settings/wflynn/Desktop/sanity_spins.txt'
        interfile=r'C:/Documents and Settings/wflynn/Desktop/sanity_montecarlo.txt'    
    temperature = 0.0001

    tau_list = [np.array([0,0,0])]

    atom_list, jnums, jmats,N_atoms_uc=readFiles(interfile,spinfile)
    
#    correction_driver(interfile,spinfile,tau_list,temperature)
#    sys.exit()
    
    N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wt_list,fflist = run_cross_section(interfile,spinfile,tau_list,temperature)
#    left_conn, right_conn = Pipe()
#    p = Process(target = create_latex, args = (right_conn, csection, "Cross-Section"))
#    p.start()
#    eig_frame = LaTeXDisplayFrame(self.parent, p.pid, left_conn.recv(), 'Cross-Section')
#    self.process_list.append(p)
#    p.join()
#    p.terminate()

    h_list = kapvect[:,0]
    k_list = kapvect[:,1]
    l_list = kapvect[:,2]
    w_list = wt_list

#    syms = (DD_KTPQ_SYM,DD_KTMQ_SYM,W_SYM,WQ_SYM,KAPXHAT_SYM,KAPYHAT_SYM,KAPZHAT_SYM)
#    vals = (1.0,       1.0        ,W_SYM,WQ_SYM,KAPXHAT_SYM,KAPYHAT_SYM,KAPZHAT_SYM)
#    cs_func = lambdify_expr(csection,syms,vals)
#
#    print cs_func(W_SYM,WQ_SYM,KAPXHAT_SYM,KAPYHAT_SYM,KAPZHAT_SYM)
#    sys.exit()

    # FASTER/MORE-ACCURATE METHOD TO GENERATE CROSS SECTION
    if 1:
        st = clock()
        x,y,z=run_eval_pointwise(N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wt_list,fflist,temperature)
        en = clock()
        print en-st
        
        EN = clock()
        print 'TIME TIME TIME', EN - ST
        plot_cross_section(x,y,z)
        clear_cache()
        sys.exit()

    # ORIGINAL METHOD TO GENERATE CROSS SECTION
    if 0:
        st = clock()
        kapvect,wt_list,csdata=run_eval_cross_section(N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wt_list,fflist)
        
        en = clock()
        print en-st
        plot_cross_section(kapvect[:,0],wt_list,csdata)
        sys.exit()
    
    # TEST SINGLE VALUE SPHERICAL_AVERAGING
    if 0:
        radius = 0.2
        ffval = 0.9
        same_args = (N_atoms_uc, atom_list, csection, eig_list, temperature)
        vals=[]
        for wvalue in w_list:
            val=spherical_averaging(radius , wvalue, tau_list[0], ffval,*same_args)
            vals.append(val)
            print 'wvalue', wvalue, 'val',val
        print vals
        sys.exit()
    
    # TEST FOR SINGLE_CROSS_SECTION_CALC
    if 0:
        st = clock()
        aa = bb = cc = np.array([2.0*np.pi], 'Float64')
        alpha = beta = gamma = np.array([np.pi/2.0], 'Float64')
        vect1 = np.array([[1,0,0]])
        vect2 = np.array([[0,0,1]])
        lattice = Lattice(aa, bb, cc, alpha, beta, gamma, Orientation(vect1, vect2))
        
        ffval = 0.9
        rad_list = kaprange
        theta_list = np.arccos(l_list/rad_list)
        phi_list = np.arctan2(k_list,h_list)
        
        same_args = (N_atoms_uc, atom_list, csection, eig_list, temperature)
        
        final_vals=[]
        for wti in wt_list:
            inner_vals=[]
            for i in range(len(rad_list)):
                inner_vals2=[]
                for tau in tau_list:

                    val = single_cross_section_calc(theta_list[i], phi_list[i], rad_list[i], N_atoms_uc, atom_list, csection, tau, eig_list, wti,
                                                     temperature, ffval, eief = True, efixed = 14.7)
                    inner_vals2.append(val)
                inner_vals.append(sum(inner_vals2))
            final_vals.append(inner_vals)
        final_vals = np.array(final_vals)
        
        en = clock()
        print en-st

        plot_cross_section(h_list,wt_list,final_vals)
        
        sys.exit()

    rad = 1.0
    tau = np.array([0,0,0])
    wt = np.array(1.0)
    
    x,y,z=run_spherical_averaging(N_atoms_uc,atom_list,rad,csection,kapvect,tau_list,eig_list,wt_list,temperature)
    np.save(os.path.join(file_pathname,r'myfilex.txt'),x)
    np.save(os.path.join(file_pathname,r'myfiley.txt'),y)
    np.save(os.path.join(file_pathname,r'myfilez.txt'),z)

    #plot_cross_section(x,y,z)
    
#    for h,k,l in zip(h_list,k_list,l_list):
#        for ele in w_list:
#            t = np.arccos(l)
#            p = np.arctan2(k,h)
#            eig = deepcopy(eig_list[0][0])
#            kx = sp.Symbol('kx',real=True)
#            ky = sp.Symbol('kx',real=True)
#            kz = sp.Symbol('kx',real=True)
#            S = sp.Symbol('S',real=True)
#            eig = eig.subs(kx,h).subs(ky,k).subs(kz,l).subs(S,1.0).evalf(chop=True)
#            rad = np.sqrt(h*h+k*k+l*l)
#            res = spherical_averaging(N_atoms_uc, atom_list, rad, csection, tau, eig_list, ele, 0.0001,theta=t,phi=p)
#            points.append(res)
#    points = np.array(points)
#    points = points.reshape((50,50)).T
#    print points
    #print csdata
    #plot_cross_section(h_list,wtlist,points)
    #plot_cross_section(kapvect[:,0],wtlist,csdata)
