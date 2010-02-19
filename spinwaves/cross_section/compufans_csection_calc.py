import sys
import os
#sys.path.append(r"/home/npatel/python_code/spinwaves_git")
import sympy as sp
import sympy.matrices as spm
from sympy import I,pi,exp,oo,sqrt
from sympy.physics.paulialgebra import delta
import numpy as np
from scipy.integrate import simps

#import matplotlib
#matplotlib.use('WXAgg')
#import pylab
#from matplotlib._pylab_helpers import Gcf
#import matplotlib.ticker as ticker
#import matplotlib.pyplot as plt

from spinwaves.cross_section.util.subin import sub_in
from spinwaves.spinwavecalc.readfiles import atom, readFiles
from spinwaves.spinwavecalc.spinwave_calc_file import calculate_dispersion, calc_eigs_direct
from rescalculator.lattice_calculator import Lattice, Orientation
from periodictable import elements

from multiprocessing import Pipe, Process
from copy import deepcopy,copy
from timeit import default_timer as clock

import Pyro.core
import Pyro.errors
Pyro.core.initClient()

import park
#from cluster.william_mapper import Mapper
from cluster.csection_mapper import Mapper
import cluster.csection_proxy as csproxy

#------------ GLOBAL VARIABLES ---------------------------------------

# ALL GLOBACL VARIABLES ARE CAPTIALIZED
# ALL GLOBAL SYMBOLS WILL BE IN CAPS, FOLLOWED BY _SYM.
# ALL OTHER SYMBOLS ARE USUALLY CONSTRUCTED WITH 'sym%i'%(num,)

# define global variables
T_SYM = sp.Symbol('t', real = True) # t
L_SYM = sp.Symbol('L', real = True) # L
Q_SYM = sp.Symbol('q', real = True) # q
QP_SYM = sp.Symbol('qp', real = True) # qp
W_SYM = sp.Symbol('w', real = True)
WQ_SYM = sp.Symbol('wq', real = True) # wq
WQP_SYM = sp.Symbol('wqp', real = True) #wqp
S_SYM = sp.Symbol('S', commutative = True) # S

KAP_SYM = sp.Symbol('kappa', real = True) # kap
TAU_SYM = sp.Symbol('tau', real = True) # tau
KX_SYM = sp.Symbol('kx', real = True)
KY_SYM = sp.Symbol('ky', real = True)
KZ_SYM = sp.Symbol('kz', real = True)
DD_KTPQ_SYM = sp.Symbol('DD(KT+Q)', real = True)
DD_KTMQ_SYM = sp.Symbol('DD(KT-Q)', real = True)

KAPXHAT_SYM = sp.Symbol('kapxhat',real=True)
KAPYHAT_SYM = sp.Symbol('kapyhat',real=True)
KAPZHAT_SYM = sp.Symbol('kapzhat',real=True)

# Wilds for sub_in method
A_WILD = sp.Wild('A',exclude = [0,T_SYM])
B_WILD = sp.Wild('B',exclude = [0,T_SYM])
C_WILD = sp.Wild('C')
D_WILD = sp.Wild('D')
K_WILD = sp.Wild('K')
    
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
    if isinstance(expr, int):
        return 0
    expr = sp.collect(expr, term)
    #print 'expr',expr
    symbols = list(term.atoms(sp.Symbol))
    #print 'symbols',symbols
    w = sp.Wild("coeff", exclude = symbols)
    #print 'w',w
    m = expr.match(w * term + sp.Wild("rest"))
    #print 'm',m
    m2 = expr.match(w * term)
    #print 'm2',m2
    res = False
    if m2 != None:
        #print 'm2[w]',m2[w]
        res = m2[w] * term == expr
    if m and res!= True:
        return m[w]
    #added the next two lines
    elif m2:
        return m2[w]

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

    for i in range(N):
        a_list.append(exp(I*(Q_SYM*L_SYM - WQ_SYM*T_SYM)) * b_list[i])
        ad_list.append(exp(-I*(Q_SYM*L_SYM - WQ_SYM*T_SYM)) * bd_list[i])

    a = sp.Pow(sp.sqrt(N),-1) * sum(a_list)
    ad = sp.Pow(sp.sqrt(N),-1) * sum(ad_list)

    print "Operators Generated: a, ad"
    return (a,ad)

# Generates the Sp and Sm operators
def generate_Sp_Sm_operators(atom_list, a, ad):
    """Generates S+ and S- operators"""

    Sp = sp.sqrt(2*S_SYM) * a
    Sm = sp.sqrt(2*S_SYM) * ad

    print "Operators Generated: Sp, Sm"
    return (Sp,Sm)

def generate_Sa_Sb_Sn_operators(atom_list, Sp, Sm):
    """Generates Sa, Sb, Sn operators"""

    Sa = ((1./2.)*(Sp+Sm)).expand()
    Sb = ((1./2.)*(1./I)*(Sp-Sm)).expand()
    Sn = (S_SYM - sp.Pow(2*S_SYM,-1) * Sm.expand() * Sp.expand()).expand()

    print "Operators Generated: Sa, Sb, Sn"
    return (Sa, Sb, Sn)

# Generates the Sx, Sy and Sz operators
def generate_Sx_Sy_Sz_operators(atom_list, Sa, Sb, Sn):
    """Generates Sx, Sy and Sz operators"""
    Sx_list = []; Sy_list = []; Sz_list = []
    N = len(atom_list)

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
            
    Sx_list.append(KAPXHAT_SYM)
    Sy_list.append(KAPYHAT_SYM)
    Sz_list.append(KAPZHAT_SYM)
    print "Operators Generated: Sx, Sy, Sz"
    return (Sx_list,Sy_list,Sz_list)

# Define a method that generates the possible combinations of operators
def generate_possible_combinations(atom_list, alist):
    """This method returns the possible operator combinations from a list of operators"""
    # For a combination to be returned, the product must have an equal number of b
    # and b dagger operators. If not, they are rejected.
    op_list = []
    alista = []
    N = len(atom_list)

    alista = [[subelement.subs(T_SYM, 0) for subelement in element] for element in alist]
    for ele in alista:
        for sub in ele:
            sub = sub.subs(L_SYM,0)

    for i in range(len(alist)):
        for j in range(len(alist)):
            vect1 = alist[i][-1]
            vect2 = alist[j][-1]
            if cmp(vect1, vect2) == 0: delta = 1
            else: delta = 0

            allzerolist = [alista[i][0].subs(L_SYM,0) for k in range(len(alista[i])-1)]+[delta-vect1*vect2]
            otherlist = [alist[j][k].subs(Q_SYM,QP_SYM).subs(WQ_SYM,WQP_SYM) for k in range(len(alist[j])-1)]+[1]
            append_list = list_mult(allzerolist,otherlist)
            op_list.append(append_list)
    print "Generated: Possible Operator Combinations"
    return op_list

# 
def holstein(atom_list, arg):
    new = []
    N = len(atom_list)

    for k in range(len(arg)):
        temp = []
        for i in range(N):
            Snew = atom_list[i].spinMagnitude

            #gets rid of time independent terms
            #works since arg[k][i] is an Add instance so args breaks it up at '+'s
            pieces = arg[k][i].args
            for piece in pieces:
                if not piece.has(T_SYM):
                    arg[k][i] = arg[k][i] - piece

            S2coeff = coeff(arg[k][i], S_SYM**2)
            Scoeff = coeff(arg[k][i], S_SYM)
            if S2coeff != None and Scoeff != None:
                temp.append((S2coeff*S_SYM**2 + Scoeff*S_SYM).subs(S_SYM,Snew))
            elif S2coeff != None and Scoeff == None:
                temp.append((S2coeff*S_SYM**2).subs(S_SYM,Snew))
            elif S2coeff == None and Scoeff != None:
                temp.append((Scoeff*S_SYM).subs(S_SYM,Snew))
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

def eq(a,b,tol=2e-1):
        c = (abs(a-b) < tol)
        if c:
            return 0
        else:
            if a>b: return abs(a-b)
            else: return abs(b-a)

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

    eigs = Hsave.eigenvals().keys()
    for q in qlist:
        eig_list.append(eigs)
    eig_list = np.array(eig_list)
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
        ff_list = Mq #ff_list.append(Mq)
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

    csection = csection.expand()

    csection = csection.subs(KAPXHAT_SYM*KAPYHAT_SYM,0)
    csection = csection.subs(KAPXHAT_SYM*KAPZHAT_SYM,0)
    csection = csection.subs(KAPYHAT_SYM*KAPZHAT_SYM,0)
    csection = csection.subs(KAPZHAT_SYM*KAPYHAT_SYM,0)
    csection = csection.subs(KAPZHAT_SYM*KAPXHAT_SYM,0)
    csection = csection.subs(KAPYHAT_SYM*KAPXHAT_SYM,0)
        
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
    
    print csection
    
    print "Generated: Analytic Cross-Section Expression"
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
    print fflist.shape
    csdata = (0.5*G_VALUE*fflist)**2*csdata

    return kapvect, wtlist, csdata

def chop(expr,tol=1e-8):
    for item in expr.atoms():
        if item < tol:
            expr = expr.subs(item,0)
    return expr

#@profile
def single_cross_section_calc(theta, phi, rad, N_atoms_uc, atom_list, csection, tau, eig_list, wt,
                       temperature, eief = True, efixed = 14.7):

    temperature = temperature

    front_constant = ((GAMMA_R0_VALUE**2)*DEBYE_WALLER_VALUE/(2*pi*HBAR_VALUE)).evalf()

    kx = rad*np.sin(theta)*np.cos(phi)
    ky = rad*np.sin(theta)*np.sin(phi)
    kz = rad*np.cos(theta)
    kap = np.array([kx,ky,kz])

#    kapunit = kap.copy()
#    kapunit[0]=kap[0]/rad
#    kapunit[1]=kap[1]/rad
#    kapunit[2]=kap[2]/rad    

    ws=[]

    #csectempp = deepcopy(csection)  #good
    #csectempm = deepcopy(csection)  #good             
    
    csectempp = copy(csection)  #good
    csectempm = copy(csection)  #good   

    csectempp = csectempp.subs(DD_KTMQ_SYM,sp.S(1))
    csectempp = csectempp.subs(DD_KTPQ_SYM,sp.S(0))
    csectempm = csectempm.subs(DD_KTMQ_SYM,sp.S(0))
    csectempm = csectempm.subs(DD_KTPQ_SYM,sp.S(1))

#    csectempp = csectempp.subs(sp.DiracDelta(KAP_SYM - Q_SYM - TAU_SYM),sp.S(1))
#    csectempp = csectempp.subs(sp.DiracDelta(KAP_SYM + Q_SYM - TAU_SYM),sp.S(0))
#    csectempm = csectempm.subs(sp.DiracDelta(KAP_SYM - Q_SYM - TAU_SYM),sp.S(0))
#    csectempm = csectempm.subs(sp.DiracDelta(KAP_SYM + Q_SYM - TAU_SYM),sp.S(1))

    csectempp = csectempp.subs(KAPXHAT_SYM,kap[0]/rad)
    csectempp = csectempp.subs(KAPYHAT_SYM,kap[1]/rad)
    csectempp = csectempp.subs(KAPZHAT_SYM,kap[2]/rad)
    csectempm = csectempm.subs(KAPXHAT_SYM,kap[0]/rad)
    csectempm = csectempm.subs(KAPYHAT_SYM,kap[1]/rad)
    csectempm = csectempm.subs(KAPZHAT_SYM,kap[2]/rad)

#    eig_list = eig_list[0].tolist()
#    eig_func = sp.lambdify((S_SYM,KX_SYM,KY_SYM,KZ_SYM),eig_list,modules="numpy")
#    eigens = np.abs(np.array(eig_func(sp.S(1,0),kx,ky,kz)))

#    cs_func = sp.lambdify((DD_KTPQ_SYM,DD_KTMQ_SYM,KAPXHAT_SYM,KAPYHAT_SYM,KAPZHAT_SYM,W_SYM,WQ_SYM)
#                          ,[csection,csection],modules="numpy")
#    csvals = cs_func([1,0],[0,1],kap[0]/rad,kap[1]/rad,kap[2]/rad,)

    for eigi in range(len(eig_list)):
#    for eigi in eigens:
        #eigcsecp=deepcopy(csectempp)  #good
        #eigcsecm=deepcopy(csectempm)
        #eigtemp = deepcopy(eig_list[0][eigi])
        eigcsecp=copy(csectempp)  #good
        eigcsecm=copy(csectempm)
        eigtemp = copy(eig_list[0][eigi]) #good

        eigtemp = eigtemp.subs(S_SYM, sp.S(1.0))
        eigtemp = eigtemp.subs(KX_SYM, kap[0])
        eigtemp = eigtemp.subs(KY_SYM, kap[1])
        eigtemp = eigtemp.subs(KZ_SYM, kap[2])
        eigtemp = sp.abs(eigtemp.evalf(chop=True)) #works
        #eigtemp = chop(np.abs(eigtemp))

#        nval = sp.Pow(sp.exp(eigi/(BOLTZ_VALUE*temperature))-1,-1).evalf()
        #nval = sp.Pow(sp.exp(eigtemp/(BOLTZ_VALUE*temperature))-1,-1).evalf() #works
        #nval = np.power(np.exp(np.abs(eigtemp)/(BOLTZ*temperature))-1,-1)
        nval = sp.S(0)
        for i in range(N_atoms_uc):
            nq = sp.Symbol('n%i'%(i,), real = True)
            eigcsecp = eigcsecp.subs(nq,nval)
            eigcsecm = eigcsecm.subs(nq,nval) 

        wvalp = eigtemp - wt
        wvalm = eigtemp + wt
#        wvalp = eigi - wt
#        wvalm = eigi + wt

        eigcsecp = eigcsecp.subs((W_SYM - WQ_SYM),wvalp)
        eigcsecp = eigcsecp.subs((W_SYM + WQ_SYM),wvalm)
        eigcsecp = sp.re(eigcsecp.evalf(chop = True)) # works
        #eigcsecp = chop(eigcsecp)
        eigcsecm = eigcsecm.subs((W_SYM - WQ_SYM),wvalp)
        eigcsecm = eigcsecm.subs((W_SYM + WQ_SYM),wvalm)
        eigcsecm = sp.re(eigcsecm.evalf(chop = True)) #works
        #eigcsecm = chop(eigcsecm)

        # eief == True => ei=efixed
        # eief == False => ef=efixed
        kpk = 0.
        if eief == True:
            ei = efixed
#            ef = ei - eigi
            ef = ei - eigtemp
            kpk = ef/ei
        else:
            ef = efixed
#            ef = ei + eigi
            ei = ef + eigtemp
            kpk = ef/ei

        ws.append(kpk*(eigcsecp+eigcsecm))
    
    csdata = sp.re(sum(ws))

    #Multiply data by front constants
    csdata = front_constant*csdata

	# CHECK THIS! THIS IS IMPORTANT AND MAY SCREW THINGS UP
    #Multiply by Form Factor
    ff = 0
    for i in range(N_atoms_uc):
        el = elements[atom_list[i].atomicNum]
        val = atom_list[i].valence
        if val != None:
            Mq = el.magnetic_ff[val].M_Q(rad)
        else:
            Mq = el.magnetic_ff[0].M_Q(rad)
        ff = Mq
    csdata = (0.5*G_VALUE*ff)**2*csdata
    
    #print kx,'\t\t',ky,'\t\t',kz,'\t\t',csdata
    return csdata#*np.sin(theta)*rad**2


def spherical_averaging(rad, wt, tau, N_atoms_uc, atom_list, csection, eig_list,
                       temperature, eief = True, efixed = 14.7,thetas=None,phis=None,):
    """
    N_atoms_uc - number of atoms in unit cell
    csection - analytic cross-section expression
    kap - kappa vector
    tau - tau
    eig_list - list of eigenvalues
    wt - omega
    temperature - temperature
    eief - True => E_initial = efixed, False => E_final = efixed
    efixed - fixed energy; either E_final or E_initial, subject to eief
    """
    
    args=(rad, N_atoms_uc, atom_list, csection, tau, eig_list, wt, temperature, eief, efixed)

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

def plot_cross_section(xi, wtlist, csdata, myFlag = True):
    xi = xi # kapvect[:,0]
    yi = wtlist
    zi = np.array(csdata,'Float64')
    
    zmin, zmax = np.min(zi), np.max(zi)
    if myFlag:
        locator = ticker.MaxNLocator(10) # if you want no more than 10 contours
        locator.create_dummy_axis()
        locator.set_bounds(zmin, zmax)
        levs = locator()
        levs[0]=1.0
        plt.contourf(xi,yi,zi, levs)
        
        l_f = ticker.LogFormatter(10, labelOnlyBase=False)
        cbar = plt.colorbar(ticks = levs, format = l_f)
    else: 
        plt.contourf(xi,yi,zi)
        cbar = plt.colorbar()
    #print zmin, zmax
    


    plt.show()    
    
def run_cross_section(interactionfile, spinfile):
    start = clock()

    # Generate Inputs
    atom_list, jnums, jmats,N_atoms_uc=readFiles(interactionfile,spinfile)
    
    atom_list=atom_list[:N_atoms_uc]
    N_atoms = len(atom_list)

    k = spm.Matrix([KX_SYM,KY_SYM,KZ_SYM])
    
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
#    list_print(ops)
    
    print "prelims complete. generating cross-section","\n"

    aa = bb = cc = np.array([2.0*np.pi], 'Float64')
    alpha = beta = gamma = np.array([np.pi/2.0], 'Float64')
    vect1 = np.array([[1,0,0]])
    vect2 = np.array([[0,0,1]])
    lattice = Lattice(aa, bb, cc, alpha, beta, gamma, Orientation(vect1, vect2))
    
    tau_list = []
    for i in range(1):
        tau_list.append(np.array([0,0,0], 'Float64'))

    h_list = np.linspace(0.1,6.3,25)
    k_list = np.zeros(h_list.shape)
    l_list = np.zeros(h_list.shape)
    
    w_list = np.linspace(-10,10,25)

    (N_atoms_uc,csection,kaprange,
     tau_list,eig_list,kapvect,wtlist,fflist) = generate_cross_section(interactionfile, spinfile, lattice, ops, 
                                                                tau_list, h_list, k_list, l_list, w_list)
#    print csection

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

def run_spherical_averaging(N_atoms_uc,atom_list,rad,csection,kapvect,tau_list,eig_list,wt_list,temperature):
#    theta_list=[]
#    phi_list=[]
    print 'running spherical averaging'
    mapper = Pyro.core.getProxyForURI("PYRONAME://:Mapper.dispatcher")
    rad_list=[]
    for kappa in kapvect:
        kx,ky,kz = kappa[0],kappa[1],kappa[2]
        r = np.sqrt(kx*kx+ky*ky+kz*kz)
#        theta_list.append(np.arccos(kz/r))
#        phi_list.append(np.arctan2(ky,kx))
        rad_list.append(r)
        
    same_args = (N_atoms_uc, atom_list, csection, eig_list, temperature)
    rand_wt_list = np.append(wt_list[::2],wt_list[1::2])
#    rand_wt_list = wt_list.copy()
#    np.random.shuffle(rand_wt_list)
    xvals = np.array(rad_list)
#    xvals = np.array(xvals)
    yvals = np.array(wt_list)
    res_array = []
    
    for tau in tau_list:
        job = csproxy.calcservice("PUT SOMETHING HERE")
        vals = mapper.map(spherical_averaging,(rad_list,wt_list,tau,same_args))
        vals = np.array(vals)
        res_array.append(vals)

    res_array = np.sum(res_array, axis=0)
    print res_array
    return rad_list,wt_list,res_array.T

#---------------- MAIN --------------------------------------------------------- 

# Methodized version of MAIN
def cs_driver():
    file_pathname = os.path.abspath('')
    interfile = os.path.join(file_pathname,r'montecarlo.txt')
    spinfile = os.path.join(file_pathname,r'spins.txt')

    h_list = np.linspace(0.001,3.14,15)
    k_list = np.zeros(h_list.shape)
    l_list = np.zeros(h_list.shape)
    w_list = np.linspace(0,5,15)
    tau = np.array([0,0,0])
    wt = np.array(1.0)
    rad = 1.0

    atom_list, jnums, jmats,N_atoms_uc=readFiles(interfile,spinfile)
    N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wt_list,fflist = run_cross_section(interfile,spinfile)
    
    x,y,z=run_spherical_averaging(N_atoms_uc,atom_list,rad,csection,kapvect,tau_list,eig_list,wt_list,temperature)
    np.save(os.path.join(file_pathname,r'myfilex.txt'),x)
    np.save(os.path.join(file_pathname,r'myfiley.txt'),y)
    np.save(os.path.join(file_pathname,r'myfilez.txt'),z)

if __name__=='__main__':
#def pd():
    #from spinwaves.cross_section.csection_calc import spherical_averaging as sph

    file_pathname = os.path.abspath('')
    interfile = os.path.join(file_pathname,r'montecarlo.txt')
    spinfile = os.path.join(file_pathname,r'spins.txt')

    atom_list, jnums, jmats,N_atoms_uc=readFiles(interfile,spinfile)
    
    N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wt_list,fflist = run_cross_section(interfile,spinfile)
#    left_conn, right_conn = Pipe()
#    p = Process(target = create_latex, args = (right_conn, csection, "Cross-Section"))
#    p.start()
#    eig_frame = LaTeXDisplayFrame(self.parent, p.pid, left_conn.recv(), 'Cross-Section')
#    self.process_list.append(p)
#    p.join()
#    p.terminate()

    # ORIGINAL METHOD TO GENERATE CROSS SECTION
    if 0:
        kapvect,wt_list,csdata=run_eval_cross_section(N_atoms_uc,csection,kaprange,tau_list,eig_list,kapvect,wt_list,fflist)
        plot_cross_section(kapvect[:,0],wt_list,csdata)

    h_list = np.linspace(0.1,6.3,25)
    k_list = np.zeros(h_list.shape)
    l_list = np.zeros(h_list.shape)
    w_list = np.linspace(-10,10,25)
    temperature = 0.0001
    points = []

    # TEST SINGLE VALUE SPHERICAL_AVERAGING
    if 0:
        radius = 0.2
        same_args = (N_atoms_uc, atom_list, csection, eig_list, temperature)
        vals=[]
        for wvalue in w_list:
            val=spherical_averaging(radius , wvalue, tau_list[0],*same_args)
            vals.append(val)
            print 'wvalue', wvalue, 'val',val
        print vals
        sys.exit()
    
    # TEST FOR SINGLE_CROSS_SECTION_CALC
    if 0:
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
                                                     temperature, eief = True, efixed = 14.7)
                    inner_vals2.append(val)
                inner_vals.append(sum(inner_vals2))
            final_vals.append(inner_vals)
        final_vals = np.array(final_vals)

        plot_cross_section(h_list,wt_list,final_vals)


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
