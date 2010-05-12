import sympy
import numpy as N
#I=sympy.I
#I=1.0j
#pi=sympy.pi
from sympy import exp,I,pi,sin,cos
import matplotlib
matplotlib.use('WXAgg')
#import pylab
import readfiles
#from readfiles import atom
#from sympy import pngview,latex
import scipy.linalg
from matplotlib._pylab_helpers import Gcf
import sys
#from sympy import latex
import matplotlib.pyplot as plt
import spinwaves.cross_section.util.printing as printing
from sympy.core.cache import clear_cache
from copy import copy
import os

#translations=[[0,0,0],
#              [0,0,1],[0,0,-1]
#              [0,1,0],[0,-1,0]
#              [0,1,1],[0,1,-1],[0,-1,1],[0,-1,-1]
#              [1,0,0],[-1,0,0],[]              
#              ]

def print_matplotlib(s):
    #pylab.figure()
    #pylab.text(0,0,s)
    #pylab.axis('off')
    #pylab.figure()
    ##pylab.show()
    return 


def coeff(expr, term):
    if isinstance(expr, int):
        return 0
    expr = sympy.collect(expr, term)
    #print 'expr',expr
    symbols = list(term.atoms(sympy.Symbol))
    #print 'symbols',symbols
    w = sympy.Wild("coeff", exclude=symbols)
    #print 'w',w
    m = expr.match(w*term+sympy.Wild("rest"))
    #print 'm',m
    m2=expr.match(w*term)
    #print 'm2',m2
    res=False
    if m2!=None:
        #print 'm2[w]',m2[w]
        res=m2[w]*term==expr
    if m and res!=True:
        return m[w]
    #added the next two lines
    elif m2:
        return m2[w]



def generate_sabn(N_atoms):
    "generate spins in local coordinate system, with Z as quantization axis"
    Sabn=[]
    S=sympy.Symbol("S",real=True)
    for i in range(N_atoms):
        c=sympy.Symbol('c%d'%(i,),commutative=False,real=True)
        cd=sympy.Symbol('cd%d'%(i,),commutative=False,real=True)
        curr=sympy.matrices.Matrix([sympy.sqrt(S/2.0)*(c+cd),sympy.sqrt(S/2.0)*(c-cd)/I,S-cd*c])
        Sabn.append(curr.reshape(3,1))
    return Sabn


def generate_sxyz(Sabn,atomlist):
    "transform spins from local coordinate system to global system"
    Sxyz=[]
    i=0
    for currS in Sabn:
        #This was actually a matrix
        #tempS=atomlist[i].spin*currS
        tempS = atomlist[i].spinRmatrix*currS
        tempS=tempS.reshape(1,3)
        Sxyz.append(tempS)
        i=i+1
    return Sxyz




def generate_translations():
    translations=[[0,0,0]]
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if i==0 and j==0 and k==0:
                    continue
                else:
                    translations.append([i,j,k])
    return translations

def generate_sabnt(N_atoms,t=''):
    Sabn=[]
    S=sympy.Symbol("S",real=True)
    for i in range(N_atoms):
        ci=sympy.Symbol("c%d%s"%(i,t),commutative=False)
        cdi=sympy.Symbol("cd%d%s"%(i,t),commutative=False)
        curr=[sympy.sqrt(S/2)*(ci+cdi),sympy.sqrt(S/2)*(ci-cdi)/I,S-cdi*ci]
        Sabn.append(curr)
    return Sabn




def generate_hdef(atom_list,Jij,Sxyz,N_atoms_uc,N_atoms):
    "generate the hamiltonian for a set of interacting spins defined by Sxyz"
    N_atoms=len(atom_list)
#    Hdef=[]
#    print 'Jij',Jij,len(Jij)

    N_int = max([max(atom_list[i].neighbors) for i in range(N_atoms_uc)])+1
    Hdef = sympy.matrices.Matrix(N_int,N_int,lambda i,j:0)
    print 'N_atoms_uc', N_atoms_uc
    for i in range(len(atom_list)):
        print 'neightbors', atom_list[i].neighbors
        print 'interactions', atom_list[i].interactions
    print 'N_int', N_int
        
    #for i in range(N_atoms):
    for i in range(N_atoms_uc): #correct
#        N_int=len(atom_list[i].interactions)
        for j in range(N_int):  
#        for j in range(N_int):            
            if j in atom_list[i].neighbors:
                ind = atom_list[i].neighbors.index(j)
                print i,'i',j,'j'
                Hij=Sxyz[i]*Jij[atom_list[i].interactions[ind]]
                print '1', Hij
                Sxyz_transpose=Sxyz[atom_list[i].neighbors[ind]].reshape(3,1)
                print '2', Sxyz_transpose
                Hij=Hij*Sxyz_transpose
                print '3', Hij
                Hij=Hij[0]
                print 'Hij',Hij.expand()
                #Hij=-Hij-atom_list[i].Dx*Sxyz[i][0]**2-atom_list[i].Dy*Sxyz[i][1]**2-atom_list[i].Dz*Sxyz[i][2]**2
                Dij=-atom_list[i].Dx*Sxyz[i][0]**2-atom_list[i].Dy*Sxyz[i][1]**2-atom_list[i].Dz*Sxyz[i][2]**2
#                    print 'Dij',Dij.expand()
#                    print 'D+H',(Dij-Hij).expand()
            else:
                Hij = sympy.S(0)
                Dij = sympy.S(0)

            Hdef[i,j]=(-Hij).expand()
            Hdef[i,i]=(Hdef[i,i]+Dij).expand()
#    print 'generated hdef'
#    print 'Hdef:', Hdef
#    print '\nHdef.atoms: ', Hdef.atoms(sympy.Symbol)
    Hdef = N.array(Hdef)   
    print Hdef
    return Hdef


def holstein(Hdef):
        S = sympy.Symbol('S',real=True)
        print 'holstein'

        p = sympy.Wild('p',exclude='S')
        q = sympy.Wild('q',exclude='S')
        r = sympy.Wild('r',exclude='S')
        l = sympy.Wild('l',exclude='S')

        Hlin=copy(Hdef)
        for i in range(len(Hdef)):
            for j in range(len(Hdef[0])):
                Hdef[i][j]=Hdef[i][j].expand()
                S2coeff=coeff(Hdef[i][j],S**2)
                Scoeff=coeff(Hdef[i][j],S)
                
                if Scoeff!=None and S2coeff!=None:
                    Hlin[i][j]=S2coeff*S**2+Scoeff*S
                elif Scoeff==None and S2coeff!=None:
                    Hlin[i][j]+=S2coeff*S**2
                elif Scoeff!=None and S2coeff==None:
                    Hlin[i][j]+=Scoeff*S
        return Hlin


def fouriertransform(atom_list,Jij,Hlin,k,N_atoms_uc,N_atoms):
    #N_atoms=len(atom_list)
    #N_atoms_uc=1
    #N_atoms_uc=N_atoms
    #Hdef=0
    #print 'atom_list',atom_list
    #print 'Sxyz',Sxyz
    #print 'Jij',Jij
    print 'fourier'
    print Hlin
#    print Hlin.atoms(sympy.Symbol)
#    print 'expand'
    #Hlin=Hlin.expand()
#    print Hlin.atoms(sympy.Symbol)
    #print Hlin
    #for i in range(N_atoms):
    Hlin = sympy.Matrix(Hlin)
    N_int = max([max(atom_list[i].neighbors) for i in range(N_atoms_uc)])+1
    for i in range(N_atoms_uc): #correct
#        N_int=len(atom_list[i].interactions)
        ci=sympy.Symbol('c%d'%(i,),commutative=False,real=True)
        cdi=sympy.Symbol('cd%d'%(i,),commutative=False,real=True)
        cki=sympy.Symbol('ck%d'%(i,),commutative=False,real=True)
        ckdi=sympy.Symbol('ckd%d'%(i,),commutative=False,real=True)
        cmki=sympy.Symbol('cmk%d'%(i,),commutative=False,real=True)
        cmkdi=sympy.Symbol('cmkd%d'%(i,),commutative=False,real=True)
        ri=atom_list[i].pos
#        N_int=len(atom_list[i].interactions)
        for j in range(N_int): #range(N_int)
            rj=atom_list[j].pos # atom_list[
            j2=i#atom_list[i].neighbors[j]
            print 'i, j', i, j

            cj=sympy.Symbol('c%d'%(j,),commutative=False,real=True)
            cdj=sympy.Symbol('cd%d'%(j,),commutative=False,real=True)
            ckj=sympy.Symbol('ck%d'%(j2,),commutative=False,real=True)
            ckdj=sympy.Symbol('ckd%d'%(j2,),commutative=False,real=True)
            cmkj=sympy.Symbol('cmk%d'%(j2,),commutative=False,real=True)
            cmkdj=sympy.Symbol('cmkd%d'%(j2,),commutative=False,real=True)

            diffr=ri-rj
            kmult=N.dot(k,diffr)
            t1=1.0/2*(ckdi*cmkdj*exp(-I*kmult)+cmkdi*ckdj*exp(I*kmult))
            t2=1.0/2*(cki*cmkj*exp(I*kmult)+cmki*ckj*exp(-I*kmult))
            t3=1.0/2*(ckdi*ckj*exp(-I*kmult)+cmkdi*cmkj*exp(I*kmult))
            t4=1.0/2*(cki*ckdj*exp(I*kmult)+cmki*cmkdj*exp(-I*kmult))
            t5=1.0/2*(ckdj*ckj+cmkdj*cmkj)

            f1=cdi*cdj
            f2=ci*cj
            f3=cdi*cj
            f4=ci*cdj
            f5=cdj*cj

            Hlin=Hlin.subs(f1,t1)
            Hlin=Hlin.subs(f2,t2)
            Hlin=Hlin.subs(f3,t3)
            Hlin=Hlin.subs(f4,t4)
            Hlin=Hlin.subs(f5,t5)


#            Hlin[i][j]=Hlin[i][j].subs(f1,t1)
#            Hlin[i][j]=Hlin[i][j].subs(f2,t2)
#            Hlin[i][j]=Hlin[i][j].subs(f3,t3)
#            Hlin[i][j]=Hlin[i][j].subs(f4,t4)
#            Hlin[i][j]=Hlin[i][j].subs(f5,t5)
#
#            Hlin[j][i]=Hlin[j][i].subs(f1,t1)
#            Hlin[j][i]=Hlin[j][i].subs(f2,t2)
#            Hlin[j][i]=Hlin[j][i].subs(f3,t3)
#            Hlin[j][i]=Hlin[j][i].subs(f4,t4)
#            Hlin[j][i]=Hlin[j][i].subs(f5,t5)

    return Hlin

def applycommutation(atom_list,Jij,Hfou,k,N_atoms_uc,N_atoms):
    """Operate commutation relations to put all the 2nd order term as ckd**ck, cmk**cmkd, cmk**ck and ckd**cmkd form"""
    print "commutation application"
    N_int = max([max(atom_list[i].neighbors) for i in range(N_atoms_uc)])+1
    for i in range(N_int):
        N_int=len(atom_list[i].interactions)
        ci=sympy.Symbol("c%d"%(i,),commutative=False,real=True)
        cdi=sympy.Symbol("cd%d"%(i,),commutative=False,real=True)
        cki=sympy.Symbol("ck%d"%(i,),commutative=False,real=True)
        ckdi=sympy.Symbol("ckd%d"%(i,),commutative=False,real=True)
        cmki=sympy.Symbol("cmk%d"%(i,),commutative=False,real=True)
        cmkdi=sympy.Symbol("cmkd%d"%(i,),commutative=False,real=True)
        for j in range(N_int):
            cj=sympy.Symbol("c%d"%(j,),commutative=False,real=True)
            cdj=sympy.Symbol("cd%d"%(j,),commutative=False,real=True)
            ckj=sympy.Symbol("ck%d"%(j,),commutative=False,real=True)
            ckdj=sympy.Symbol("ckd%d"%(j,),commutative=False,real=True)
            cmkj=sympy.Symbol("cmk%d"%(j,),commutative=False,real=True)
            cmkdj=sympy.Symbol("cmkd%d"%(j,),commutative=False,real=True)
            if i==j:
                Hfou=Hfou.subs(cki*ckdj,ckdj*cki+1)
                Hfou=Hfou.subs(cmkdi*cmkj,cmkj*cmkdi+1)
                
#                Hfou[j][i]=Hfou[j][i].subs(cki*ckdj,ckdj*cki+1)
#                Hfou[j][i]=Hfou[j][i].subs(cmkdi*cmkj,cmkj*cmkdi+1)
            else:
                Hfou=Hfou.subs(cki*ckdj,ckdj*cki)
                Hfou=Hfou.subs(cmkdi*cmkj,cmkj*cmkdi)
                
#                Hfou[j][i]=Hfou[j][i].subs(cki*ckdj,ckdj*cki)
#                Hfou[j][i]=Hfou[j][i].subs(cmkdi*cmkj,cmkj*cmkdi)

            Hfou=Hfou.subs(cki*cmkj,cmkj*cki)
#            Hfou[j][i]=Hfou[j][i].subs(cki*cmkj,cmkj*cki)
                
            Hfou=Hfou.subs(cmkdi*ckdj,ckdj*cmkdi)
#            Hfou[j][i]=Hfou[j][i].subs(cmkdi*ckdj,ckdj*cmkdi)
            
            Hfou.expand()
#            Hfou[j][i].expand()
    
    return Hfou

def gen_operator_table(atom_list,N_atoms_uc):
    """Operate commutation relations to put all the 2nd order term as ckd**ck, cmk**cmkd, cmk**ck and ckd**cmkd form"""

    operator_table=[]
    operator_table_minus=[]
    
    for i in range(N_atoms_uc):
        N_int=len(atom_list[i].interactions)
        ci=sympy.Symbol("c%d"%(i,),commutative=False,real=True)
        cdi=sympy.Symbol("cd%d"%(i,),commutative=False,real=True)
        cki=sympy.Symbol("ck%d"%(i,),commutative=False,real=True)
        ckdi=sympy.Symbol("ckd%d"%(i,),commutative=False,real=True)
        cmki=sympy.Symbol("cmk%d"%(i,),commutative=False,real=True)
        cmkdi=sympy.Symbol("cmkd%d"%(i,),commutative=False,real=True)
        operator_table.append(cki)
        operator_table_minus.append(cmkdi)

    operator_table=[operator_table,operator_table_minus]
    operator_table=N.ravel(operator_table)
    return operator_table


def gen_operator_table_dagger(atom_list,N_atoms_uc):
    """Operate commutation relations to put all the 2nd order term as ckd**ck, cmk**cmkd, cmk**ck and ckd**cmkd form"""

    operator_table=[]
    operator_table_minus=[]
    
    for i in range(N_atoms_uc):
        N_int=len(atom_list[i].interactions)
        ci=sympy.Symbol("c%d"%(i,),commutative=False,real=True)
        cdi=sympy.Symbol("cd%d"%(i,),commutative=False,real=True)
        cki=sympy.Symbol("ck%d"%(i,),commutative=False,real=True)
        ckdi=sympy.Symbol("ckd%d"%(i,),commutative=False,real=True)
        cmki=sympy.Symbol("cmk%d"%(i,),commutative=False,real=True)
        cmkdi=sympy.Symbol("cmkd%d"%(i,),commutative=False,real=True)
        operator_table.append(ckdi)
        operator_table_minus.append(cmki)

    operator_table=[operator_table,operator_table_minus]
    operator_table=N.ravel(operator_table)
    return operator_table

def gen_XdX(atom_list,operator_table,operator_table_dagger,Hcomm,N_atoms_uc):
    """Operate commutation relations to put all the 2nd order term as ckd**ck, cmk**cmkd, cmk**ck and ckd**cmkd form"""
    print "gen_XdX"
    exclude_list=[]
    coeff_list=[]

    print N_atoms_uc
    N_int = max([max(atom_list[i].neighbors) for i in range(N_atoms_uc)])+1

    XdX=sympy.zeros(2*N_atoms_uc)
    g=sympy.zeros(2*N_atoms_uc)
    for i in range(2*N_atoms_uc):
        curr_row=[]
        for j in range(2*N_atoms_uc):
            mycoeff=operator_table_dagger[i]*operator_table[j]
            exclude_list.append(mycoeff)
            currcoeff1=coeff(Hcomm[i,j].expand(),mycoeff)
            currcoeff2=coeff(Hcomm[j,i].expand(),mycoeff)
            print Hcomm[i,j]
            print mycoeff
            print currcoeff1,currcoeff2
            if currcoeff1!=None and currcoeff2!=None:
                XdX[i,j]=currcoeff1+currcoeff2
                curr_row.append(currcoeff1+currcoeff2)
            if i!=j:
                g[i,j]=0
            else:
                if i>=N_atoms_uc:
                    g[i,j]=-1
                else:
                    g[i,j]=1
    print 'XdX done'
    return XdX,g



def calculate_dispersion(atom_list,N_atoms_uc,N_atoms,Jij,showEigs=False):
    Sabn=generate_sabn(N_atoms)       
#    print 'Sabn',Sabn 
    Sxyz=generate_sxyz(Sabn,atom_list)
    print 'Sxyz', Sxyz
        
    if 1:
        #print len(translations)   
        J=sympy.Symbol('J',real=True)
        #Jij=[N.matrix([[J,0,0],[0,J,0],[0,0,J]])]
        #Hdef=generate_hdef(atom_list,Jij,Sabn,N_atoms_uc,N_atoms)
        Hdef=generate_hdef(atom_list,Jij,Sxyz,N_atoms_uc,N_atoms)
        print 'Hdef',Hdef.sum()
#        file_pathname = os.path.abspath('')
#        N.save(os.path.join(file_pathname,r'newHlin.txt'),[Hdef.sum()])
#        sys.exit()
        
        #pngview(Hdef)
        #print_matplotlib(latex(Hdef)) 
    #if 0:
        Hlin=holstein(Hdef)
#        print Hlin.sum()
#        file_pathname = os.path.abspath('')
#        N.save(os.path.join(file_pathname,r'newHlin.txt'),[Hlin.sum()])
#        sys.exit()
        print 'Hlin',Hlin
        kx=sympy.Symbol('kx',real=True)
        ky=sympy.Symbol('ky',real=True)
        kz=sympy.Symbol('kz',real=True)
        k=[kx,ky,kz]
        Hfou=fouriertransform(atom_list,Jij,Hlin,k,N_atoms_uc,N_atoms)
        print 'Hfou',Hfou
    if 1:
        Hcomm=applycommutation(atom_list,Jij,Hfou,k,N_atoms_uc,N_atoms)
#        Hcomm=Hcomm.sum()
#        print 'Hcomm',Hcomm
        operator_table=gen_operator_table(atom_list,N_atoms_uc)
#        print 'optable',operator_table
        operator_table_dagger=gen_operator_table_dagger(atom_list,N_atoms_uc)
#        print 'optable_dagger',operator_table_dagger
        XdX,g=gen_XdX(atom_list,operator_table,operator_table_dagger,Hcomm,N_atoms_uc)
        print XdX
#        print 'XdX',XdX
#        print 'g',g
        TwogH2=g*XdX
#        TwogH2=2*g*XdX
#        print 'TwogH2',TwogH2
        print 'trigifying'
        m,n=TwogH2.shape
        if 1:
                for i in range(m):
                    for j in range(n):
                        #print i,j
                        #print Ntwo[i,j]
                        #print 'matching'
                        #print 'kx',Ntwo[i,j].match(kx)
                        #print 'ky',Ntwo[i,j].match(ky)
                        #Ntwo[i,j]=sympy.re(Ntwo[i,j].evalf())
                        #Ntwo[i,j]=Ntwo[i,j].evalf()
                        TwogH2[i,j]=TwogH2[i,j].expand(complex=True, trig = True)#.subs(I,1.0j)
        print 'trigified'
#        print 'trigified',TwogH2
#        print TwogH2[0,1]            
                        
        Hsave=TwogH2        
        if showEigs:
            #print 'calculating'
            x=sympy.Symbol('x')
            #eigspoly=TwogH2.berkowitz_charpoly(x)
            #print 'eigspoly'
            #print 'eigs poly',eigspoly
            if 0:
                print 'shape', TwogH2.shape
                print 'recalculating\n\n'
                print TwogH2, "\n\n"

            eigs=TwogH2.eigenvals()
            x=sympy.Symbol('x')
           # print_matplotlib(latex(TwogH2))
            #eigs=TwogH2.berkowitz_charpoly(x)
            print 'eigs', eigs
            keys=eigs.keys()
            #print 'key',keys[0]
            #print keys[0].expand(complex=True)
            #print 'charpoly',TwogH2.charpoly(x)
            #eigs=TwogH2.eigenvals()
            #print 'eigenvalues', sympy.simplify(eigs[1][0])
            return (Hsave, Hsave.charpoly(x), eigs)
        print 'calc dispersion: complete'
        clear_cache()
        return Hsave
    
def calc_eigs(Hsave,kx_val,ky_val,kz_val):
    kx=sympy.Symbol('kx',real=True)
    ky=sympy.Symbol('ky',real=True)
    kz=sympy.Symbol('kz',real=True)
    k=[kx,ky,kz]
    TwogH2=Hsave
    S=sympy.Symbol('S',real=True)
    D=sympy.Symbol('D',real=True)
    #TwogH2=TwogH2.subs(J,-1.0)
    TwogH2=TwogH2.subs(S,1.0)
    #eigs=TwogH2.eigenvals()
    #print 'subbed_eigs',eigs
    #TwogH2=TwogH2.subs(D,1.0)
    #qrange=[]
    #wrange0=[]
    #wrange1=[]
    #wrange=[]
    #wrangec=[]
    currnum=kx_val
    #print 'currnum x', currnum
    TwogH3=TwogH2.subs(kx,currnum)
    currnum=ky_val
    #print 'currnum y',currnum
    TwogH3=TwogH3.subs(ky,currnum)
    currnum=kz_val
    TwogH3=TwogH3.subs(kz,currnum)
    #I=sympy.Symbol('I')
    Ntwo=TwogH3#.subs(I,1.0j)
    m,n=Ntwo.shape
    #print Ntwo.applyfunc(sympy.Basic.evalf)
    Nthree=N.empty([m,n],complex)
    if 1:
        for i in range(m):
            for j in range(n):
                #print i,j
                #print Ntwo[i,j]
                #print 'matching'
                #print 'kx',Ntwo[i,j].match(kx)
                #print 'ky',Ntwo[i,j].match(ky)
                #Ntwo[i,j]=sympy.re(Ntwo[i,j].evalf())
                #Ntwo[i,j]=Ntwo[i,j].evalf()
                Nthree[i,j]=complex(Ntwo[i,j].expand(complex=True))#.subs(I,1.0j)
                if 1:
                    if N.absolute(Nthree[i,j])<1e-5:
                        Nthree[i,j]=0
        #print 'Ntwo',Ntwo
        #print 'Nthree',Nthree
        if 1:
            
            l,v=scipy.linalg.eig(Nthree)
            #print l[1]
            #qrange.append(q)
            #print 'num eigs', l.shape
            #wrange0.append(l[0])
            #wrange1.append(l[3])
            #wrange.append(l)
            return l
        #print N.linalg.eigvals(Ntwo)
        #eigs=TwogH2.eigenvals()
        #print 'eigs', eigs
        #print 'eigenvalues', sympy.simplify(eigs[1][0])
    #wrange0=N.real(wrange0)
    #for wrange1 in wrange:
        #pylab.plot(qrange,wrange0,'s')
        #print qrange
        #print wrange1
        #print len(qrange),len(wrange1)
        #print inum
        #inum=inum+1
        #pylab.plot(qrange,wrange1,'s')
        
def calc_eigs_direct(Hsave,H,K,L):
        kx=sympy.Symbol('kx',real=True)
        ky=sympy.Symbol('ky',real=True)
        kz=sympy.Symbol('kz',real=True)
        k=[kx,ky,kz]
        TwogH2=Hsave
        S=sympy.Symbol('S',real=True)
        D=sympy.Symbol('D',real=True)
        #TwogH2=TwogH2.subs(J,-1.0)
        TwogH2=TwogH2.subs(S,1.0)
        #eigs=TwogH2.eigenvals()
        #print 'subbed_eigs',eigs
        #TwogH2=TwogH2.subs(D,1.0)
        qrange=[]
        wrange0=[]
        wrange1=[]
        wrange=[]
        #wrangec=[]
        for p in range(len(H)): 
            TwogH3=TwogH2.subs(kx,H[p])
            TwogH3=TwogH3.subs(ky,K[p])
            TwogH3=TwogH3.subs(kz,L[p])
            #currnum=q*direction['ky']
            #print 'currnum y',currnum
            #TwogH3=TwogH3.subs(ky,currnum)
            #currnum=q*direction['kz']
            #TwogH3=TwogH3.subs(kz,currnum)
            #I=sympy.Symbol('I')
            Ntwo=TwogH3#.subs(I,1.0j)
            m,n=Ntwo.shape
            #print Ntwo.applyfunc(sympy.Basic.evalf)
            Nthree=N.empty([m,n],'Float64')
            if 1:
                for i in range(m):
                    for j in range(n):
                        #print i,j
                        #print Ntwo[i,j]
                        #print 'matching'
                        #print 'kx',Ntwo[i,j].match(kx)
                        #print 'ky',Ntwo[i,j].match(ky)
                        #Ntwo[i,j]=sympy.re(Ntwo[i,j].evalf())
                        #Ntwo[i,j]=Ntwo[i,j].evalf()
                        #Nthree[i,j]=complex(Ntwo[i,j].expand(complex=True))#.subs(I,1.0j)
                        Nthree[i,j]=Ntwo[i,j]
                        if 1:
                            if N.absolute(Nthree[i,j])<1e-5:
                                Nthree[i,j]=0.0
            #print 'Ntwo',Ntwo
            #print 'Nthree',Nthree
            if 1:
                
                l,v=scipy.linalg.eig(Nthree)
                for cur_l in l:
                    cur_l=cur_l.real
#                print l[1]
                wrange.append(l)
        return N.array(wrange,'Float64')



def multiply_ab(atom_list,Sxyz,a=0,b=0):
    N_atoms=len(atom_list)
    Sdef=0
    print 'atom_list',atom_list
    print 'Sxyz',Sxyz
    T=sympy.Symbol('T',commutative=False)
    Sij0=Sxyz[0][a]
    t=''
    c=sympy.Symbol("c%d%s"%(0,t),commutative=False)
    cd=sympy.Symbol("cd%d%s"%(0,t),commutative=False)
    t='t'
    ct=sympy.Symbol("c%d%s"%(0,t),commutative=False)
    cdt=sympy.Symbol("cd%d%s"%(0,t),commutative=False)
    Sij0=Sij0.subs(ct,c)
    Sij0=Sij0.subs(cdt,cd)  
    for i in range(1,N_atoms):
        Sdef=Sdef+Sij0*Sxyz[i][b]
 
    return Sdef




def Sfouriertransform(atom_list,Slin,k):
    N_atoms=len(atom_list)
    #Hdef=0
    #print 'atom_list',atom_list
    #print 'Sxyz',Sxyz
    #print 'Jij',Jij
    kxp=sympy.Symbol('kxp')
    kyp=sympy.Symbol('kyp')
    kzp=sympy.Symbol('kzp')
    kp=[kxp,kyp,kzp]
    wk=sympy.Symbol("wk")
    wkp=sympy.Symbol("wkp")
    t=sympy.Symbol("t")
    ri=atom_list[0].pos
    
    kmult=N.dot(k,ri)
    #kmultp=N.dot(kp,ri)
    
    ci=sympy.Symbol("c%d"%(0,),commutative=False)
    cdi=sympy.Symbol("cd%d"%(0,),commutative=False)
    cki=sympy.Symbol("ck%d"%(0,),commutative=False)
    ckdi=sympy.Symbol("ckd%d"%(0,),commutative=False)
    
    t1=sympy.exp(I*kmult)*cki
    t3=sympy.exp(-I*(kmult))*ckdi
    Slin=Slin.subs(ci,t1)
    Slin=Slin.subs(cdi,t3)


    for i in range(1,N_atoms):
        N_int=len(atom_list[i].interactions)
        #ci=sympy.Symbol("c%d"%(i,),commutative=False)
        #cdi=sympy.Symbol("cd%d"%(i,),commutative=False)
        cit=sympy.Symbol("c%dt"%(i,),commutative=False)
        cdit=sympy.Symbol("cd%dt"%(i,),commutative=False)

        cki=sympy.Symbol("ck%d"%(0,),commutative=False)
        ckdi=sympy.Symbol("ckd%d"%(0,),commutative=False)
        
        ri=atom_list[i].pos
        kmult=N.dot(k,ri)
        kmultp=N.dot(kp,ri)
        
        t2=sympy.exp(I*(kmultp-wk*t))*cki
                     
        
        t4=sympy.exp(-I*(kmultp-wkp*t))*ckdi
        
        Slin=Slin.subs(cit,t2)
        
        Slin=Slin.subs(cdit,t4)
    
    #Note that I have already assumed that k=kp because I didn't include cqp terms
    Slin=Slin.expand()
    Slin=Slin.subs(wkp,wk)
    Slin=Slin.subs(kxp,kx)
    Slin=Slin.subs(kyp,ky)
    Slin=Slin.subs(kzp,kz)
    return Slin

def Sapplycommutation(atom_list,Sfou,k):
    """Operate commutation relations to put all the 2nd order term as ckd**ck, cmk**cmkd, cmk**ck and ckd**cmkd form"""
    N_atoms=len(atom_list)
    #Hdef=0
    #print 'atom_list',atom_list
    #print 'Sxyz',Sxyz
    #print 'Jij',Jij
    for i in range(N_atoms):
        N_int=len(atom_list[i].interactions)
        ci=sympy.Symbol("c%d"%(0,),commutative=False)
        cdi=sympy.Symbol("cd%d"%(0,),commutative=False)
        cki=sympy.Symbol("ck%d"%(0,),commutative=False)
        ckdi=sympy.Symbol("ckd%d"%(0,),commutative=False)
        cmki=sympy.Symbol("cmk%d"%(0,),commutative=False)
        cmkdi=sympy.Symbol("cmkd%d"%(0,),commutative=False)
        for j in range(N_atoms):
            cj=sympy.Symbol("c%d"%(0,),commutative=False)
            cdj=sympy.Symbol("cd%d"%(0,),commutative=False)
            ckj=sympy.Symbol("ck%d"%(0,),commutative=False)
            ckdj=sympy.Symbol("ckd%d"%(0,),commutative=False)
            cmkj=sympy.Symbol("cmk%d"%(0,),commutative=False)
            cmkdj=sympy.Symbol("cmkd%d"%(0,),commutative=False)
            Sfou=Sfou.expand()
            if i==j:
                Sfou=Sfou.subs(cki*ckdj,ckdj*cki+1)
                #Sfou.subs(cmkdi*cmkj,cmkj*cmkdi)
                nkj=sympy.Symbol("nk%d"%(j,),commutative=True)
                
            else:
                Sfou=Sfou.subs(cki*ckdj,ckdj*cki) #just added
            
            Sfou=Sfou.expand()
            nkj=sympy.Symbol("nk%d"%(j,),commutative=True)
            Sfou=Sfou.subs(ckdj*cki,nkj)
            Sfou=Sfou.expand()
            Sfou=Sfou.subs(cki*ckj,0)
            Sfou=Sfou.expand()
            Sfou=Sfou.subs(ckdi*ckdj,0)

    
    
    return Sfou


def calc_eigs_numerically(mat,h,k,l,S=1):
    """
    Give it a matrix, and the (h,k,l) values to substitute into that matrix, each in a separate list.
    S is automatically evaluated as one, but can be changed. h,k,l lists must be the same length.
    """
    #get rid of these
    S_SYM = sympy.Symbol('S')
    KX_SYM = sympy.Symbol('kx')
    KY_SYM = sympy.Symbol('ky')
    KZ_SYM = sympy.Symbol('kz')        

    #lambdification functionality
    syms = (S_SYM,KX_SYM,KY_SYM,KZ_SYM)
    matsym = mat.tolist()
    func = sympy.lambdify(syms,matsym,modules=["sympy"])
    
    eigarr = []
    Slist = S*N.ones(h.shape)
    
    # reduce symbolic matrix to numerical matrix and calculate the eigenvalues
    for i in range(len(h)):
        eigmat = N.array(func(Slist[i],h[i],k[i],l[i]))
        
        # Convert numpy array to sympy matrix and lambdify it to
        # exchange sympy.I with numpy's 1j. Then convert it back to 
        # a numpy array and append it to the list of eigs. 
        eigmat = sp.Matrix(eigmat)
        I2jfunc = sp.lambdify((sympy.I),eigmat,modules="numpy")
        eigmat = N.array(I2jfunc(1j))

        eigs,vects = N.linalg.eig(eigmat)
        eigarr.append(eigs)
    return N.array(eigarr)


def driver1(spinfile,interactionfile):
    """generates Hsave"""
    atom_list, jnums, jmats,N_atoms_uc=readfiles.readFiles(interactionfile,spinfile)
    N_atoms=len(atom_list)

    print 'N_atoms',N_atoms,'Natoms_uc',N_atoms_uc

    for atom in atom_list:
        print atom.neighbors
        print atom.interactions
    
    Hsave = calculate_dispersion(atom_list,N_atoms_uc,N_atoms,jmats,showEigs=False)   
    
    print 'driver1: complete'
    print Hsave
    
    Hamfile = open('Hsave.txt','w')
    Hamfile.write(sympy.sstr(Hsave))
    Hamfile.close()
    
    return Hsave

def driver2(Hsave,direction, steps, kMin, kMax):
    """plots"""
#    myfilestr=spinfile#r'c:\spins.txt'
#    myspins=readfiles.read_spins(myfilestr)#Returns all spins from file in array form
#    spins=readfiles.find_collinear(myspins)#This is actually a list of Rotation matrices
#    print 'driver spins',spins

    #Now that rotation matrices are calculated algebraically, there is no need for
    #find_collinear.  The functionality of read_spins has been put into readFiles
    
    #myfilestr=interactionfile#r'c:\montecarlo.txt'
    
    
    #any atom.spin opbjects past here would have actually been rotation matrices
    #so they can be replaced with the new spinRmatrix
    #atom_list, jnums, jmats,N_atoms_uc=readfiles.readFiles(interactionfile,spinfile)
    #sympy.matrices.Matrix
    #atom_list[1].spinRmatrix = N.matrix([[-1, 0, 0],
    #                                     [0, 1, 0],
    #                                     [0,0,-1]],'Float64')
    
    #N_atoms=len(atom_list)
    #N_atoms_uc=1
    #print 'N_atoms',N_atoms,'Natoms_uc',N_atoms_uc
    #atom_list=generate_atoms()
    #atom_list=generate_atoms_rot()
    #for atom in atom_list:
        #print atom.neighbors
        #print atom.interactions
    
    #(Hsave, charpoly, eigs)=calculate_dispersion(atom_list,N_atoms_uc,N_atoms,jmats,showEigs=True)
    #sys.exit()
    print "driver2"
    print "Hsave: ", Hsave
    print "kmin: ", kMin
    print "kmax: ", kMax
    print "stapes: ", steps
    print "kx: ", direction['kx']
    print "ky: ", direction['ky']
    print "kz: ", direction['kz']
    qrange = []
    wrange = []
    for q in N.arange(kMin,kMax,(kMax- kMin)/steps):
#        wrange.append(calc_eigs(Hsave,q*direction['kx'], q*direction['ky'], q*direction['kz']))
        wrange.append(calc_eigs(Hsave,q*direction['kx'], q*direction['ky'], q*direction['kz']))
        qrange.append(q)
    
    wrange=N.real(wrange)
    wrange=N.array(wrange)
    wrange=N.real(wrange.T)
    return qrange, wrange

#<<<<<<< .mine
    #print wrange.shape
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #for wrange1 in wrange:
        #ax.plot(qrange, wrange1)
        #plt.hold(True)
    #plt.show()
#=======
    #print wrange.shape
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #for wrange1 in wrange:
        #ax.plot(qrange, wrange1)
        #plt.hold(True)
    #plt.title('Dispersion')
    #plt.xlabel('q || (%i,%i,%i)'%(direction['kx'],direction['ky'],direction['kz']))
    #plt.ylabel(r'$\omega$')
    #plt.show()
#>>>>>>> .r246
    
    
    #direction={}
    #direction['kx']=0.
    #direction['ky']=0.
    #direction['kz']=1.
    
    #pylab.figure()
    #calc_eigs(Hsave,direction,steps)
    
    #pylab.show()
    #for figwin in Gcf.get_all_fig_managers():
    #    figwin.frame.Show()
    #print jmats
    #print direction
    #print steps
    #print spinfile
    #print interactionfile
#@profile
def profile_driver():
    if 1:
        if 0:
            #NEED TO CHANGE THESE! SPECIFIC TO BILL"S MACHINE
            spinfile=r'C:/Users/Bill/Desktop/Spins.txt.'#'C:/eig_test_Spins.txt'
            interactionfile=r'C:/Users/Bill/Documents/montecarlo.txt'#'C:/eig_test_montecarlo.txt'
        if 1:
            spinfile=r'C:/Documents and Settings/wflynn/Desktop/spins.txt'
            interactionfile=r'C:/Documents and Settings/wflynn/Desktop/yang_montecarlo.txt'
        
        steps=100
        data={}
        data['kx']=1.
        data['ky']=0.
        data['kz']=0.
        direction=data
        Hsave = driver1(spinfile,interactionfile)
        eigs = Hsave.eigenvals().keys()
        print 'eigs'
        print eigs
        printing.generate_output(eigs)
        driver2(Hsave,direction,steps,0,2*N.pi)
        #atom_list, jnums, jmats=readfiles.read_interactions(myfilestr,spins)
        #N_atoms=len(atom_list)
        #N_atoms_uc=1
        #print N_atoms
        #atom_list=generate_atoms()
        #calculate_dispersion(atom_list,N_atoms_uc,N_atoms,jmats)
        #print jmats
    
if __name__=='__main__':
    if 1:
        #translations=generate_translations()
        #atom_list=generate_atoms()
        #N_atoms_uc=1
        #N_atoms=2
         #print spins[0]
        #print spins[1]
        #print N.linalg.det(spins[0]), N.linalg.det(spins[1])
        
        #square lattice fm
        #spinfile=r'c:\spins_square.txt'
        #interactionfile=r'c:\montecarlo_square.txt'

        #simple cubic fm
        #spinfile=r'C:/Documents and Settings/wflynn/My Documents/workspace/spinwaves/spinwaves/spinwavecalc/tests/spins_sc.txt'
        #interactionfile=r'C:/Documents and Settings/wflynn/My Documents/workspace/spinwaves/spinwaves/spinwavecalc/tests/montecarlo_sc.txt'
        
        
        if 1:
            #NEED TO CHANGE THESE! SPECIFIC TO BILL"S MACHINE
            spinfile=r'C:/Documents and Settings/wflynn/Desktop/fm_chain_spins_x.txt'#'C:/eig_test_Spins.txt'
            interactionfile=r'C:/Documents and Settings/wflynn/Desktop/fm_chain_montecarlo_x.txt'
        if 0:
            spinfile=r'C:/Documents and Settings/wflynn/Desktop/spins.txt'
            interactionfile=r'C:/Documents and Settings/wflynn/Desktop/yang_montecarlo.txt'
        
        steps=100
        data={}
        data['kx']=1.
        data['ky']=0.
        data['kz']=0.
        direction=data
        Hsave = driver1(spinfile,interactionfile)
        sys.exit()
        eigs = Hsave.eigenvals().keys()
        print 'eigs'
        print eigs
        printing.generate_output(eigs)
        driver2(Hsave,direction,steps,0,2*N.pi)
        #atom_list, jnums, jmats=readfiles.read_interactions(myfilestr,spins)
        #N_atoms=len(atom_list)
        #N_atoms_uc=1
        #print N_atoms
        #atom_list=generate_atoms()
        #calculate_dispersion(atom_list,N_atoms_uc,N_atoms,jmats)
        #print jmats



    