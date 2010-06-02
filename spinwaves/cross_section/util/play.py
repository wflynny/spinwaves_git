import matplotlib
matplotlib.use('WXAgg')
import pylab
import matplotlib.pyplot as plt
from matplotlib._pylab_helpers import Gcf
import sympy as sp
#from sympy import oo,I,cos,sin,exp
import numpy as np
from numpy import sin, cos, exp, pi
from numpy import newaxis as nax
import scipy as sci
from scipy.integrate import quad, inf, dblquad, simps, composite, cumtrapz
from scipy.sparse import bsr_matrix
import sys
from scipy.optimize.slsqp import approx_jacobian
import periodictable
from periodictable import elements
from spinwaves.spinwavecalc.readfiles import atom
from numpy.random import uniform
from printing import *
from multiprocessing import Process, Lock
import matplotlib.pyplot as plt
import spinwaves.spinwavecalc.readfiles as rf
from timeit import default_timer as time
from spinwaves.cross_section.csection_calc import plot_cross_section

if 0:
    file_pathname = r'C:\Documents and Settings\wflynn\My Documents\workspace\spinwaves_git\spinwaves\spinwavecalc'#os.path.abspath('')

    x = np.load(os.path.join(file_pathname,'oldHlin.txt.npy'))
    y = np.load(os.path.join(file_pathname,'newHlin.txt.npy'))
    
    x=x[0].expand()
    y=y[0].expand()

    print x
    print y    
    
    z = x - y
    print sp.simplify(z)

if 0:
    e = 1 + sp.I
    print e.subs(sp.I,np.complex(0,1))

if 0:
    h = np.linspace(1,20,600)
    k = np.linspace(1,20,600)
    l = np.linspace(1,20,600)
    S = sp.Symbol('S')
    kx = sp.Symbol('kx')
    ky = sp.Symbol('ky')
    kz = sp.Symbol('kz')
    hsave = np.array([[1.9999993389278*S - 1.99999966945827*S*sp.cos(kx) - 4.69089402684015e-6*S*sp.sin(kx),-1.5273185983403e-7*S*sp.cos(kx) + 2.8703938375953e-7*sp.I*S*sp.cos(kx)],
                      [1.5273185983403e-7*S*sp.cos(kx) + 2.8703938375953e-7*sp.I*S*sp.cos(kx), -1.9999993389278*S + 1.99999966945827*S*sp.cos(kx) - 4.69089402684015e-6*S*sp.sin(kx)]])
    
    def calc_eigs(mat,h,k,l,S=1):
        """
        Give it a matrix, and the (h,k,l) values to substitute into that matrix, each in a separate list.
        S is automatically evaluated as one, but can be changed.
        """
        #get rid of these
        S_SYM = sp.Symbol('S')
        KX_SYM = sp.Symbol('kx')
        KY_SYM = sp.Symbol('ky')
        KZ_SYM = sp.Symbol('kz')        

        #lambdification
        syms = (S_SYM,KX_SYM,KY_SYM,KZ_SYM)
        matsym = mat.tolist()
        func = sp.lambdify(syms,matsym,modules=["sympy"])

        # reduce symbolic matrix to numerical matrix
        matarr = []
        eigarr = []
        Slist = S*np.ones(h.shape)
        
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
    
    start = time()
    res = calc_eigs(hsave,h,k,l)
    print time()-start
    print res
    print res.shape
    
    
                        

if 0:
    x = sp.Symbol('x')
    e = sp.sin(x)
    f = sp.sqrt(x)
    res1 = sp.ccode(e)
    res2 = sp.ccode(f)
    print res1,res2

if 0:
    ts = ['t%i'%i for i in range(5)]
    ps = ['p%i'%i for i in range(5)]
    arr = np.array([[t+p for p in ps] for t in ts])
    print arr
    print arr[0]

if 0:
    ff = 0
    el = elements[26]
    val = 3
    if val != None:
        Mq = el.magnetic_ff[val].M_Q(1)
    else:
        Mq = el.magnetic_ff[0].M_Q(1)
    ff = Mq
    
    print ff

if 0:
    x,y,z = sp.symbols('xyz')
    exprs = [x+y,x*x+y*y,x*x*x+y*y*y]
    xx = np.linspace(0,1,3)
    yy = xx
    zz = xx
    
    func = sp.lambdify((x,y),exprs,modules="numpy")
    np_func = func(xx[:,nax],xx[nax,:])
    for ele in np_func:
        print ele
    print np.sum(np_func,axis=0)

if 0:
    thetas = np.linspace(0,np.pi,25) 
    phis = np.linspace(0,2*np.pi,25)
    def func(t,p):
        return np.array([[np.tanh(phi)*np.tanh(the) for phi in p] for the in t])
    temp_cs = func(thetas,phis)
    print 'z',temp_cs

    partial_res=[]
    for i in range(len(thetas)):
        single_res = simps(temp_cs[i],phis)
        partial_res.append(single_res)
    total_res = simps(partial_res,thetas)
    print 'res',total_res
    
    def func2(t,p):
        return np.tanh(t)*np.tanh(p)
    pL = lambda t: 0
    pU = lambda t: np.pi
    print 'dblquad',dblquad(func2,0,2*np.pi,pL,pU)

if 0:
    expr = 0
    num = 5
    for i in range(num):
        nq = sp.Symbol('n%i'%(i,))
        expr += nq
    print expr
    npx = np.array([expr])
    print npx
    print npx.shape

if 1:
#    thetas = np.linspace(0,np.pi,25) 
#    phis = np.linspace(0,2*np.pi,25)
#
#    val_func = lambda t,p: 1
#    cs_vals = np.array([[val_func(t,p) for p in phis] for t in thetas])
#    
#    print cs_vals.shape

    file_pathname = r'C:\Documents and Settings\wflynn\My Documents\workspace\spinwaves_git\spinwaves\cross_section'#os.path.abspath('')

#    x = np.load(os.path.join(file_pathname,'myfilex.txt.npy'))
#    y = np.load(os.path.join(file_pathname,'myfiley.txt.npy'))
    z = np.load(os.path.join(file_pathname,'myfilez.txt.npy'))
#    x = np.load(os.path.join(file_pathname,'testx.txt.npy'))
#    y = np.load(os.path.join(file_pathname,'testy.txt.npy'))
#    z = np.load(os.path.join(file_pathname,'testz.txt.npy'))

#    print x.shape
#    print y.shape
    print z.shape
    print z
    
    print z[250,1]
    
#    plt.contourf(x,y,z)
#    plt.show()


if 0:
    csfilestr = 'c:\myfile.txt'
    csfile = open(csfilestr,'r')
    arr = []
    myFlag=1
    while myFlag:
        tokenized = rf.get_tokenized_line(csfile)
        if not(tokenized) or tokenized==[]:
            break
        mytokens = [float(ele) for ele in tokenized]
        arr.append(mytokens)
    csoutput = np.array(arr)
    plot_cross_section(csoutput[0],csoutput[1],csoutput[2])

if 0:
    Sval = 1.0
    xval = 1.0
    yval = 2.0
    zval = 3.0
    Svals = [1.0]
    xvals = np.linspace(0,1,5)
    yvals = np.linspace(0,1,5)
    zvals = np.linspace(0,1,5)
    
    S,kx,ky,kz = sp.Symbol('S'), sp.Symbol('kx'), sp.Symbol('ky'), sp.Symbol('kz')
    expr = -9.3817880536803e-6*S*sp.sin(kx) + (-127.999936536715*S**2*sp.cos(kx) + 63.9999576913864*S**2 + 63.9999788453292*S**2*sp.cos(kx)**2)**(1./2.)/2.
    expr2 = -9.3817880536803e-6*sp.sin(kx) + 1e-99*kz + 1e-99*ky +(-127.999936536715*sp.cos(kx) + 63.9999576913864 + 63.9999788453292*sp.cos(kx)**2)**(1./2.)/2.

    
    expr_np = sp.lambdify((S,kx,ky,kz), expr, modules = "numpy")
    expr_np2 = sp.lambdify((kx,ky,kz), expr2, modules = "numpy")

    #res_np = expr_np(Svals[:,newaxis,newaxis,newaxis],xvals[newaxis,:,newaxis,newaxis],yvals[newaxis,newaxis,:,newaxis],zvals[newaxis,newaxis,newaxis,:])
    res_np = expr_np(sp.Symbol('S'),xvals[:,nax,nax],yvals[:,nax,nax],zvals[:,nax,nax])
    
    print res_np.shape
    print res_np

    
    exprs = [expr,expr]
    res_sym = sp.lambdify((S,kx,ky,kz),exprs)
    print res_sym(1,np.pi,np.pi,np.pi)
    
    
if 0:
    x = np.array([1,2,3,4])
    y = np.array([1,2,3,4])
    z = np.array([1,2,3,4])

    newarr = np.array([x,y,z])
    newarr.tofile(r'c:/myfile.txt', sep=' ', format = "%e")
    
if 0:

    def chop(expr,tol=1e-8):
        for item in expr.atoms():
            if item < tol:
                expr = expr.subs(item,0)
        return expr
        
    print chop(1e-16*sp.sin(1.34323412e-9))
    print np.round(1e-16*np.sin(1.34323412e-9))

if 0:
    def func1(num):
        s = time()
        res = isinstance(num,sp.Symbol)
        e = time()
        print e-s
        return res

    def func2(num):
        s = time()
        num = sp.sympify(num)
        res = num.is_Symbol
        e = time()
        print e-s
        return res
    
    p = 3.1415926
    x = sp.symbols('x')
    
    print func1(p)
    print func2(p)  
    print func1(x)
    print func2(x)  

if 0:
    def func(y,x,factor,stuff=1):
        for j in range(100):
            z=x**2+y**2-x*y
        return y*sin(x)*factor
    res, err = dblquad(func, 0, pi, lambda x: 0, lambda x: 2*pi, args=(100,))
    print res/4.0/pi

if 0:
    def func2d(theta,phi):
        q = 1
        qx=q*sin(theta)*cos(phi)
        qy=q*sin(theta)*sin(phi)
        qz=q*cos(theta)
        
        func = q**2 * sin(theta) * 1.0
        
        return func/(4*pi)

    phiL = 0
    phiU = 2*pi
    def thetaL(phi):
        return lambda phi: 0
    def thetaU(phi):
        return lambda phi: np.pi

    def driver(func2d, phiL, phiU, thetaL, thetaU):
        return dblquad(func2d, phiL, phiU, thetaL, thetaU)
        
    print driver(func2d, phiL, phiU, thetaL, thetaU)


if 0:
    a = [1,2,3,4,5,6,7]
    b = [1,2,5]
    comms = [a[i] in b for i in range(len(a))]
    print comms
    x = sp.Symbol('x')
    y = x**2
    print type(y)
    print y.is_commutative
    print y.atoms()
    

if 0:
    interfile = 'c:/test_montecarlo.txt'
    spinfile = 'c:/test_Spins.txt'  
    atom_list, jnums, jmats,N_atoms_uc=rf.readFiles(interfile,spinfile)
    N_atoms = len(atom_list)
    jij_values = []
    jij_columns = []
    jij_rowIndex = []
    
    # Scan through atom_list
    num_inters = 0
    for i in range(N_atoms):
        ints_list = atom_list[i].interactions
        nbrs_list = atom_list[i].neighbors
        print 'ints', ints_list
        print 'nbrs', nbrs_list
        nbrs_ints = [ (nbrs_list[x],ints_list[x]) for x in range(len(nbrs_list)) ]
        nbrs_ints.sort()

        print 'nbr/int', nbrs_ints
        # Now we have a sorted list of (nbr,intr) tuples from lowest neighbor to highest neighbor
        # Scan through interactions
        for j in range(len(nbrs_ints)):
            nbr = nbrs_ints[j][0]
            intr = nbrs_ints[j][1]
            
            print nbr, intr
            #Get an interaction matrix and flatten it
            curr_mat = np.array(jmats[intr])
            curr_mat = curr_mat.ravel()
            
            # Scan through elements in the interaction matrix
            for k in range(len(curr_mat)):
                # Values just get appended
                jij_values.append(curr_mat[k])
                # Columns are 3*nbr# + el# mod3
                jij_columns.append(3*nbr + k%3)
                # RowIndex is 9*total# of interactions processed 
                #              + el# mod3 = 0 if its the first interaction
                if j == 0 and k%3 == 0:
                    jij_rowIndex.append(k + 9*num_inters)
            num_inters = num_inters + 1
            # Done :)
    print jij_values
    print jij_columns
    print jij_rowIndex
     

def jijToSparse(atom_list):
    """ Creates a scipy bsr sparse array """
    N_atoms = len(atom_list)
    jij_values = []
    jij_columns = []
    jij_rowIndex = []   
    
    # Counts total number of interactions: needed for row indexing
    num_inters = 0
    # Scan through atom_list
    for i in range(N_atoms):
        print 'atom %i'%(i,)
        ints_list = atom_list[i].interactions
        nbrs_list = atom_list[i].neighbors
        nbrs_ints = [ (nbrs_list[x],ints_list[x]) for x in range(len(nbrs_list)) ]
        nbrs_ints.sort()

        # Now we have a sorted list of (nbr,intr) tuples from lowest neighbor to highest neighbor
        # Scan through interactions
        for j in range(len(nbrs_ints)):
            nbr = nbrs_ints[j][0]
            intr = nbrs_ints[j][1]
            
            print 'atom',i, 'nbr',nbr, 'inter',intr

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
    new_mat = bsr_matrix( (jij_values,jij_columns,jij_rowIndex), shape=(3*N_atoms,3*N_atoms) ).todense()
    return new_mat

def gen_spinVector(atom_list):
    N_atoms = len(atom_list)
    vect_list = []
    for i in range(N_atoms):
        spinvect = atom_list[i].spin
        vect_list.append(spinvect[0])
        vect_list.append(spinvect[1])
        vect_list.append(spinvect[2])
    return vect_list
if 0: 
    
    mat = jijToSparse(atom_list)
    print mat
    

if 0:
    x = np.arange(0, 10, 0.2)
    y = np.sin(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    plt.show()


if 0:
    x=sp.Symbol('x')
    z=sp.Symbol('z')
    g=cos(z)*x**4+1
    res=sp.solve(g,x)
    print res
    


if 0:
    kx=sp.Symbol('kx')
    ky=sp.Symbol('ky')
    kz=sp.Symbol('kz')
    x=sp.Symbol('x')
    S=sp.Symbol('S')
    
    g=x**4 + 0.172354909847195*S*sin(0.5*kz)*x**3 + (15.8564808233679*S**2*cos(0.5*kz) - 2.77555756156289e-17*S**2*sin(0.5*kz) + 8.32667268468867e-17*S**2*cos(0.5*kz)*sin(0.5*kz) - 15.813759930039*S**2 + 0.011139830605663*S**2*sin(0.5*kz)**2 - 7.94959042160519*S**2*cos(0.5*kz)**2)*x**2 + (-1.36278958354353*S**3*sin(0.5*kz) + 1.30104260698261e-18*S**3*sin(0.5*kz)**2*cos(0.5*kz) + 1.36647116140267*S**3*cos(0.5*kz)*sin(0.5*kz) - 0.685075470218944*S**3*cos(0.5*kz)**2*sin(0.5*kz) + 3.30872245021211e-24*I*S**3*cos(0.5*kz)**2*sin(0.5*kz) - 3.5527136788005e-15*S**3*cos(0.5*kz)**2 - 3.28590695258778e-19*S**3*sin(0.5*kz)**2 - 4.2351647362715e-22*I*S**3*cos(0.5*kz)**2 + 2.42725761365192e-17*S**3*cos(0.5*kz)**3 + 0.000320000749958679*S**3*sin(0.5*kz)**3)*x + 5.55128452971523e-17*S**4*cos(0.5*kz)**3*sin(0.5*kz) + 0.0294397517290437*S**4*sin(0.5*kz)**2*cos(0.5*kz) + 2.22043659264565e-16*S**4*cos(0.5*kz)**2*sin(0.5*kz) - 4.44089209850063e-16*S**4*cos(0.5*kz)*sin(0.5*kz) +\
     6.59659131232511e-24*I*S**4*cos(0.5*kz)**3*sin(0.5*kz) - 1.82487859077024e-23*I*S**4*cos(0.5*kz)**2*sin(0.5*kz) + 62.8564572347012*S**4*cos(0.5*kz)**2 - 0.0293604344765427*S**4*sin(0.5*kz)**2 - 0.0147595151135139*S**4*cos(0.5*kz)**2*sin(0.5*kz)**2 + 4.27705919711774e-25*I*S**4*cos(0.5*kz)**2*sin(0.5*kz)**2 + 3.73735135389744e-20*S**4*sin(0.5*kz)**3 - 63.0262640369058*S**4*cos(0.5*kz)**3 + 8.44363687977613e-22*I*S**4*cos(0.5*kz)**3 + 3.44710627563518e-6*S**4*sin(0.5*kz)**4 + 15.7989969678193*S**4*cos(0.5*kz)**4
    print 'g'
    res=sp.solve(g,x)
    print res

if 0:
    x,y = sp.symbols('xy')
    create_latex(sp.cos(2.0*x)+y)

if 0:
    def f(arg):
        print arg
    p = Process(target=f, args = ('hi'))
    p.start()
    p.terminate()
    
    print p.pid
    print p.is_alive()
    p.join()
    print p.is_alive()
    
    q = Process(target=f, args = ('hi'))
    q.start()
    q.terminate()
    print q.pid
    print q.is_alive()
    p.terminate()
    print q.is_alive()
    q.terminate()
if 0:
    def f(l, i):
        l.acquire()
        print 'hello world', i
        l.release()
    
    if __name__ == '__main__':
        lock = Lock()
    
        for num in range(10):
            Process(target=f, args=(lock, num)).start()

if 0:
    X = sp.DeferredVector('X')
    Y = sp.DeferredVector('Y')
    print X
    print Y
    print X[0]+X[1]+Y[2]
    print X[0]*Y[0] == Y[0]*X[0]
    X[0].is_commutative = False; Y[0].is_commutative = False
    print X[0]*Y[0] == Y[0]*X[0]
    print Y[2]
    a,b,c = sp.symbols('abc')
    e = a*X[0] + b*X[1] + c*Y[0]
    print e
    e = e.subs(X[0],sp.S(1))
    print e


if 0:
    x,y,z = sp.symbols('xyz')
    generate_output((x+y)/z)

if 0:
    def blahblah(x,y,z):
        if y > z:
            check = lambda x,y: x+y
        else: check = lambda x,y: x**y
        return check(x,y)
    print blahblah(1,2,3)
    print blahblah(3,2,1)
    x = sp.Symbol('x')
    f = lambda x: sp.sin(x+2)
    print f(x)

if 0:
    d,e = sp.symbols('de')
    print (sp.DiracDelta(0)*d+e).evalf()
    a = 1.00001
    b = 1.00001
    print sp.DiracDelta(a-b)
    c = 1.00000
    
    print sp.DiracDelta(a-c)*d

if 0:
    x = sp.Symbol('x', commutative = False)
    y = sp.Symbol('y', commutative = False)
    print (x*y*x*y).subs(x*y,2)
    a,b = sp.symbols('ab')
    print (x*y).subs(x*y,y*x+1)
    print (y*x).subs(x*y,y*x+1)
    e = a*b
    f = e
    f = f.subs(a,b)
    print e
    print f
    sys.exit()

if 0:
    #x=uniform(-2,2,200)
    #y=uniform(-2,2,200)
    #z=x**2+y**2
    #X,Y=np.meshgrid(x,y)
    xi=np.linspace(-2.1,2.1,100)
    yi=np.linspace(-2.1,2.1,100)
    zi=xi**2+yi**2
    Z=np.zeros((xi.shape[0],yi.shape[0]))
    #zi=matplotlib.mlab.griddata(x, y, z, xi, yi)
    #print zi.shape
    
    Z[range(len(xi)),range(len(yi))]=zi
    pylab.contourf(xi,yi,Z)
    pylab.colorbar()
    pylab.show()
    sys.exit()

if 0:
    x,y = sp.symbols('xy')
    out = x*y
    print sp.latex(out)

if 0:
    a = np.array([1,1,1])
    b = 5
    print b/a
    print a/b

if 0:
    at1 = atom(pos=[0,0,0], atomicNum = 26, valence = 3)
    print at1.atomicNum
    el = elements[at1.atomicNum]
    print el
    Q=np.linspace(0.,1.,100.)
    print Q
    Mq = el.magnetic_ff[at1.valence].M_Q(Q)
    print type(Mq)
    print Mq
    print type(Q)
    pylab.plot(Q,Mq)
    pylab.show()

if 0:
    x,y = sp.symbols('xy')
    print (x+1)*y

if 0:
    x,y,z = sp.symbols('xyz', real = False)
    a,b,c = sp.symbols('xyz')
    e = x+y
    f = a+b
    print (4*x+y-z)**(0.5) - (4*a+b-c)**(0.5) == 0
    print sp.cse((4*x+y-z)**(0.5) - (4*a+b-c)**(0.5))
    print sp.sqrt(4-4*sp.cos(x)**2).expand(mul = True, multinomial=True)

if 0:
    a,b,c = sp.symbols('abc')
    x,y,z = sp.symbols('xyz')
    e = x*y*sp.exp(a*x)*sp.exp(b*y)*sp.exp(c*z)
    print e
    print sp.powsimp(e, combine = 'exp')

if 0:
    x = sp.Symbol('x', real = True)
    kx = sp.Symbol('kx', real = True)
    ky = sp.Symbol('ky', real = True)
    kz = sp.Symbol('kz', real = True)
    J,S = sp.symbols('JS', real = True)
    a = (-8.0*J**2*S**2*sp.cos(kx) + 4*J**2*S**2 + 4.0*J**2*S**2*sp.cos(kx)**2)**(0.5)
    b = (-8.0*sp.cos(kx) + 4 + 4.0*sp.cos(kx)**2)**(0.5)
    c = (-8.0*sp.cos(kx) + 4 + 4.0*sp.cos(kx)**2)
    d = sp.sqrt(sp.cos(kx)**2+sp.sin(kx)**2)
    e = (1-sp.cos(kx))**2
    
    f = -8.0*J**2*S**2*cos(kx) + 4*J**2*S**2 + 4.0*J**2*S**2*cos(kx)**2
    g = -8.0*J**2*S**2*cos(kx) + 4*J**2*S**2 + 4.0*J**2*S**2*cos(kx)**2
    print f == g
    
#    print c
#    print sp.simplify(c)
#    print c.expand()
#    print sp.factor((c.subs(sp.cos(kx),x)))
#    print sp.simplify(sp.factor((a.args[0].subs(sp.cos(kx),x))).subs(x,sp.cos(kx))**a.args[1])

if 0:
    x = np.linspace(0.,1.,num=10)
    y = np.linspace(0.,1.,num=10)
    z = np.linspace(0.,1.,num=10)
    print [x,y,z]
    k = np.array([[x[i],y[i],z[i]] for i in range(len(x))])
    print k
    k = np.array([x,y,z])
    print k
    print np.indices((4,4))
    print k.shape[0] - 3

if 0:
    d = 3
    f = d != 0 
    print 'f',f,isinstance(f,bool)
    a = [1,2,3,4,5]
    c = [0,1,2,0,3]
    b = {'1':1,'2':2,'3':3}
    print isinstance(a,list)
    print isinstance(b,dict)
    print
    
    print (np.nonzero((np.array([True,False])!=0.) | (np.array([False,False])!=0.)))[0]
    print
    
    x = np.array([[True,False],[False,False]])
    y = (x != 0)
    print 'x',x
    print 'y',y
    print x[x>0]
    print x[y]
    
    y = np.array([False, False])
    print y
    z = np.array([True])
    print y[z]
    print np.where(y != True)[0]


if 0:
    f = sys._getframe(0)
    file_to_run = f.f_locals.get('__file__', None)
    assert file_to_run is not None
    print file_to_run

if 0:
    x,y,z = sp.symbols('xyz')
    e = x*y+x*z
    a = sp.Wild('a'); b = sp.Wild('b')
    print e.match(a*x)
    
    #alist = [1,2,3,4]
    #blist = alist[:]
    #blist.pop()
    #print alist,blist
    print quad(lambda x: sp.DiracDelta(x), -1, 1)
    print quad(lambda x: sp.DiracDelta(x), -100, 100)
    print quad(lambda x: sp.DiracDelta(x), -10000000, 10000000)
    print quad(lambda x: sp.DiracDelta(x), -inf, inf)
    
    x = sp.Symbol('x')
    e = sp.DiracDelta(x)
    print sp.integrate(e, (x,-oo,oo))
    print sp.Integral(e, (x,-oo,oo))
    print sp.Integral(e, (x,-oo,oo)).evalf()
    print sp.Integral(e, (x,-oo,oo)).evalf(method='mpmath')
    print sp.Integral(e, (x,-oo,oo)).evalf(method='scipy')
    print sp.Integral(e, (x,-oo,oo)).evalf(method='numpy')
    print sp.Integral(e, (x,-oo,oo)).evalf(method='')

if 0:
    print 1e-6
    
    
    def permutations(l):
        sz = len(l)
        if sz <= 1:
            return [l]
        return [p[:i]+[l[0]]+p[i:] for i in xrange(sz) for p in permutations(l[1:])]
    def other_permutations(l):
        all = permutations(l)
        all.remove(l)
        return all
    
    def perms(li):
        sz = len(li)
        if sz <= 1:
            return li
        perm_list = []
        for i in range(sz):
            if li[i].is_commutative:
                el = li[i]
                lis = li[:]; del lis[i]
                for j in range(len(lis)):
                    lis.insert(j,el)
                    print lis
                    #if lis not in perm_list: 
                    perm_list.append(lis)
                    del lis[j]
    #            lis.append(el)
    #            if lis not in perm_list: perm_list.append(lis)
            else: pass
        return perm_list
    
    
    a,b,c,d = sp.symbols('abcd')
    l = [a,b,c]
    print perms(l)
    print ''
    for p in permutations([a,b,c]):
        print p
    print
    for p in other_permutations([a,a,b]):
        print p
    
    x,y,z = sp.symbols('xyz',commutative=False)
    
    print ''
    ex = (a*b*c).subs(a*b,d)
    print ex
    print''
    
    
    x = sp.Symbol("X",commutative=True,real=True)
    y = sp.Symbol("Y",commutative=False,real=True)
    
    
    print x*y
    
    zlist=[]
    
    for i in range(10):
        zi = sp.Symbol("z%i"%(i,),commutative=False,real=True)
        zlist.append(zi)
        
    print zlist
    
    print x**(1/2)
    
    print exp(x)
    n=sp.Symbol("n",commutative=True,real=True)
    #print sp.sum(6*n**2 + 2**n, (n, 0, 6))
    
    
    #print sp.solve(x**2+x+1,x)
    
    #Convolution
    tau = sp.Symbol('tau')
    t,T = sp.symbols('tT')
    #print sp.integrate(tau**2*sp.DiracDelta(tau-(t-T)),(tau,-oo,oo))


print "End"