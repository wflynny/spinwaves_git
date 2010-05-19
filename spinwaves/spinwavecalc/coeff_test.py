import sympy as sp
import numpy as np
from timeit import default_timer as time

def coeff(expr, term):
    if isinstance(expr, int):
        return 0
    expr = sp.collect(expr, term)
    #print 'expr',expr
    symbols = list(term.atoms(sp.Symbol))
    #print 'symbols',symbols
    w = sp.Wild("coeff", exclude=symbols)
    #print 'w',w
    m = expr.match(w*term+sp.Wild("rest"))
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

def coeff_bins(expr,bins):
    # get rid of None expressions
    if not expr:
        return 0
    # chop up expr at '+'
    expr_list = sp.make_list(expr, sp.Add)
    retbins = np.zeros(len(bins),dtype=object)
    #scan through expr
    for subexpr in expr_list:
        #see if it contains a bin element
        for i in range(len(bins)):
            curr_coeff = subexpr.as_coefficient(bins[i])
            if curr_coeff:
                retbins[i] += curr_coeff
    return retbins

def test_raw():
    # expr
    # our code
    # sympy code
    x,y,z = sp.symbols('xyz')
    
  ######################################  
    expr1 = ((x+2*y+3*z)**6).expand()
    
    st = time()
    for i in range(100):
        res_o = coeff(expr1,x*y*z)
    en = time()
    print 'o', en-st
    
    st = time()
    for i in range(100):
        res_s = expr1.coeff(x*y*z)
    en = time()
    print 's', en-st
    
    assert res_s == res_o;
  
  ######################################  
    expr2 = ((sp.sin(x)*sp.cos(y)+z)**4).expand()
    
    st = time()
    for i in range(100):
        res_o = coeff(expr2,z*sp.sin(x)**2*sp.cos(y)**2)
    en = time()
    print 'o', en-st
    
    st = time()
    for i in range(100):
        res_s = expr2.coeff(z*sp.sin(x)**2*sp.cos(y)**2)
    en = time()
    print 's', en-st
    
    assert res_s == res_o;

def test_order():
    x,y,z = sp.symbols('xyz')
    
    expr1 = sp.Add(evaluate=False,*[((x+y+2)**5).expand() for i in range(10)])
    expr2 = sp.Add(*[((x+y+2)**5).expand() for i in range(10)])
    
    print '1 should be much slower than 2'
    
    expr3 = sp.Add(*[((i+i*x**i)*(z*y)**10) for i in range(25)])
    expr4 = sp.Add(*[((i+i*x**i)*(z*y)**10) for i in range(25)]).expand()
    
    print '3 should be slightly slower than 4'

    st = time()
    for i in range(100):
        res1 = expr1.coeff(x*y**3)
    en = time()
    print '1', en-st
    
    st = time()
    for i in range(100):
        res2 = expr2.coeff(x*y**3)
    en = time()
    print '2', en-st
    
    st = time()
    for i in range(100):
        res3 = expr3.coeff(1+x)
    en = time()
    print '3', en-st   
    
    st = time()
    for i in range(100):
        res4 = expr4.coeff(1+x)
    en = time()
    print '4', en-st    

#@profile
def test_coeff():
    x,y,z = sp.symbols('xyz')
    
    expr = sp.Add(*[(x+y)**i*((i+i*x**i)*(z*y)**10) for i in range(10)]).expand()
    
    expr.coeff(x**2*y*z)

#@profile
def test_collect():
    x,y,z = sp.symbols('xyz')
    
    expr = sp.Add(*[(x+y)**i*((i+i*x**i)*(z*y)**10) for i in range(10)]).expand()
    for i in range(50):
        sp.collect(expr,x**2*y*z)

#@profile
def test_subs():
    x,y,z = sp.symbols('xyz')
    
    expr = sp.Add(*[(x+y)**i*((i+i*x**i)*(z*y)**10) for i in range(10)]).expand()
    for i in range(10):
        expr.subs(x**2*y*z,2)

def test_bins():
    # expr
    # our code
    # sympy code
    x,y,z = sp.symbols('xyz')
    
    expr1 = (((1+x)+2*(y-4)+3*(z/7))**10).expand()
    
    st = time()
    for i in range(25):
        res = coeff_bins(expr1,[x**i for i in range(10)])
    en = time()
    print en-st
    
    st = time()
    for i in range(25):
        for j in range(10):
            res = expr1.coeff(x**j)
    en = time()
    print en-st

def test_zeros():
    
    st = time()
    for i in range(10000):
        np.zeros(1000,dtype=object)
    en = time()
    print en-st
    
    st = time()
    for i in range(10000):
        [0 for i in range(1000)]
    en = time()
    print en-st


 
#test_coeff()
#test_collect()
#test_subs()
#test_bins()

if __name__ == "__main__":
#    test_func()
#    test_collect()
#    test_subs()
#    test_bins()
    test_zeros()
    