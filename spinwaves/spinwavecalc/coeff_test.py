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

if __name__ == "__main__":
    test_order()
    