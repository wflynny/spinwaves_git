import sympy

if __name__ == "__main__":
    e = sympy.Symbol('e',commutative=False)
    f = sympy.Symbol('f',commutative=False)
    s = sympy.Mul(*[sympy.Symbol('x%i'%i, commutative=False) for i in range(1000)])
    
    print s**100