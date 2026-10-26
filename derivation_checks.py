#!/usr/bin/env python3
"""Sanity checks for formulas in the manuscript; """
import numpy as np
import sympy as sp
x,y=sp.symbols('x y',real=True)
gamma=sp.Rational(3,2)
for n in range(1,9):
    actual=sp.diff(sp.exp(-gamma*(x-y)**2),x,n,y,n).subs(y,x)
    expected=gamma**n*sp.factorial(2*n)/sp.factorial(n)
    assert sp.simplify(actual-expected)==0
rng=np.random.default_rng(88)
for _ in range(10000):
    b0,b1=rng.normal(size=2)
    lift_H2=(b1*b1-b0*b1+b0*b0/3+b0*b0)**.5
    assert lift_H2 <= 2*(abs(b0)+abs(b1))+1e-14
print('All Gaussian derivative identities (orders 1..8) and boundary-lift sanity checks passed.')
