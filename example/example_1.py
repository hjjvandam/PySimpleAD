#   Copyright 2024 Hubertus van Dam 
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#      
#       http://www.apache.org/licenses/LICENSE-2.0
#      
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import sys

def func_f(a,b):
    '''A simple function that we want to minimize'''
    f = a*a + b*b
    return f

def pick_start(a,b):
    '''Pick random starting values

    Pick random samples from a uniform distribution of values between
    the values of low and high
    '''
    rng = np.random.default_rng()
    low = -10.0
    high = 10.0
    v = rng.random()*(high-low)+low
    a.set(v)
    v = rng.random()*(high-low)+low
    b.set(v)
    return (a,b)

def minimize(func,a,b):
    '''Minimize function "func"'''
    print(f"starting from {a.get_val()},{b.get_val()}")
    x = a
    y = b
    f = func(x,y)
    dfda = f.get_grad(a)
    dfdb = f.get_grad(b)
    fab  = f.get_val()
    maxd = max(abs(dfda),abs(dfdb))
    ii = 0
    while maxd > 1.0e-5:
        ii += 1
        print(f"iteration, function, gradient = {ii}, {fab}, {maxd}")
        x = x - dfda/4.0
        y = y - dfdb/4.0
        f = func(x,y)
        dfda = f.get_grad(a)
        dfdb = f.get_grad(b)
        fab  = f.get_val()
        maxd = max(abs(dfda),abs(dfdb))
    print(f"the minimum of f is at {x.get_val()},{y.get_val()}")
    print("the function value is:")
    f.print()

def run():
    '''Run a calculation'''
    from pysimplead import PySAD
    x1 = PySAD(name="x",value=0.0,maxvar=2)
    x2 = PySAD(name="y",value=0.0)
    x1,x2 = pick_start(x1,x2)
    minimize(func_f,x1,x2)
    sys.modules.pop("pysimplead")

if __name__ == '__main__':
    run()

