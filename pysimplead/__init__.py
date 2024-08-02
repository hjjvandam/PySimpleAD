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
from multipledispatch import dispatch
import math

class PySAD:
    '''Class for automatic differentiation

    We use class global variables to keep track of the number
    of independent variables we have created (numvar),
    the maximum number of independent variables we allow for
    (maxvar), as well as the variable names (names).
    In the class methods these class global variables can be
    accessed as PySAD.maxvar, PySAD.numvar, and PySAD.names.

    Class instances will of course have their own settings. 
    self.values is a list of values indexed by the order of
    differentiation. I.e. self.values[0] is the value of the
    variable whereas self.values[1] is an array of first order
    derivatives.
    self.varnum is the variable number of this variable. For
    independent variables this number corresponds to the 
    position in the variable list. I.e. PySAD.names[self.varnum]
    is the name of the current independent variable. For dependent
    variables self.varnum is -1.
    '''
    maxvar = 0
    numvar = 0
    names = []
    def __init__(self,value=None,name=None,maxvar=None):
        '''Create a new AD instance

        The first time we create an instance we need to set the maximum
        number of independent variables. This determines the lenght of
        the derivative array. After this is set it should not be 
        changed. 

        When a name is provided a new independent variable is created.
        This requires storing the name in the names list and incrementing
        the number of independent variables.

        If no name is provided then a dependent variable is created.

        Finally PySAD(value=x) creates a constant represented as a
        dependent variable. This is needed for example to express
        2/x because PySAD.__truediv__ requires 2 to be represented
        as an PySAD instance. I.e. this needs to be written as
           a = PySAD(value=2)
           a/x
        This is a limitation of Python.
        '''
        self.values = []
        if not maxvar is None:
            if PySAD.maxvar != 0 and PySAD.maxvar != maxvar:
                raise RuntimeError(f"the maximum number of independent variables has already been set to {PySAD.maxvar} and cannot be reset to {maxvar}")
            else:
               PySAD.maxvar = maxvar
        curvar = None
        self.varnum = -1 # Independent variables will have variable numbers >= 0
                         # When varnum remains < 0 were are dealing with dependent variables
        if not name is None:
            PySAD.names.append(name)
            curvar = PySAD.numvar
            self.varnum = curvar
            PySAD.numvar += 1
            if PySAD.numvar > PySAD.maxvar:
                raise RuntimeError(f"the number of independent variables {PySAD.numvar} exceeds the maximum number of independent variables {PySAD.maxvar}")
            if value is None:
                raise RuntimeError(f"a new independent variable must be given a value")
        if not value is None:
            self.values.append(float(value))
        else:
            self.values.append(float(0))
        array = np.zeros((PySAD.maxvar))
        if not curvar is None:
            array[curvar] = 1.0
        self.values.append(array)

    def print(self):
        '''Print the contents of the current variable'''
        print("")
        if self.varnum < 0:
            print(f"f = {self.values[0]}")
            for ii in range(PySAD.numvar):
                print(f"df/d{PySAD.names[ii]} = {self.values[1][ii]}")
        else:
            print(f"{PySAD.names[self.varnum]} = {self.values[0]}")
            for ii in range(PySAD.numvar):
                print(f"d{PySAD.names[self.varnum]}/d{PySAD.names[ii]} = {self.values[1][ii]}")

    def set(self,value):
        '''Set the value for an independent variable'''
        if self.varnum < 0:
            raise RuntimeError("the variable is not an independent variable")
        self.values[0] = value

    def get_val(self):
        '''Get the value of this instance'''
        return self.values[0]

    def get_grad(self,independent):
        '''Get the derivative component corresponding to the given independent variable

        independent - the independent variable the derivative with respect to we want
                      to extract. I.e. if independent == x then this function returns
                      d self/d x.
        '''
        if not isinstance(independent,PySAD):
            raise RuntimeError(f"independent is not an instance of PySAD but of {type(independent)}")
        if independent.varnum < 0:
            raise RuntimeError(f"independent is not an independent variable")
        return self.values[1][independent.varnum]

    # Binary operators + - * / **

    def __add__(self,x1):
        '''Implement the addition operator

        We can have two scenarios:
        1. We are adding two AD variables, or
        2. We are adding an AD variable and a constant.
        '''
        x2 = PySAD()
        if isinstance(x1,PySAD):
            x2.values[0] = self.values[0] + x1.values[0]
            x2.values[1] = self.values[1] + x1.values[1]
        else:
            x2.values[0] = self.values[0] + x1
            x2.values[1] = self.values[1]
        return x2

    def __sub__(self,x1):
        '''Implement the subtraction operator

        We can have two scenarios:
        1. We are subtracting two AD variables, or
        2. We are subtracting an AD variable and a constant.
        '''
        x2 = PySAD()
        if isinstance(x1,PySAD):
            x2.values[0] = self.values[0] - x1.values[0]
            x2.values[1] = self.values[1] - x1.values[1]
        else:
            x2.values[0] = self.values[0] - x1
            x2.values[1] = self.values[1]
        return x2

    def __mul__(self,x1):
        '''Implement the multiplication operator

        We can have two scenarios:
        1. We are multiplying two AD variables, or
        2. We are multiplying an AD variable and a constant.
        '''
        x2 = PySAD()
        if isinstance(x1,PySAD):
            x2.values[0] = self.values[0] * x1.values[0]
            x2.values[1] = self.values[0] * x1.values[1] + self.values[1] * x1.values[0]
        else:
            x2.values[0] = self.values[0] * x1
            x2.values[1] = self.values[1] * x1
        return x2

    def __truediv__(self,x1):
        '''Implement the division operator

        We can have two scenarios:
        1. We are dividing two AD variables, or
        2. We are dividing an AD variable and a constant.
        '''
        x2 = PySAD()
        if isinstance(x1,PySAD):
            x2.values[0] = self.values[0] / x1.values[0]
            x2.values[1] = self.values[1] / x1.values[0] - self.values[0] * x1.values[1] / (x1.values[0] ** 2)
        else:
            x2.values[0] = self.values[0] / x1
            x2.values[1] = self.values[1] / x1
        return x2

    def __pow__(self,x1):
        '''Implement the multiplication operator

        We can have two scenarios:
        1. We are multiplying two AD variables, or
        2. We are multiplying an AD variable and a constant.
        '''
        x2 = PySAD()
        if isinstance(x1,PySAD):
            x2.values[0] = self.values[0] ** x1.values[0]
            x2.values[1] = ((self.values[0] ** x1.values[0]) * (x1.values[0] / self.values[0]) * self.values[1] +
                            (self.values[0] ** x1.values[0]) * math.log(self.values[0]) * x1.values[1])
        else:
            x2.values[0] = self.values[0] ** x1
            x2.values[1] = (self.values[0] ** x1) * (x1 / self.values[0]) * self.values[1]
        return x2

    # Unary operator -

    def __neg__(self):
        '''The negation operator'''
        x2 = PySAD()
        x2.values[0] = -self.values[0]
        x2.values[1] = -self.values[1]
        return x2

    # Comparison operators < <= == != >= >

    def __lt__(self,x1):
        '''Lesser than operator'''
        if isinstance(x1,PySAD):
           return self.values[0] < x1.values[0]
        else:
           return self.values[0] < x1

    def __le__(self,x1):
        '''Lesser than or equal operator'''
        if isinstance(x1,PySAD):
           return self.values[0] <= x1.values[0]
        else:
           return self.values[0] <= x1

    def __eq__(self,x1):
        '''Equal operator'''
        if isinstance(x1,PySAD):
           return self.values[0] == x1.values[0]
        else:
           return self.values[0] == x1

    def __ne__(self,x1):
        '''Not Equal operator'''
        return not (self == x1)

    def __ge__(self,x1):
        '''Greater or equal operator'''
        return not (self < x1)

    def __gt__(self,x1):
        '''Greater than operator'''
        return not (self <= x1)

    # Functions 

    def __abs__(self):
        '''Absolute value function'''
        x2 = PySAD()
        if self.values[0] < 0.0:
            x2.values[0] = -self.values[0]
            x2.values[1] = -self.values[1]
        else:
            x2.values[0] =  self.values[0]
            x2.values[1] =  self.values[1]
        return x2

@dispatch(PySAD)
def sqrt(x1):
    '''The sqrt() function'''
    x2 = PySAD()
    x2.values[0] = math.sqrt(x1.values[0])
    x2.values[1] = 0.5/math.sqrt(x1.values[0])*x1.values[1]
    return x2

@dispatch(float)
def sqrt(x):
    '''The sqrt() function'''
    return math.sqrt(x)

@dispatch(PySAD)
def exp(x1):
    '''The exp() function'''
    x2 = PySAD()
    x2.values[0] = math.exp(x1.values[0])
    x2.values[1] = math.exp(x1.values[0])*x1.values[1]
    return x2

@dispatch(float)
def exp(x):
    '''The exp() function'''
    return math.exp(x)

@dispatch(PySAD)
def log(x1):
    '''The log() function'''
    x2 = PySAD()
    x2.values[0] = math.log(x1.values[0])
    x2.values[1] = x1.values[1]/x1.values[0]
    return x2

@dispatch(float)
def log(x):
    '''The log() function'''
    return math.log(x)

@dispatch(PySAD)
def sin(x1):
    '''The sin() function'''
    x2 = PySAD()
    x2.values[0] = math.sin(x1.values[0])
    x2.values[1] = math.cos(x1.values[0])*x1.values[1]
    return x2

@dispatch(float)
def sin(x):
    '''The sin() function'''
    return math.sin(x)

@dispatch(PySAD)
def cos(x1):
    '''The cos() function'''
    x2 = PySAD()
    x2.values[0] =  math.cos(x1.values[0])
    x2.values[1] = -math.sin(x1.values[0])*x1.values[1]
    return x2

@dispatch(float)
def cos(x):
    '''The cos() function'''
    return math.cos(x)

@dispatch(PySAD)
def tan(x1):
    '''The tan() function'''
    x2 = PySAD()
    x2.values[0] = math.tan(x1.values[0])
    x2.values[1] = x1.values[1]/math.cos(x1.values[0])**2
    return x2

@dispatch(float)
def tan(x):
    '''The tan() function'''
    return math.tan(x)

@dispatch(PySAD)
def sinh(x1):
    '''The sinh() function'''
    x2 = PySAD()
    x2.values[0] = math.sinh(x1.values[0])
    x2.values[1] = x1.values[1]*math.cosh(x1.values[0])
    return x2

@dispatch(float)
def sinh(x):
    '''The sinh() function'''
    return math.sinh(x)

@dispatch(PySAD)
def cosh(x1):
    '''The cosh() function'''
    x2 = PySAD()
    x2.values[0] = math.cosh(x1.values[0])
    x2.values[1] = x1.values[1]*math.sinh(x1.values[0])
    return x2

@dispatch(float)
def cosh(x):
    '''The cosh() function'''
    return math.cosh(x)

@dispatch(PySAD)
def tanh(x1):
    '''The tanh() function'''
    x2 = PySAD()
    x2.values[0] = math.tanh(x1.values[0])
    x2.values[1] = x1.values[1]/math.cosh(x1.values[0])**2
    return x2

@dispatch(float)
def tanh(x):
    '''The tan() function'''
    return math.tanh(x)

@dispatch(PySAD)
def asin(x1):
    '''The asin() function'''
    x2 = PySAD()
    x2.values[0] = math.asin(x1.values[0])
    x2.values[1] = x1.values[1]/math.sqrt(1.0-x1.values[0]**2)
    return x2

@dispatch(float)
def asin(x):
    '''The asin() function'''
    return math.asin(x)

@dispatch(PySAD)
def acos(x1):
    '''The acos() function'''
    x2 = PySAD()
    x2.values[0] =  math.acos(x1.values[0])
    x2.values[1] = -x1.values[1]/math.sqrt(1.0-x1.values[0]**2)
    return x2

@dispatch(float)
def acos(x):
    '''The acos() function'''
    return math.acos(x)

@dispatch(PySAD)
def atan(x1):
    '''The atan() function'''
    x2 = PySAD()
    x2.values[0] = math.atan(x1.values[0])
    x2.values[1] = x1.values[1]/(1.0+x1.values[0]**2)
    return x2

@dispatch(float)
def atan(x):
    '''The atan() function'''
    return math.atan(x)

@dispatch(PySAD)
def asinh(x1):
    '''The asinh() function'''
    x2 = PySAD()
    t1 = 1.0 + x1.values[0]**2
    t12 = math.sqrt(t1)
    x2.values[0] = math.log(x1.values[0]+t12)
    x2.values[1] = x1.values[1]/t12
    return x2

@dispatch(float)
def asinh(x):
    '''The asinh() function'''
    return math.asinh(x)

@dispatch(PySAD)
def output(x1):
    x1.print()

@dispatch(float)
def output(x):
    print(x)
