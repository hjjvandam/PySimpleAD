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
        if maxvar:
            if PySAD.maxvar != 0 and PySAD.maxvar != maxvar:
                raise RuntimeError(f"the maximum number of independent variables has already been set to {PySAD.maxvar} and cannot be reset to {maxvar}")
            else:
               PySAD.maxvar = maxvar
        curvar = None
        self.varnum = -1 # Independent variables will have variable numbers >= 0
        if name:
            PySAD.names.append(name)
            curvar = PySAD.numvar
            self.varnum = curvar
            PySAD.numvar += 1
            if PySAD.numvar > PySAD.maxvar:
                raise RuntimeError(f"the number of independent variables {PySAD.numvar} exceeds the maximum number of independent variables {PySAD.maxvar}")
            if not value:
                raise RuntimeError(f"a new independent variable must be given a value")
        if value:
            self.values.append(float(value))
        else:
            self.values.append(float(0))
        array = np.zeros((PySAD.maxvar))
        if curvar != None:
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
            x2.values[1] = (x1.values[0] * (self.values[0] ** (x1.values[0]-1)) * self.values[1] +
                            self.values[0] ** x1.values[0] * x1.values[1])
        else:
            x2.values[0] = self.values[0] ** x1
            x2.values[1] = x1 * (self.values[0] ** (x1-1)) * self.values[1]
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
def sin(x):
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
def cos(x):
    '''The sin() function'''
    x2 = PySAD()
    x2.values[0] =  math.cos(x1.values[0])
    x2.values[1] = -math.sin(x1.values[0])*x1.values[1]
    return x2

@dispatch(float)
def cos(x):
    '''The sin() function'''
    return math.cos(x)
