# PySimpleAD
A simple implementation of automatic differentiation in Python for educational purposes

## Introduction
Automatic Differentiation (AD) created by Wengert [^1] from the realization that every mathematical
operation can be extended to generate its derivatives by simply applying the chain rule. As
a result any computer program, which is nothing more than a sequence of mathematical 
operations, can be extended to generate the derivatives of the functions it evaluates. 
This can be accomplished simply by applying the chain rule to every operation. 

As the paper by Wengert was published in 1964 it predated the first language standard
for Fortran from 1966. Even Fortran 66 did not support user defined data types or
operator overloading. Hence the examples in the paper rely on replacing operators and
intrinsic functions with subroutines. For example, if 
```math
     y = x1*x2+sin(x2*x3)
```
then calculating d<i>y</i>/d<i>t</i> becomes
```fortran
      SUBROUTINE FUN (X1, X2, X3, Y)
      DIMENSION X1(2), X2(2), X3(2), Y(2), Z1(2), Z2(2), Z3(2)
      CALL PROD (X1, X2, Z1)
      CALL PROD (X2, X3, Z2)
      CALL SINE (Z2, Z3)
      CALL ADD (Z1, Z3, Y)
      RETURN
      END 
```
clearly this implementation is neither easy to read, nor is it likely to be fast as every
subroutine call incurs a cost. To deal with this situation a number of source-to-source 
compilers have been developed that add automatic differentiation to existing code.
Examples of such compilers are ADIC, ADIFOR, DAFOR, and ADiMat [^2].

More recent programming languages enabled operator overloading. This includes C++, Fortran,
and Python. Using operator overloading the equation can be kept in its original form, just
the implementation of the operators and intrinsic functions are replaced by ones that
evaluate the derivatives in addition to the expression itself. This way the readability of 
the code can be maintained while adding derivatives. 

## The automatic differentiation library

This repo implements a very simple automatic differentation library. The repo is 
structured as:

- PySimpleAD
  - pysimplead
    - \_\_init\_\_.py  - The automatic differentiation library. 
  - tests - The Python unit tests to be run under the control of the `unittest` module.
    - \_\_init\_\_.py  - just turn this directory into a Python module
    - test\_create.py  - creation of variables tests
    - test\_add.py     - addition tests
    - test\_comp.py    - comparison tests
    - test\_mul.py     - multiplication tests
    - test\_pow.py     - exponentiation (power) tests
    - test\_trigon.py  - trigonal function tests
  - example - examples uses of the automatic differentiation library
    - \_\_init\_\_.py  - just turn this directory into a Python module
    - example\_1.py    - as simple example code, the run() method runs the example
  - run.py - a Python script to run the example. Just invoke ./run.py

Some basic terminology to help with the discussion:

- independent variables - variables that do not depend on any other variables
- dependent variables - variables that are computed from other variables
- active independent variables - independent variables that we differentiate with respect to
- inactive independent variables - independent variables that are not included in the differentiation

For example if I want to calculate the radius in 2D space I can write that as
```
   a = 2
   r(x,y) = (x^a + b^a)**(1.0/a)
```
Typically if I am interested in derivatives I would probably want to differentiate `r(x,y)`
with respect to `x` and `y`, so they would be active independent variables. In general
I would not want to differentiate with respect to `a` as that is essentially a constant
in the definition of the radius. So `a` would be an inactive independent variable. In
practice which variables are active and inactive ones even within the same equation might
depend on the question you want to answer.

This library always differentiates with respect to all active independent variables. It
does not keep track of which variables are not involved in sub-expressions. For cases
with many active independent variables explicitly keeping track of this could be beneficial
from both a performance and memory usage point of view. 

If an automatic differentation library always differentiates with respect to all active
independent variables then there is no performance gain to be had from chosing between
forward and backward differentiation. So this library only implements the simpler forward
differentiation approach. For simplicity it also only implements first order derivatives.
While keeping track of active independent variables the following holds with respect to the
efficiency of forward and backware differentation approaches [^3]:

- Forward differentiation is more efficient for functions f : R<sub>n</sub> → R<sub>m</sub> with n ≪ m
- Backward differentation is more efficient for functions f : R<sub>n</sub> → R<sub>m</sub> with n ≫ m

In other words the approach that keeps the number of active independent variables smallest
for longest is the one that is going to give the best performance. The only complicating 
factor for backward differentiation is that you need to store the expression graph so you
can evaluate it backwards to compute the derivatives. This adds complexity to the code and
can requires additional memory.

Note that Python is dynamically typed and so normally it cannot distinguish between
different implementations of a function like, for example `sin(x)`, based on the type
of `x`. For automatic differentiation that is a problem that is handled with the
multiple dispatch module.

## Further reading

The automatic differentiation community has created a website to collect a broad range
of projects and their outcomes [^4].

## References

[^1]: R. E. Wengert, _A simple automatic derivative evaluation program_, 
      Communications of the ACM 7 (8) (1964) 463.
      doi:<a href="http://dx.doi.org/10.1145/355586.364791">10.1145/355586.364791</a>.

[^2]: <a href="http://www.autodiff.com/">http://www.autodiff.com/</a> [accessed July 24, 2024].

[^3]: <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">https://en.wikipedia.org/wiki/Automatic_differentiation</a> [accessed July 24, 2024].

[^4]: _Community Portal for Automatic Differentiation_ <a href="https://www.autodiff.org/">https://www.autodiff.org/</a> [accessed July 24, 2024].
