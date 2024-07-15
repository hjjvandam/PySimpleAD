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
then calculating d_y_/d_t_ becomes
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

## Further reading

The automatic differentiation community has created a website to collect a broad range
of projects and their outcomes [^3].

## References

[^1]: R. E. Wengert, _A simple automatic derivative evaluation program_, 
      Communications of the ACM 7 (8) (1964) 463.
      doi:<a href="http://dx.doi.org/10.1145/355586.364791">10.1145/355586.364791</a>.

[^2]: <a href="http://www.autodiff.com/">http://www.autodiff.com/</a>.

[^3]: _Community Portal for Automatic Differentiation_ <a href="https://www.autodiff.org/">https://www.autodiff.org/</a>.
