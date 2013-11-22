# Halide tutorial lesson 3.

# This lessons demonstrates clamping, the use of multiple input pixels per output,
# and a multi-stage pipeline

# We will compute The magnitude of the gradient of an image. 

import os, sys
from halide import *

#Python Imaging Library will be used for IO
import imageIO

def main():

    # This program defines a multi-stage Halide imaging pipeline
    # One stage computes the horixontal gradient of an image dI/dx
    # Another stage computes dI/dy (for all three channels of RGB in both cases)
    # The final stage computes the magnitude of the corresponding vector

    # We will compute the gradient with finite differences: dI/dx=I(x+1)-I(x)

    
    # As usual, let's load an input
    im=imageIO.imread('rgb.png')    
    # and create a Halide representation of this image
    input = Image(Float(32), im)

    # Next we declaure the Vars
    # We here give an extra argument to the Var constructor, an optional string that
    # can help for debugging by naming the variable in the Halide representation.
    # Otherwise, the names x, y, c are only known to the Python side
    x, y, c = Var('x'), Var('y'), Var('c') 
    
    # Next we declare the three Funcs corresponding to the various stages of the gradient .
    # Similarly, we pass strings to name them. 
    gx = Func('gx') 
    gy = Func('gy') 
    gradientMagnitude=Func('gradientMagnitude') 

    # Define our horizontal gradient Func using finite difference
    # The value at a pixel is the input at that pixel minus its left neighbor.
    # Note how we now use the more direct definition of Funcs without declaring
    # intermediate Exprs
    gx[x,y,c]=input[x+1,y,c]-input[x, y,c]
    # Similarly define the vertical gradient. 
    gy[x,y,c]=input[x,y+1,c]-input[x,y,c]

    # Finally define the gradient magnitude as the Euclidean norm of the gradient vector
    # We use overloaded operators and functions such as **, + and sqrt
    # Through he magic of metaprogramming, this creates the appropriate algebraic tree
    # in Halide representation
    # Most operators and functions you expect are supported.
    # Check the documentation for the full list.
    gradientMagnitude[x,y,c]= sqrt(gx[x,y,c]**2+gy[x,y,c]**2)
    
    # As usual, all we have done so far is create a Halide internal representation.
    # No computation has happened yet.
    # We now call realize() to compile and execute. 
    # You'll note that we subtracted 1 from the width and height to make sure that the
    # x+1 and y+1 neighbors always exist. We'll see a more general solution in the next tutorial
    output = gradientMagnitude.realize(input.width()-1, input.height()-1, input.channels());

    outputNP=numpy.array(Image(output))
    imageIO.imwrite(outputNP, gamma=1.0)
    print 'success!'

    return 0


#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

# no exercise
