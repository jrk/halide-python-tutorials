# Halide tutorial lesson 4.

# This lessons demonstrates clamping to deal with boundary pixels for stencil
# operations. When using neighboring pixel at the border of the image, we might
# use indices that are negative or beyond the image size and would return an error.  
# We will simply clamp the coordinates.

# This tutorial is a minor modification of the previous one. 
# We will compute The magnitude of the gradient of an image, but the output will be
# the same size as the input 

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

    ##### NEW CODE ####
    
    # In addition to these stages, we need a clamping Func that can deal with x, y,
    # locations outside the original image. For this we use the function clamp(x, min, max)
    # which returns x if it's in the interval, min if it's below and max if it's above.
    # In terms of images, it returns the pixel on the dge nearest to x, y
    clamped = Func('clamped') 
    clamped[x, y, c] = input[clamp(x, 0, input.width()-1),
                             clamp(y, 0, input.height()-1), 
                             c]

    ##### MODIFIED CODE ####

    # We can now safely define our horizontal gradient using finite difference
    # The value at a pixel is the input at that pixel minus its left neighbor.
    # Note how we now use the more direct definition of Funcs without declaring
    # intermediate Exprs
    gx[x,y,c]=clamped[x,y,c]-clamped[x-1,y,c]
    # Similarly define the vertical gradient. 
    gy[x,y,c]=clamped[x,y,c]-clamped[x,y-1,c]

    # Finally define the gradient magnitude as the Euclidean norm of the gradient vector
    # We use overloaded operators and functions such as **, + and sqrt
    # Through he magic of metaprogramming, this creates the appropriate algebraic tree
    # in Halide representation
    # Most operators and functions you expect are supported.
    # Check the documentation for the full list. 
    gradientMagnitude[x,y,c]= sqrt(gx[x,y,c]**2 + gy[x,y,c]**2)
    
    # As usual, all we have done so far is create a Halide internal representation.
    # No computation has happened yet.
    # We now call realize() to compile and execute. 
    output = gradientMagnitude.realize(input.width(), input.height(), input.channels());

    outputNP=numpy.array(Image(output))
    imageIO.imwrite(outputNP, gamma=1.0)

    print 'success!'

#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

# exercises
# compute a Sobel gradient
# don't try to be too smart, brute-force it.
# or you can be smart and meta-program it

def clamp(a, mini, maxi):
    if a<mini: a=mini
    if a>maxi: a=maxi
    return a
def clamped(y, x, im):
    return im[clamp(y, 0, lumi.shape[0]),
              clamp(x, 0, lumi.shape[1])]

def sobelMagnitude(lumi):
    '''lumi has a signle channel'''
    gx=numpy.empty(lumi.shape)
    for y in xrange(lumi.shape[0]):
        for x in xrange(lumi.shape[1]):
            gx[y,x]= (- pix(y-1, x-1, lumi) + pix(y-1, x+1, lumi) 
                     - 2*pix(y, x-1, lumi) + 2*pix(y, x+1, lumi) 
                     - pix(y+1, x-1, lumi) + pix(y+1, x+1, lumi) )/4.0
    gy=numpy.empty(lumi.shape)
    for y in xrange(lumi.shape[0]):
        for x in xrange(lumi.shape[1]):
            gx[y,x]= (- pix(y-1, x-1, lumi) + pix(y+1, x-1, lumi) 
                     - 2*pix(y-1, x, lumi) + 2*pix(y+1, x, lumi) 
                     - pix(y-1, x+1, lumi) + pix(y+1, x+1, lumi) )/4.0
    mag=numpy.empty(lumi.shape)
    for y in xrange(lumi.shape[0]):
        for x in xrange(lumi.shape[1]):
            mag[y,x]=sqrt(gx[y,x]**2+gy[y,x]**2)
    return mag



