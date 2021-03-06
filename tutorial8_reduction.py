# Halide tutorial lesson 7.

# This lesson illustrates reductions

# We will compute the average RGB value of an image
# that is, we will return a 1D array with only 3 values

# One important thing to remember from this lesson is that there is never 
# any explicit for loop in Halide, whether is't the loop over output pixels
# or the reduction loop over the input pixels for the reduction. It's all implicit

import os, sys
from halide import *

#Python Imaging Library will be used for IO
import imageIO

def main():
    
## SUM ONLY

    # As usual, let's load an input
    im=imageIO.imread('rgb.png')    
    # and create a Halide representation of this image
    input = Image(Float(32), im)

    # Next we declaure the Vars and Func
    x, y, c = Var(), Var(), Var() 
    mySum = Func()

    # The central tool to express a reduction is a reduction domain, called RDom
    # it corresponds to the bounds of the reduction loop you would write in an 
    # imperative language. Here we want to iterate over a 2D domain corresponding 
    # to the whole image. 
    # Note however that we have decided to compute a different sum for each channel, 
    # So we will not reduce over channels. 
    r = RDom(0,    input.width(), 0,    input.height())                
    # careful, RDom are defined as base, extent, not min, max.
    # This means that RDom(a, b) goes from a to a+b (obviously not an issue here)

    # Given a reduction domain, we define the Expr that we will sum over, in this
    # case the pixel values. By construction, the first and second dimension of a 
    # reduction domain are called x and y. In this case they happen to correspond 
    # to the image x and y coordinates but they don't have to. 
    # Note that x & y are the reduction variables but c is a normal Var.
    # this is because our sum is over x,y but not over c. There will be a different 
    # sum for each channel. 
    val=input[r.x, r.y, c]

    # A reduction Func first needs to be initialized. Here, our sum gets initialized to 0
    # Note that the function domain here is only the channel. 
    mySum[c]=0.0

    # Finally, we define what the reduction should do for each reduction value. 
    # In this case, we eant to add each reduction value to the output
    # This is called the update function, and it's going to be called for each 
    # location in the RDom. 
    # You never write an explicit loop over the RDom, Halide does it for you. 
    mySum[c] +=val

    # We now call realize() to compile and execute. 
    output = mySum.realize(input.channels());

    outputNP=numpy.array(Image(output))
    print outputNP

    # equivalent Python code

    out = numpy.empty((3));

    # first loop to initialize the reduction
    for c in xrange(input.channels()):
        out[c]=0.0
    # VERY IMPORTANT : 
    # Note that loops for the reduction variables are outermost. 
    # whereas c in this case is innermost. 
    for ry in xrange(0, input.height()):
        for rx in xrange(0, input.width()):
            for c in nxrange(input.channels()):
                #update function
                out[c] += input[rx, ry, c]



## AVERAGE

    # As usual, let's load an input
    im=imageIO.imread('rgb.png')    
    # and create a Halide representation of this image
    input = Image(Float(32), im)

    # Next we declaure the Vars
    x, y, c = Var('x'), Var('y'), Var('c') 
    
    # Next we declare our Funcs. 
    # We will have one for the sum before division by the number of pixels, 
    # and one for the final average. 

    mySum = Func('mySum')
    myAverage = Func('myAverage')

    # The central tool to express a reduction is a reduction domain, called RDom
    # it corresponds to the bounds of the reduction loop you would write in an 
    # imperative language. Here we want to iterate over a 2D domain corresponding 
    # to the whole image. Note however that we will note reduce over channels. 
    r = RDom(0,    input.width(), 0,    input.height(), 'r')                

    # Given a reduction domain, we define the Expr that we will sum over, in this
    # case the pixel values. By construction, the first and second dimension of a 
    # reduction domain are called x and y. In this case they happen to correspond 
    # to the image x and y coordinates but they don't have to. 
    # Note that x & y are the reduction variables but c is a normal Var.
    # this is because our sum is over x,y but not over c. There will be a different 
    # sum for each channel. 
    val=input[r.x, r.y, c]

    # A reduction Func first needs to be initialized. Here, our sum gets initialized at 0
    # Note that the function domain here is only the channel. 
    mySum[c]=0.0

    # Finally, we define what the reduction should do for each reduction value. 
    # In this case, we eant to add each reduction value

    # update function is going to be called....
    # can be any Expr  as a function of mySum[c] and the RDom
    mySum[c] = mySum[c]+val
    # as usual, the short version is 
    # mySum[c] += input[r.x, r.y, c]

    # Finally, we define our final Func as the sum divided by the image number of pixels. 
    myAverage[c]=mySum[c]/(input.width()*input.height())

    # As usual, all we have done so far is create a Halide internal representation.
    # We now call realize() to compile and execute. 
    output = myAverage.realize(input.channels());

    outputNP=numpy.array(Image(output))
    print outputNP

    # equivalent Python code

    tmp = numpy.empty((3));
    out = numpy.empty((3));

    # loop for myAverage
    for c in xrange(input.channels()):
        # inline schedule: dump the code for 

        tmp[c]=0.0
    for ry in xrange(0, input.height()):
        for rx in xrange(0, input.width()):
            for c in nxrange(input.channels()):
                tmp[c]+=input[rx, ry, c]
    for c in xrange(input.channels()):
        out[c]=tmp[c]/(input.width()*input.height())


# Misc discussion with jrk

    x, y, c = Var('x'), Var('y'), Var('c')     
    mySum = Func('mySum')
    myAverage = Func('myAverage')
    helper=Func()
    r = RDom(0,    input.width(), 0,    input.height(), 'r')

    val=input[r.x, r.y, c]


    mySum[c]=0.0    
    mySum[c]+=val

    mySuperSum[c]=mySum[c]

    # Finally, we define our final Func as the sum divided by the image number of pixels. 
    myAverage[c]=mySum[c]/(input.width()*input.height())

    # As usual, all we have done so far is create a Halide internal representation.
    # We now call realize() to compile and execute. 
    output = myAverage.realize(input.channels());

    outputNP=numpy.array(Image(output))
    print outputNP


    # 8 bit histogram





    # let's extend the example above with an extra reduction stage that compute the 
    # average across the three channels

    avAcrossChannel = Func('avAcrossChannel')
    r = RDom(0, input.channels(), 'rChannel')
    val=myAverage[r.x]
    avAcrossChannel[]= 0
    avAcrossChannel[]+=val




#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

# exercises


