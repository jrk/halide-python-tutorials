# Halide tutorial lesson 8.

# This lesson illustrates convolution using reduction


import os, sys
from halide import *
import time
import numpy

#Python Imaging Library will be used for IO
import imageIO

def convolution(im, indexOfBlur=0):

    input = Image(Float(32), im)
    kernel_width=5

    blur=Func('blur')  #just declaring the Python variable
    x, y, c = Var('x'), Var('y'), Var('c') #declare domain variables

    # declare and define a clamping function that restricts x,y to the image size
    # boring but necessary
    clamped = Func('clamped') 
    clamped[x, y, c] = input[clamp(x, 0, input.width()-1),
                              clamp(y, 0, input.height()-1), c]


    if indexOfBlur==0:

        r = RDom(0,    kernel_width, 0,    kernel_width, 'r')                            
        blur[x,y,c] = 0.0
        blur[x,y,c] += clamped[x+rx.x-kernel_width/2, y, c]

        #equivalent Python code
        if True: 
            out=numpy.empty([input.width(), input.height(), input.channels()])
            for y in xrange(input.height()):
                for x in xrange(input.width()):
                    for c in xrange(input.channels):
                        out[x,y,c]=0
            for ry in xrange(kernel_width):
                for rx in xrange(kernel_width):
                    for y in xrange(input.height()):
                        for x in xrange(input.width()):
                            for c in xrange(input.channels):
                                out[x,y,c]+=clampedInput[x,y,c]
            #Note how the reduction variables  are the outer loop

  if indexOfBlur==1:

        r = RDom(0,    kernel_width, 0,    kernel_width, 'r')                            
        blur[x,y,c] = 0.0
        blur[x,y,c] += clamped[x+rx.x-kernel_width/2, y, c]

        superBlur = Func('superBlur')
        superBlur[x,y,c]=blur[x,y,c]
        #equivalent Python code
        if True: 
            superBlur=numpy.empty([input.width(), input.height(), input.channels()])
            #loops for superBlur
            for y in xrange(input.height()):
                for x in xrange(input.width()):
                    for c in xrange(input.channels):
                        # Now inline blur
                        # For the now-constant values of x, y, c
                        # the width and height we need are just 1
                        tmp=numpy.empty([1,1,1])
                        for yi in xrange(1):
                            for xi in xrange(1):
                                for ci in xrange(1):
                                    tmp[xi,yi,ci]=0
                                    #
                        for ry in xrange(kernel_width):
                            for rx in xrange(kernel_width):
                                for yi in xrange(1):
                                    for xi in xrange(1):
                                        for ci in xrange(1):
                                            tmp[xi,yi,ci]+=clampedInput[x,y,c]
                                            #where xi=0, yi=0, ci=0
                        superBlur[x,y,c]=tmp[0,0,0]
                        #Note how the reduction variables  are the outer loop
       #equivalent Python code with empty loops removed
        if True: 
            superBlur=numpy.empty([input.width(), input.height(), input.channels()])
            #loops for superBlur
            for y in xrange(input.height()):
                for x in xrange(input.width()):
                    for c in xrange(input.channels):
                        tmp=0
                        for ry in xrange(kernel_width):
                            for rx in xrange(kernel_width):
                                tmp+=clampedInput[x,y,c]
                        superBlur[x,y,c]=tmp
    if indexOfBlur==2:
        r = RDom(0,    kernel_width, 0,    kernel_width, 'r')                            
        blur[x,y,c] = sum(clamped[x+rx.x-kernel_width/2, y, c])



    blur.compile_jit()
    t=time.time()
    numTimes=5
    for i in xrange(numTimes):
        output = blur.realize(input.width(), input.height(), input.channels())
    dt=time.time()-t
    print '           took ', dt/numTimes, 'seconds'

    return output, dt

def main():    
    #im=imageIO.imread('hk.png')
    im=numpy.load('Input/hk.npy')
    w=5
    k=numpy.ones([w,w])/(k**2)

    output=None
    for i in xrange(1):
        output, dt=boxBlur(im, k, i)
        
    #outputNP=numpy.array(Image(output))
    #imageIO.imwrite(outputNP)

    #numpy.save('Input/hk.npy', im)

#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

# exercises


