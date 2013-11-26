# Halide tutorial lesson 8.

# This lesson illustrates convolution using reduction


import os, sys
from halide import *
import time
import numpy

#Python Imaging Library will be used for IO
import imageIO

def boxBlur(im, indexOfSchedule, tileX=128, tileY=128):

    kernel_width=5
    input = Image(Float(32), im)

    x, y, c = Var('x'), Var('y'), Var('c') #declare domain variables

    blur_x = Func('blur_x') 
    blur_y = Func('blur_y')
    blur   = Func('blur')

    # declare and define a clamping function that restricts x,y to the image size
    # boring but necessary
    clamped = Func('clamped') 
    clamped[x, y, c] = input[clamp(x, 0, input.width()-1),
                          clamp(y, 0, input.height()-1), c]

    rx = RDom(0,    kernel_width, 'rx')                            
    blur_x[x,y,c] = 0.0
    blur_x[x,y,c] += clamped[x+rx.x-kernel_width/2, y, c] 

    clampedBlurx = Func('clampedBlurx') 
    clampedBlurx[x, y, c] = blur_x[clamp(x, 0, input.width()-1),
                              clamp(y, 0, input.height()-1), c]

    ry = RDom(0,    kernel_width, 'ry')                
    blur_y[x,y,c] = 0.0
    blur_y[x,y,c] += clampedBlurx[x, y+ry.x-kernel_width/2, c]
    # note rx
    
    blur[x,y,c] = blur_y[x,y,c]/(kernel_width**2)

    #schedule attempt
    if indexOfSchedule==0: 
        print '\n ', 'default schedule'

    if indexOfSchedule==1: 
        print '\n', 'root first stage'
        clampedBlurx.compute_root()

    vectorWidth=8

    if indexOfSchedule==2: 
        print '\n', 'tile ', tileX,'x', tileY, ' + interleave'
        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')
        blur.tile(x, y, xo, yo, xi, yi, tileX, tileY)
        clampedBlurx.compute_at(blur, xo)

    if indexOfSchedule==3: 
        print '\n', 'tile ', tileX,'x', tileY, '+ parallel'
        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')
        blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        clampedBlurx.compute_at(blur, xo)

    if indexOfSchedule==4: 
        print '\n', 'tile ', tileX,'x', tileY, '+ vector'
        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')
        blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).vectorize(xi, vectorWidth)
        clampedBlurx.compute_at(blur, xo).vectorize(x, vectorWidth)

    if indexOfSchedule==5: 
        print '\n', 'tile ', tileX,'x', tileY, '+ parallel+vector'
        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')
        blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo).vectorize(xi, vectorWidth)
        clampedBlurx.compute_at(blur, xo).vectorize(x, vectorWidth)

    if indexOfSchedule==6: 
        print '\n', 'tile ', tileX,'x', tileY, ' + parallel+vector without interleaving'
        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')
        blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo).vectorize(xi, vectorWidth)
        clampedBlurx.compute_root().tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo).vectorize(xi, vectorWidth)


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
    output=None
    for i in xrange(7):
        if i<2:
            output, dt=boxBlur(im, i)
        else:
            for tileY in [256]: 
                for tileX in [256]: 
                    output, dt=boxBlur(im, i, tileX, tileY)
    
    #outputNP=numpy.array(Image(output))
    #imageIO.imwrite(outputNP)

    #numpy.save('Input/hk.npy', im)

#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

# exercises


