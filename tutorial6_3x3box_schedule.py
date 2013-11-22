# Halide tutorial lesson 6.

# This lessons demonstrates the scheduling of a two-stage pipeline
# We will see how to schedule one (earlier) producer stage with respect to
# its consumer.

# We will implement a separable 3x3 box blur and explore different schedules

# we should also teach them inline

# should I use it for normalization ?


myMax=max
mySum=sum

import os, sys
from halide import *
import imageIO
import time
import numpy

def runAndMeasure(myFunc, w, h, nTimes=5):
    L=[]
    output=None
    myFunc.compile_jit()
    for i in xrange(nTimes):
        t=time.time()
        output = myFunc.realize(w,h)
        L.append (time.time()-t)
    #print 'running times :', L
    print 'best: ', numpy.min(L), 'average: ', numpy.mean(L)
    return numpy.min(L)        

def main():
    #load the input, convert to single channel and turn into Halide Image
    inputP=imageIO.imread('rgb.png')[:,:,1] #we'll just use the green channel
    input=Image(Float(32), inputP)
    numpy.transpose(inputP) #flip x and y to follow Halide indexing and
    #make the code look more similar between Python and Halide
    
    ##Declarations of Halide variable and function names
    
    x, y = Var('x'), Var('y') #declare domain variables
    blur_x = Func('blur_x') #declare the horizontal blur function
    blur_y = Func('blur_y') #declare the vertical blur function


    #### ALGORITHM ###
    
    # This is a separable box blur. First blur along x, then along y 
    # we will ignore boundary issues and crop by two pixels n both dimensions
    # this is why we don't center the kernel
    
    blur_x[x,y] = (input[x,y]+input[x+1,y]+input[x+2,y])/3
        #definition of the horizontal blur declared above.
        #simply average  three neighbors in the input image
    
    blur_y[x,y] = (blur_x[x,y]+blur_x[x,y+1]+blur_x[x,y+2])/3
        #definition of the vertical blur declared above
        #simply average  three neighbors of the blur_x output

#### SCHEDULE 1 : ROOT ####

    # C or Python-style schedule: root. We compute the first stage (blur_x)
    # on the entire image before computing the second stage.
    blur_y.compute_root()
    blur_x.compute_root()
    # this schedule has bad locality because the data produced by blur_x
    # are long ejected from the cache by the time blur_y needs them
    # it also doesn't have any parallelism
    # But it doesn't introduce any extra computation
    
    # Finally compile and run. 
    # Note that subtract two to the height and width to avoid boundary issues
    print 'schedule 1:'
    refTime = runAndMeasure(blur_y, input.width()-2, input.height()-2)

    # equivalent Python code:
    if False: # I just want to save time and not execute it
        width, height = input.width()-2, input.height()-2
        out=numpy.empty((width, height))
        # compute blur_x at root
        tmp=numpy.empty((width, height))
        for y in xrange(height):
            for x in xrange(width):
                tmp[x,y]=(inputP[x,y]+inputP[x+1,y]+inputP[x+2,y])/3
        #compute blur_y
        for y in xrange(height):
            for x in xrange(width):
                out[x,y] = (blur_x[x,y]+blur_x[x,y+1]+blur_x[x,y+2])/3



#### SCHEDULE 2 : INLINE ####

    # redefine everything to start fresh
    x, y = Var('x'), Var('y') 
    blur_x, blur_y  = Func('blur_x') , Func('blur_y') 
    blur_x[x,y] = (input[x,y]+input[x+1,y]+input[x+2,y])/3
    blur_y[x,y] = (blur_x[x,y]+blur_x[x,y+1]+blur_x[x,y+2])/3

    # schedule

    # In this schedule, we compute values for blur_x each time they are
    # required by blur_y (inline). This means excellent locality between
    # producer and consumer, since blur_x values are produced as needed and
    # directly consumed by blur_y.
    # However, this introduces significant redundant computation since each
    # blur_x value is recomputed 3 times, once for each blur_y computation
    # that needs it
    
    blur_y.compute_root()
    blur_x.compute_inline()
    # inline is the default schedule, however. This makes it easy to express
    # long expressions as chains of Funcs without paying a performance price 
    # in general, inline is good when the dependency is a single pixel
    # (no branching factor that would introduce redundant computation)

    print 'schedule 2:'
    t = runAndMeasure(blur_y, input.width()-2, input.height()-2)
    print 'speedup: ', refTime/t

    # equivalent Python code:

    if False: # I just want to save time and not execute it
        width, height = input.width()-2, input.height()-2                        
        out=numpy.empty((width, height))
        #compute blur_y
        for y in xrange(height):
            for x in xrange(width):
                # compute blur_x inline, inside of blur_y
                 out[x,y] = ((inputP[x,y]+inputP[x+1,y]+inputP[x+2,y])/3
                           + (inputP[x,y+1]+inputP[x+1,y+1]+inputP[x+2,y+1])/3
                           + (inputP[x,y+2]+inputP[x+1,y+2]+inputP[x+2,y+2])/3 )/3

    # In effect, this schedule turned a separable blur into the brute force
    # 2D blur
    # The compiler would also probably merge the various divisions by 3
    # into a single division by 9

#### SCHEDULE 3 : TILED AND MERGED ####

    # This is a high-performance schedule that computes computation
    # in tiles and merges the two stages of the pipeline within a tile

    # redefine everything to start fresh
    x, y = Var('x'), Var('y') 
    blur_x, blur_y = Func('blur_x'), Func('blur_y') 
    blur_x[x,y] = (input[x,y]+input[x+1,y]+input[x+2,y])/3
    blur_y[x,y] = (blur_x[x,y]+blur_x[x,y+1]+blur_x[x,y+2])/3

    # schedule
    
    # Declare the inner tile variables 
    xo, yo, xi, yi = Var('xo'), Var('yo'), Var('xi'), Var('yi')

    # First schedule the last (output) stage
    # In Halide, the schedule is always driven by the output
    # Earlier stages are scheduled with respect to later schedules
    # That is, we schedule a producer with respect to its consumer(s)
    
    blur_y.tile(x, y, xo, yo, xi, yi, 256, 32)  #compute in tiles of 256x32

    # We now specify when the earlier (producer) stage blur_x gets evaluated.
    # We decide to compute it at the tile granularity of blur_y and use the
    # compute_at method. 
    # This means that blur_x will be evaluated in a loop nested inside the
    # 'xo' outer tile loop of blur_y
    # note that we do not need to specify yo, xi, yi and they are directly
    # inherited from blur_y's scheduling
    # More importantly, Halide performs automatic bound inference and enlarges
    # the tiles to make sure that all the inputs needed for a tile of blur_y
    # are available. In this case, it means making the blur_x tile one pixel
    # larger above and below to accomodate the 1x3 vertical stencil of blur_y
    # This is all done under the hood and the programmer doesn't need to worry
    # about it
    blur_x.compute_at(blur_y, x) 

    # This schedule achieves better locality than root but with a lower redundancy
    # than inline. It still has some redundancy because of the enlargement at tile
    # boundaries 

    print 'schedule 3:'
    t = runAndMeasure(blur_y, input.width()-2, input.height()-2)
    print 'speedup: ', refTime/t

    # corresponding python code:
    if False: # I just want to save time and not execute it
        width, height = input.width()-2, input.height()-2
        out=numpy.empty((width, height))
        for yo in xrange(ceil(height/32.0)): #this for loop is parallel
            for xo in xrange(ceil(width/256.0)):
                # first compute blur_x
                # allocate a temporary buffer
                tmp=numpy.empty((256, 32)) 
                for yi in xrange(32):
                    y=min(yo*32+yi, height)
                    for xi in xrange(256/8):
                        x=min(xo*256+xi, width)
                        tmp[x,y]=(inputP[x,y]+inputP[x+1,y]+inputP[x+2,y])/3
                #compute blur_y
                for yi in xrange(32):
                    y=min(yo*32+yi, height)
                    for xi in xrange(256/8):
                        x=min(xo*256+xi, width)
                        out[x,y] = (blur_x[x,y]+blur_x[x,y+1]+blur_x[x,y+2])/3


    # add scan

#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

