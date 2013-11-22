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
    # The compiler would also  merge the various divisions by 3
    # into a single division by 9, or better a multiplication by the reciprocal. 
    # The reciprocal probably would stay in register, making everything blazingly fast. 
    # Compilers can be pretty smart.  

#### SCHEDULE 3 : TILING and INTERLEAVING ####

    # This is a good schedule (good locality, limited redundancy) that performs 
    # computation in tiles and interleaves the two stages of the pipeline within a tile

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
    # There is also a shorter version of the tile syntax that reuses the original 
    # Vars x, y for the outer tile indices: 
    # blur_y.tile(x, y, xi, yi, 256, 32)

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
    blur_x.compute_at(blur_y, xo) 

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
        for yo in xrange(ceil(height/32.0)): 
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


#### SCHEDULE 4 : TILING, INTERLEAVING, and PARALLELISM ####

    # This is a high-performance schedule that adds multicore and SIMD parallelism 
    # to the tiled and interleaved schedule above. 

    # redefine everything to start fresh
    x, y = Var('x'), Var('y') 
    blur_x, blur_y = Func('blur_x'), Func('blur_y') 
    blur_x[x,y] = (input[x,y]+input[x+1,y]+input[x+2,y])/3
    blur_y[x,y] = (blur_x[x,y]+blur_x[x,y+1]+blur_x[x,y+2])/3

    # schedule
    
    # Declare the inner tile variables 
    xo, yo, xi, yi = Var('xo'), Var('yo'), Var('xi'), Var('yi')

    # First schedule the last (output) stage
    # We specify computation in tiles of 256x32    
    blur_y.tile(x, y, xo, yo, xi, yi, 256, 32)   
    # We then parallelize the for loop corresponding to the yo tile index 
    # Halide will generate multithreaded code and runtime able to take advantage 
    # of multicore processors. 
    blur_y.parallel(yo)
    # We then specify that we want to use the SIMD vector units 
    # (Single Instruction Multiple Data) and compute 8 pixels at once
    # Only try to vectorize the innermost loops.  
    # There is no guarantee that the compiler will successfully achieve vectorization
    # For example, if you specify a width larger than what your processor can achieve, 
    # it won't work 
    blur_y.vectorize(xi, 8)

    # the above three scheduling instructions can be piped into a more compact version: 
    # blur_y.tile(x, y, xo, yo, xi, yi, 256, 32).parallel(yo).vectorize(xi, 8)
    # or with nicer formatting: 
    # blur_y.tile(x, y, xo, yo, xi, yi, 256, 32) \
    #       .parallel(yo) \
    #       .vectorize(xi, 8)

    # We now specify when the earlier (producer) stage blur_x gets evaluated.
    # We decide to compute it at the tile granularity of blur_y and use the
    # compute_at method. 
    # This means that blur_x will be evaluated in a loop nested inside the
    # 'xo' outer tile loop of blur_y
    # since xo is nested inside blur_y's yo and since yo is evaluated in parallel, 
    # then blur_x will also be evaluated in parallel
    # Again, we don't need to worry about bound expansion
    blur_x.compute_at(blur_y, xo) 

    # We then specify that blur_x too should be vectorized
    # Unlike the parallelism that we inherited from blur_y's yo loop, 
    # vectorization needs to be specified again because its loop nest is lower than 
    # the "compute_at" loop xo, whereas yo was above xo. 
    blur_x.vectorize(xi, 8)


    # This schedule achieves the same excellent locality  and low redundancy 
    # as the above tiling and fusion. In addition, it leverages high parallelism. 

    print 'schedule 4:'
    t = runAndMeasure(blur_y, input.width()-2, input.height()-2)
    print 'speedup: ', refTime/t

    # the equivalent python code would be similar as above but with a parallel y loop 
    # and a modified inner xi loop. In effect, vectorization adds an extra level of nesting 
    # in strides of 8 but unrolls the innermost level into single vector instructions

    print 'success!'
    return 0


#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

############# EXERCISES ###################

# Write the equivalent python code for the following Halide schedules: 
# You can assume that the image is an integer multiple of tile sizes when convenient.

# schedule 5:
# blur_y.compute_root() 
# blur_x.compute_at(blur_y, x)

# schedule 6:
# blur_y.tile(x, y, xo, yo, xi, yi, 256, 32)
# blur_x.compute_at(blur_y, yo)

# schedule 7
# blur_y.split(x, xo, xi, 8)
# blur_x.compute_at(blur_y, y)

