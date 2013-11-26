import os, sys
from halide import *

#Python Imaging Library will be used for IO
import imageIO
import time

def clampIt(input, refImage, name='clamped'):
    #TODO : different case if 2 or 3 channels
    x, y = Var('x'), Var('y')
    clamped = Func(name) 
    clamped[x, y] = input[clamp(x, 0, refImage.width()-1),
                          clamp(y, 0, refImage.height()-1)]
    return clamped

def luminance(input):
    x, y, c = Var('x'), Var('y'), Var('c')
    lumi=Func('lumi')
    lumi[x,y]=0.3*input[x, y, 0]+0.6*input[x, y, 1]+0.1*input[x,y,2]
    return lumi

def GaussianSingleChannel(input, refImage, sigma, trunc=3):
    '''take kernel as argument ?'''

    x, y = Var('x'), Var('y') #declare domain variables

    blur_x = Func('blur_x') 
    blur_y = Func('blur_y')
    blur   = Func('blur')

    #Gaussian kernel
    kernel = Func('kernel')
    kernel_width = int(sigma*trunc*2+1)   

    kernel[x]=exp(-(x- kernel_width/2.0)**2/(2.0*sigma**2))
    # force it to be root
    kernel.compute_root()
    
    # declare and define a clamping function that restricts x,y to the image size
    # boring but necessary
    clamped = clampIt(input, refImage, 'clampedInputOfGaussianBlur')

    rx = RDom(0,    kernel_width, 'rx')                            
    blur_x[x,y] = 0.0
    blur_x[x,y] += clamped[x+rx.x-kernel_width/2, y] *kernel[rx.x]

    clampedBlurx = clampIt(blur_x, refImage, 'clampedBlurx')

    ry = RDom(0,    kernel_width, 'ry')                
    blur_y[x,y] = 0.0
    blur_y[x,y] += clampedBlurx[x, y+ry.x-kernel_width/2]*kernel[ry.x]
    # note the r.x, confusing but true
    
    blur[x,y] = blur_y[x,y]/(2*3.14159*sigma**2)

    #clampedBlurx.compute_root()

    return blur, clampedBlurx

def SobelX(input, refImage):
    x, y = Var('x'), Var('y')
    sobel=Func('SobelX')
    clamped=clampIt(input, refImage)
    sobel[x,y]= (- clamped[x-1, y-1] + clamped[x+1, y-1] 
                 - 2*clamped[x-1, y] + 2*clamped[x+1, y] 
                 - clamped[x-1, y+1] + clamped[x+1, y+1] )/4.0
    return sobel

def SobelY(input, refImage):
    x, y = Var('x'), Var('y')
    sobel=Func('SobelY')
    clamped=clampIt(input, refImage)
    sobel[x,y]= (- clamped[x-1, y-1] + clamped[x-1, y+1] 
                 - 2*clamped[x, y-1] + 2*clamped[x, y+1] 
                 - clamped[x+1, y-1] + clamped[x+1, y+1] )/4.0
    return sobel

def amIAbove(input, value):
    x, y = Var('x'), Var('y')
    thresholded=Func('thresholded')
    thresholded[x,y]=select(input[x,y]>value, 1.0, 0.0)
    return thresholded

def amILocalMax(input, refImage):
    x, y = Var('x'), Var('y')
    maxi=Func('maxi')
    input=clampIt(input, refImage)
    maxi[x,y]=select( (input[x,y]>input[x-1, y] ) 
                  and (input[x,y]>input[x+1, y] )
                  and (input[x,y]>input[x, y-1] )
                  and (input[x,y]>input[x, y+1] ),
                  1.0, 0.0)
    return maxi

def computeHarris(im, indexOfSchedule, tile=256):
    
    sigma=0.5
    k = 0.04
    threshold=0.0

    input = Image(Float(32), im)

    lumi=luminance(input)
    blurredLumi, blurredLumiXX=GaussianSingleChannel(lumi, input, sigma)

    gx=SobelX(blurredLumi, input)
    gy=SobelY(blurredLumi, input)


    # Form the tensor
    x, y = Var('x'), Var('y')
    ix2=Func('Ix2')
    iy2=Func('Iy2')
    ixiy=Func('IxIy')
    ix2[x,y] = gx[x,y]**2
    iy2[x,y] = gy[x,y]**2
    ixiy[x,y]=gx[x,y]*gy[x,y]

    # Now blur tensor
    ix2Blur, ix2BlurXX=GaussianSingleChannel(ix2, input, 4.0*sigma)
    iy2Blur, iy2BlurXX=GaussianSingleChannel(iy2, input, 4.0*sigma)
    ixiyBlur, ixiyBlurXX=GaussianSingleChannel(ixiy, input, 4.0*sigma)

    # Compute the trace
    trace=Func('trace')
    trace[x,y]=ix2Blur[x,y]+iy2Blur[x,y]

    # determinant
    det=Func('det')
    det[x,y]=ix2Blur[x,y]*iy2Blur[x,y] - ixiyBlur[x,y]**2

    # Harris criterion
    R=Func('R')
    R[x,y]=det[x,y]-k*trace[x,y]**2

    # threshold
    thresh=amIAbove(R, threshold)
    #local max
    maxi=amILocalMax(R, input)
    
    harris = Func('Harris')
    harris[x,y]=maxi[x,y]*thresh[x,y]

    #harris.compile_JIT()

    ###### Schedule 
    xi, yi = Var('xi'), Var('yi')

    #harris.tile(x, y, xi, yi, 64, 64) 

    ## list of all stages that might not want to be computed inline
    # if False:
    #     lumi, consumed by blurredLumiXX
    #     blurredLumiXX consumed by gx and gy

    #     ix2 consumed by ix2BlurXX
    #     iy2 consumed by iy2BlurXX
    #     ixiy consumed by ixiyBlurXX

    #     ix2BlurXX consumed by ix2Blur
    #     iy2BlurXX consumed by iy2Blur
    #     ixiyBlurXX consumed by ixiyBlur

    #     R consumed by amILocalMax

    #     harris

    #semi smart schedule. Root everything used by a stencil
    if indexOfSchedule==0:
        #harris.compute_root()
        R.compute_root()
        
        ix2BlurXX.compute_root()
        iy2BlurXX.compute_root()
        ixiyBlurXX.compute_root()

        ix2.compute_root()
        iy2.compute_root()
        ixiy.compute_root()

        blurredLumi.compute_root()
        lumi.compute_root()
        print 'root for all stencil producers'

    if indexOfSchedule==1: 
        harris.tile(x, y, xi, yi, tile, tile)
        R.compute_at(harris, x)
        
        ix2BlurXX.compute_at(harris, x)
        iy2BlurXX.compute_at(harris, x)
        ixiyBlurXX.compute_at(harris, x)

        ix2.compute_at(harris, x)
        iy2.compute_at(harris, x)
        ixiy.compute_at(harris, x)

        blurredLumi.compute_at(harris, x)
        #lumi.compute_root()
        print 'tile everything by ', tile

    if indexOfSchedule==2: 
        harris.tile(x, y, xi, yi, tile, tile)
        R.compute_at(harris, x)
        
        ix2BlurXX.compute_at(harris, x)
        iy2BlurXX.compute_at(harris, x)
        ixiyBlurXX.compute_at(harris, x)

        ix2.compute_at(harris, x)
        iy2.compute_at(harris, x)
        ixiy.compute_at(harris, x)

        blurredLumi.compute_at(harris, x)
        #lumi.compute_root()
        print 'tile everything by ', tile

    harris.compile_jit()

    t=time.time()

    #output = harris.realize(input.width(), input.height());
    output = harris.realize(input.width(), input.height());

    dt=time.time()-t
    print indexOfSchedule, 'took ', dt, 'seconds'




def main():
    #im=imageIO.imread('hk.png', 1.0)
    im=numpy.load('Input/hk.npy')
    output=None
    print 

    for i in xrange(3):
        if i<1:
            output=computeHarris(im, i)
        else:
            for tile in [64, 128, 256, 512]:
                output=computeHarris(im, i, tile)


    if False:
        outputNP=numpy.array(Image(output))
        norm=numpy.max(outputNP)
        imageIO.imwrite(outputNP/norm)


#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

# exercises


