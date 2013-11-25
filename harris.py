import os, sys
from halide import *

#Python Imaging Library will be used for IO
import imageIO

def clampIt(input, name='clamped'):
    #TODO : different case if 2 or 3 channels
    x, y = Var('x'), Var('y')
    clamped = Func() 
    clamped[x, y] = input[clamp(x, 0, input.width()-1),
                          clamp(y, 0, input.height()-1)]
    return clamped

def luminance(input):
    x, y, c = Var('x'), Var('y'), Var('c')
    lumi=Func('lumi')
    lumi[x,y]=0.3*input[x, y, 0]+0.6*input[x, y, 1]+0.1*input[x,y,2]
    return lumi

def GaussianSingleChannel(input, sigma, trunc=3):
    '''take kernel as argument ?'''

    x, y = Var('x'), Var('y') #declare domain variables

    blur_x = Func('blur_x') 
    blur_y = Func('blur_y')
    blur   =Func('blur')

    #Gaussian kernel
    kernel = Func('kernel')
    kernel_width = int(sigma*trunc*2+1)   
    kernel[x]=exp(-(x- kernel_width/2.0)**2/(2.0*sigma**2))
    # force it to be root
    kernel.compute_root()
    
    # declare and define a clamping function that restricts x,y to the image size
    # boring but necessary
    clamped = clampIt(input)

    rx = RDom(0,    kernel_width, 'rx')                            
    blur_x[x,y] = 0.0
    blur_x[x,y] += clamped[x+rx.x-kernel_width/2, y, c] *kernel[rx.x]

    clampedBlurx = clampIt(blur_y)

    ry = RDom(0,    kernel_width, 'ry')                
    blur_y[x,y] = 0.0
    blur_y[x,y] += clampedBlurx[x, y+ry.x-kernel_width/2, c]*kernel[ry.x]
    # note the r.x, confusing but true
    
    blur[x,y] = blur_y[x,y]/(2*3.14159*sigma**2)

    return blur

def SobelX(input):
    x, y = Var('x'), Var('y')
    sobel=Func('SobelX')
    clamped=clampIt(input)
    sobel[x,y]= (- clamped(y-1, x-1) + clamped(y-1, x+1) 
                 - 2*clamped(y, x-1) + 2*clamped(y, x+1) 
                 - clamped(y+1, x-1) + clamped(y+1, x+1) )/4.0
    return sobel

def SobelY(input):
    x, y = Var('x'), Var('y')
    sobel=Func('SobelY')
    clamped=clampIt(input)
    sobel[x,y]= (- clamped(y-1, x-1) + clamped(y+1, x-1) 
                 - 2*clamped(y-1, x) + 2*clamped(y+1, x) 
                 - clamped(y-1, x+1) + clamped(y+1, x+1) )/4.0
    return sobel

def amIAbove(input, value):
    x, y = Var('x'), Var('y')
    thresholded=Func('thresholded')
    thresholded[x,y]=select(input[x,y]>value, 1.0, 0.0)
    return thresholded

def amILocalMax(input):
    x, y = Var('x'), Var('y')
    maxi=Func('maxi')
    input=clampIt(input)
    maxi[x,y]=select( (input[x,y]>input[x-1, y] ) 
                  and (input[x,y]>input[x+1, y] )
                  and (input[x,y]>input[x, y-1] )
                  and (input[x,y]>input[x, y+1] ),
                  1.0, 0.0)
    return maxi

def main():
    
    im=imageIO.imread('rgb-small.png', 1.0)
    sigma=0.5
    k = 0.04
    threshold=1.0

    input = Image(Float(32), im)

    lumi=luminance(input)
    blurredLumi=GaussianSingleChannel(lumi, sigma)

    gx=sobelX(blurredLumi)
    gy=SobelY(blurredLumi)

    # Form the tensor
    x, y = Var('x'), Var('y')
    ix2=Func('Ix2')
    iy2=Func('Iy2')
    ixiy=Func('IxIy')
    ix2[x,y] = gx[x,y]**2
    iy2[x,y] = gy[x,y]**2
    ixiy[x,y]=gx[x,y]*gy[x,y]

    # Now blur tensor
    ix2=GaussianSingleChannel(ix2, 4.0*sigma)
    iy2=GaussianSingleChannel(iy2, 4.0*sigma)
    ixiy=GaussianSingleChannel(ixiy, 4.0*sigma)

    # Compute the trace
    trace=Func('trace')
    trace[x,y]=ix2[x,y]+iy2[x,y]

    # determinant
    det=Func('det')
    det[x,y]=ix2[x,y]*iy2[x,y] - ixiy[x,y]**2

    # Harris criterion
    R=Func('R')
    R[x,y]=det[x,y]-k*trace[x,y]**2

    # threshold
    thresh=amIAbove(R, threshold)
    #local max
    maxi=amILocalMax(R)
    
    harris = Func('Harris')
    harris[x,y]=maxi[x,y]*thresh[x,y]

    output = harris.realize(input.width(), input.height(), input.channels());

    outputNP=numpy.array(Image(output))





#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

# exercises


