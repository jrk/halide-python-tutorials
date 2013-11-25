# Halide tutorial lesson 8.

# This lesson illustrates convolution using reduction


import os, sys
from halide import *

#Python Imaging Library will be used for IO
import imageIO

def main():
    
    im=imageIO.imread('rgb.png')[:, :, 1]  
    input = Image(Float(32), im)
    x, y, c = Var('x'), Var('y'), Var('c') #declare domain variables

    blur_x = Func('blur_x') 
    blur_y = Func('blur_y')
    blur   =Func('blur')

    kernel = Func('kernel')
    
    sigma = 5.5
    kernel_width = int(sigma*6+1)   

    kernel[x]=exp(-(x- kernel_width/2.0)**2/(2.0*sigma**2))
    
    # declare and define a clamping function that restricts x,y to the image size
    # boring but necessary
    clamped = Func('clamped') 
    clamped[x, y, c] = input[clamp(x, 0, input.width()-1),
                          clamp(y, 0, input.height()-1), c]

    rx = RDom(0,    kernel_width, 'rx')                
    val=clamped[x+rx.x-kernel_width/2, y, c] *kernel[rx.x]
            
    blur_x[x,y,c] = 0.0
    blur_x[x,y,c] += val

    clampedBlurx = Func('clampedBlurx') 
    clampedBlurx[x, y, c] = blur_x[clamp(x, 0, input.width()-1),
                              clamp(y, 0, input.height()-1), c]

    ry = RDom(0,    kernel_width, 'ry')                
    val=clampedBlurx[x, y+ry.x-kernel_width/2, c]*kernel[ry.x] #ry.x is confusing but oh well
            
    blur_y[x,y,c] = 0.0
    blur_y[x,y,c] += val
    
    
    blur[x,y,c] = blur_y[x,y,c]/(2*3.14159*sigma**2)

    #schedule attempt
    xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')
    blur.tile(x, y, xo, yo, xi, yi, 128, 128).parallel(yo).vectorize(xi, 8)
    blur_y.compute_at(blur, xo).vectorize(xi, 8)
    blur_x.compute_at(blur, xo).vectorize(xi, 8)
    kernel.compute_root()
    
    output = blur.realize(input.width(), input.height(), input.channels());

    outputNP=numpy.array(Image(output))





#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

# exercises


