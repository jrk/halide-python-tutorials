

import os, sys
from halide import *
# The only Halide module  you need is halide. It includes all of Halide


def smoothGradientNormalized():
    '''use Halide to compute a 512x512 smooth gradient equal to x+y divided by 1024
    Do not worry about the schedule. 
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''


def wavyRGB():
    '''Use a Halide Func to compute a wavy RGB image like that obtained by the following 
    Python formula below. output[y, x, c]=(1-c)*cos(x)*cos(y)
    Do not worry about the schedule. 
    Hint : you need one more domain dimension than above
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''


def luminance(im):
    '''input is assumed to be our usual numpy image representation with 3 channels. 
    Use Halide to compute a 1-channel image representing 0.3R+0.6G+0.1B
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''

def  sobel(lumi):
    ''' lumi is assumed to be a 1-channel numpy array. 
    Use Halide to apply a SObel filter and return the gradient magnitude. 
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''


def pythonCodeForBoxSchedule5(lumi):    
        ''' lumi is assumed to be a 1-channel numpy array. 
        Write the python nested loops corresponding to the 3x3 box schedule 5
        and return a list representing the order of evaluation. 
        Each time you perform a computation of blur_x or blur_y, put a triplet with the name 
        of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y. '''
        
        # schedule 5:
        # blur_y.compute_root() 
        # blur_x.compute_at(blur_y, x)

def pythonCodeForBoxSchedule6(lumi):    
        ''' lumi is assumed to be a 1-channel numpy array. 
        Write the python nested loops corresponding to the 3x3 box schedule 5
        and return a list representing the order of evaluation. 
        Each time you perform a computation of blur_x or blur_y, put a triplet with the name 
        of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y. '''
        
        # schedule 6:
        # blur_y.tile(x, y, xo, yo, xi, yi, 2, 2)
        # blur_x.compute_at(blur_y, yo)


def pythonCodeForBoxSchedule7(lumi):    
        ''' lumi is assumed to be a 1-channel numpy array. 
        Write the python nested loops corresponding to the 3x3 box schedule 5
        and return a list representing the order of evaluation. 
        Each time you perform a computation of blur_x or blur_y, put a triplet with the name 
        of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y. '''
        
        # schedule 7
        # blur_y.split(x, xo, xi, 2)
        # blur_x.compute_at(blur_y, y)




