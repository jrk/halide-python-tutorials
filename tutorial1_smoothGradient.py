# Halide tutorial lesson 1.

# This lesson demonstrates basic usage of Halide as a JIT compiler for imaging.
# using the Python bindings and Python syntax

# In this tutorial we will create a function that has no input and that outputs
# a greyscale image representing a smooth gradient where the intensity of a pixel
# at x,y is equal to x+y

# Halide relies on metaprogramming,
# which means that it is a language within a language, in our case within Python.
# Creating a Halide program involves creating python objects that represent the Halide program,
# then calling the Halide compiler from python to turn these objects into executable binaries.
# and then actually executing the binary. 
# This explains some of the quirks of the syntax. For example we need to declare functions
# and variables before defining them or using them, in order to create the Halide representation.


import os, sys
from halide import *
# The only Halide module  you need is halide. It includes all of Halide

#Python Imaging Library will be used for IO
import Image as PIL

def main():
    # This program defines a single-stage imaging pipeline that
    # outputs a grayscale diagonal gradient.

    # A 'Func' object represents a pipeline stage. It's a pure
    # function that defines what value each pixel should have. You
    # can think of it as a computed image.

    # Unlike python, Halide requires that you declare functions and variable
    # before defining or using them
    # First, we declare the existence of a Func called gradient. We'll define it later
    gradient = Func()


    # Var objects are names to use as variables in the definition of
    # a Func. They have no meaning by themselves.
    # Again, we declare them first
    x=Var()
    y=Var()
    # can be shortened into x, y=Var('x'), Var('y')

    # Funcs are defined at any integer coordinate of their variables as
    # a Halide expression (called Expr) in terms of those variables and other functions.
    # Here, we'll define an Expr which has the value x + y. Vars have
    # appropriate operator overloading so that expressions like
    # 'x + y' implicitly become 'Expr' objects.
    e = x + y;

    # We then cast this expressioninto a float.
    # Note that like in any Python program we can reuse the same variable
    # name (here e) and assign it to a new value that might depend on the previous one. 
    e=cast(Float(32), e)
    
    # Now we'll add a definition for the Func object. At pixel x, y,
    # the image will have the value of the Expr e. On the left hand 
    # side we have the Func we're defining and some Vars representing the domain.
    # On the right hand side we have some Expr object that uses those same Vars.
    # Note the use of square brackets, just a syntax quirk 
    gradient[x, y] = e

    # The last three lines are the same as writing:
    # 
    #   gradient[x, y] = cast(Float(32), x + y)
    # 
    # which is the more common form, but we are showing the 
    # intermediate Expr here for completeness.

    # That line of code defined the Func, but it didn't actually
    # compute the output image yet. At this stage it's just Funcs,
    # Exprs, and Vars in memory, representing the structure of our
    # imaging pipeline. We're meta-programming. This Python program is
    # constructing a Halide program in memory (more or less an abstract syntax tree).
    # The actual computation of pixel data comes next.
    
    # Now we 'realize' the Func, which JIT (just in time) compiles some code that
    # implements the pipeline we've defined, and then runs it.  We
    # also need to tell Halide the domain over which to evaluate the
    # Func, which determines the range of x and y above, and the
    # resolution of the output image. The Halide module also provides a basic
    # templatized Image type we can use. We'll make a 512 x 512
    # image.
    output = gradient.realize(512, 512)    
 
    # Halide does type inference for you. Var objects represent
    # 32-bit integers, so the Expr object 'x + y' also represents a
    # 32-bit integer, and so 'gradient' defines a 32-bit image, and
    # so we got a 32-bit signed integer image out when we call
    # 'realize'. Halide types and type-casting rules are equivalent
    # to C (and similar enough to Python).


    # realize provides us with Halide's internal datatype for images
    # we now convert it to a numpy array using a double-conversion
    # (from Halide to the Python Imaging Library (PIL) and from PIL to numpy. 
    outputNP=numpy.array(Image(output))
    
    
    # Let's check everything worked, and we got the output we were
    # expecting. Let's use regular Python for this.

    for j in xrange (outputNP.shape[1]):
        for i in xrange(outputNP.shape[0]):
            # We can access a pixel of an Image object using similar
            # syntax to defining and using functions. 
            if outputNP[i, j] != i + j:
            #if True:
                print "Something went wrong! ", i, j, outputNP[i, j]
                       #"Pixel %d, %d was supposed to be %d, but instead it's %d\n",  i, j, i+j, output(i, j));
                #return -1;
    # Everything worked! We defined a Func, then called 'realize' on
    # it to generate and run machine code that produced an Image.
    print "Success!\n"
    
    # Let's display the image using the Python Imaging Library
    # the division by 4 is to normalize to 0..256
    PIL.fromarray(outputNP/4).show() 

    
#the usual Python module business
if __name__ == '__main__':
    main()


# Exercise:


# Modify the above code to create an RGB gradient similar to the reference Python code below
def rgbSmoothGradientPython():
    width=400
    height=400
    output=numpy.empty(400, 400, 3)
    for y in xrange(height):
        for x in xrange(width):
            for c in xrange(3):
                output[y, x, c]=(1-c)*cos(x)*cos(y)
    

