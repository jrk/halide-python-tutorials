# Halide tutorial lesson 5

# This lesson demonstrates how to manipulate the order in which you
# evaluate pixels in a Func, including vectorization,
# parallelization, unrolling, and tiling.

# A Halide algorithm only specifies the formula to compute  pixel values
# but it does not specify the order of execution and where/when things get stored
# This is the job of the Halide schedule.
# Once we have defined the algorithm of a Func (the formula as given by an Expr),
# we can specify its schedule by calling methods of our Python class Func. 
# For example, we specify that a Func f should gets computed in tiles by calling
# f.tile() with the proper arguments. 
# Under the hood scheduling results in a series of nested loops that fully specify
# the execution order.
# By default, Funcs get scheduled in regular order for y: for x:

import os, sys
from halide import *
# The only Halide module  you need is halide. It includes all of Halide

#Python Imaging Library will be used for IO
import Image as PIL

def main():

    # We're going to define and schedule our smooth gradient function in
    # several different ways, and see what order pixels are computed
    # in.

    # Let's declare and define our smooth gradient Func
    x, y =Var ("x"), Var("y")
    gradient=Func("gradient")
    gradient[x, y] = x + y

### First we observe the default ordering.

    # By default we walk along the rows and then down the columns.
    print "Evaluating gradient row-major\n"
    output  = gradient.realize(4, 4)

    # The equivalent Python is:
    print "Equivalent Python:\n"
    for y in xrange( 4):
        for x in xrange(4):
            print "Evaluating at ", x, y, " ->", x + y
    print "\n\n"

### Reorder variables.

    # We need to declare and define a new version of the gradient.
    x, y = Var(), Var()
    gradient=Func ("gradient_col_major")
    gradient[x, y] = x + y

    # If we reorder x and y, we can walk down the columns
    # instead. The reorder call takes the arguments of the func,
    # and sets a new nesting order for the for loops that are
    # generated. The arguments are specified from the innermost
    # loop out, so the following call puts y in the inner loop:
    
    gradient.reorder(y, x)

    print "Evaluating gradient column-major\n"
    output = gradient.realize(4, 4)

    print "Equivalent Python:\n"
    for x in xrange(4):
        for y in xrange(4):
            print "Evaluating at ", x, y, "->", x + y
    print "\n\n"

### Split a variable into two.

    x, y = Var(), Var()
    gradient = Func("gradient_split")
    gradient[x, y] = x + y

    # The most powerful primitive scheduling operation you can do
    # to a var is to split it into inner and outer sub-variables:
    # First we need to declare the outer and inner Var
    x_outer, x_inner=Var(), Var()
    # then do the split and provide the size of the inner sub-range
    gradient.split(x, x_outer, x_inner, 2)

    # This breaks the loop over x into two nested loops: an outer
    # one over x_outer, and an inner one over x_inner. The last
    # argument to split was the "split factor". The inner loop
    # runs from zero to the split factor. The outer loop runs
    # from zero to the extent required of x (4 in this case)
    # divided by the split factor, which gives us 2. Within the loops, the old
    # variable is defined to be outer * factor + inner. If the
    # old loop started at a value other than zero, then that is
    # also added within the loops.

    print "Evaluating gradient with x split into x_outer and x_inner \n"
    output = gradient.realize(4, 4)

    print "Equivalent Python:\n"
    for y in xrange(4):
        for x_outer in xrange(2):
            for x_inner in xrange(2):
                x = x_outer * 2 + x_inner
                print "Evaluating at ", x, y, "->", x + y
    print "\n\n"

    # Note that the order of evaluation of pixels didn't actually
    # change! Splitting by itself does nothing, but it does open
    # up all of the scheduling possibilities that we will explore
    # below.

## Fuse two variables into one.
    x, y = Var(), Var()
    gradient= Func ("gradient_fused")
    gradient[x, y] = x + y

    # The opposite of splitting is 'fusing'. Fusing two variables
    # merges the two loops into a single for loop over the
    # product of the extents. Fusing is less important that
    # splitting, but it also sees use (as we'll see later in this
    # lesson). Like splitting, fusing by itself doesn't change
    # the order of evaluation.
    fused = Var()
    gradient.fuse(x, y, fused)

    print "Evaluating gradient with x and y fused\n"
    output = gradient.realize(4, 4)

    print "Equivalent Python:\n"
    for fused in xrange(4*4):
        y = fused / 4
        x = fused % 4
        print "Evaluating at ", x, y, "->", x + y

## Evaluating in tiles.

    x, y = Var(), Var()
    gradient=Func("gradient_tiled")
    gradient[x, y] = x + y

    # Now that we can both split and reorder, we can do tiled
    # evaluation. Let's split both x and y by a factor of two,
    # and then reorder the vars to express a tiled traversal.
    #
    # A tiled traversal splits the domain into small rectangular
    # tiles, and outermost iterates over the tiles, and within
    # that iterates over the points within each tile. It can be
    # good for performance if neighboring pixels use overlapping
    # input data, for example in a blur. We can express a tiled
    # traversal like so:

    x_outer, x_inner, y_outer, y_inner = Var(), Var(), Var(), Var()
    gradient.split(x, x_outer, x_inner, 2)
    gradient.split(y, y_outer, y_inner, 2)
    gradient.reorder(x_inner, y_inner, x_outer, y_outer)

    # This pattern is common and important enough that there's a shorthand for it:
    # gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, 2, 2)

    print "Evaluating gradient in 2x2 tiles\n"
    output = gradient.realize(4, 4)

    print "Equivalent Python:\n"
    for y_outer in xrange(2):
        for x_outer in xrange(2):
            for y_inner in xrange(2):
                for x_inner in xrange(2):
                    x = x_outer * 2 + x_inner
                    y = y_outer * 2 + y_inner
                    print "Evaluating at ", x, y, "->", x + y
    print "\n\n"

    # We will work more with tiles in the next tutorial. 
    # They are critical for high-performance image processing

# Evaluating in vectors.

    x, y = Var(), Var()
    gradient=Func("gradient_in_vectors")
    gradient[x, y] = x + y

    # The nice thing about splitting is that it guarantees the
    # inner variable runs from zero to the split factor. Most of
    # the time the split-factor will be a compile-time constant,
    # so we can replace the loop over the inner variable with a
    # single vectorized computation. This time we'll split by a
    # factor of four, because on X86 we can use SSE to compute in
    # 4-wide vectors.
    x_outer, x_inner=Var(), Var()
    gradient.split(x, x_outer, x_inner, 4)
    gradient.vectorize(x_inner)

    # Splitting and then vectorizing the inner variable is common
    # enough that there's a short-hand for it. We could have also
    # said:
    #
    # gradient.vectorize(x, 4)
    #
    # which is equivalent to:
    #
    # gradient.split(x, x, x_inner, 4)
    # gradient.vectorize(x_inner)
    #
    # Note that in this case we reused the name 'x' as the new
    # outer variable. Later scheduling calls that refer to x
    # will refer to this new outer variable named x.
    #
    # Our snoop function isn't set-up to print out vectors, this
    # is why we included one called snoopx4 above.

    # This time we'll evaluate over an 8x4 box, so that we have
    # more than one vector of work per scanline.
    print "Evaluating gradient with x_inner vectorized \n"
    output = gradient.realize(8, 4)

    print "Equivalent Python:\n"
    for  y in xrange(4):
        for x_outer in xrange(2):
            # The loop over x_inner has gone away, and has been
            # replaced by a vectorized version of the
            # expression. On x86 processors, Halide generates SSE
            # for all of this.
            x_vec = numpy.array([x_outer * 4 + 0,
                                 x_outer * 4 + 1,
                                 x_outer * 4 + 2,
                                 x_outer * 4 + 3])
            val = numpy.array([x_vec[0] + y,
                               x_vec[1] + y,
                               x_vec[2] + y,
                               x_vec[3] + y])
            print "Evaluating at <", \
               x_vec[0], x_vec[1], x_vec[2], x_vec[3], ">, <",\
               y, y, y, y,"> -> <", \
               val[0], val[1], val[2], val[3], ">"
    print "\n\n"

# Unrolling a loop.

    x, y = Var(), Var()
    gradient=Func("gradient_in_vectors")
    gradient[x, y] = x + y

    # If multiple pixels share overlapping data, it can make
    # sense to unroll a computation so that shared values are
    # only computed or loaded once. We do this similarly to how
    # we expressed vectorizing. We split a dimension and then
    # fully unroll the loop of the inner variable. Unrolling
    # doesn't change the order in which things are evaluated.
    x_outer, x_inner = Var(), Var()
    gradient.split(x, x_outer, x_inner, 2)
    gradient.unroll(x_inner)

    # The shorthand for this is:
    # gradient.unroll(x, 2)

    print "Evaluating gradient unrolled by a factor of two\n"
    result = gradient.realize(4, 4)

    print "Equivalent Python:\n"
    for y in xrange(4):
        for x_outer  in xrange(2):
            # Instead of a for loop over x_inner, we get two
            # copies of the innermost statement.
            x_inner = 0
            x = x_outer * 2 + x_inner
            print "Evaluating at ", x, y, "->", x + y

            x_inner = 1
            x = x_outer * 2 + x_inner
            print "Evaluating at ", x, y,  "->", x + y

# Splitting by factors that don't divide the extent.

    x, y = Var(), Var()
    gradient=Func("gradient_split_5x4")
    gradient[x, y] = x + y

    # Splitting guarantees that the inner loop runs from zero to
    # the split factor, which is important for the uses we saw
    # above. So what happens when the total extent we wish to
    # evaluate x over isn't a multiple of the split factor? We'll
    # split by a factor of two again, but now we'll evaluate
    # gradient over a 5x4 box instead of the 4x4 box we've been
    # using.
    x_outer, x_inner= Var(), Var()
    gradient.split(x, x_outer, x_inner, 2)

    print "Evaluating gradient over a 5x4 box with x split by two \n"
    output = gradient.realize(5, 4)

    print "Equivalent Python:"
    for y in xrange(4):
        for x_outer  in xrange(3):  # Now runs from 0 to 3
            for x_inner in xrange(2):
                x = x_outer * 2
                # Before we add x_inner, make sure we don't
                # evaluate points outside of the 5x4 box. We'll
                # clamp x to be at most 3 (5 minus the split
                # factor).
                # it gets compiled to a conditional move, which is cheap
                if x > 3: x = 3
                x += x_inner
                print "Evaluating at ", x, y, "->", x + y
    print "\n\n"

    # If you read the output, you'll see that some coordinates
    # were evaluated more than once! That's generally OK, because
    # pure Halide functions have no side-effects, so it's safe to
    # evaluate the same point multiple times. If you're calling
    # out to C functions like we are, it's your responsibility to
    # make sure you can handle the same point being evaluated
    # multiple times.

    # The general rule is: If we require x from x_min to x_min + x_extent, and
    # we split by a factor 'factor', then:
    #
    # x_outer runs from 0 to (x_extent + factor - 1)/factor
    # x_inner runs from 0 to factor
    # x = min(x_outer * factor, x_extent - factor) + x_inner + x_min
    #
    # In our example, x_min was 0, x_extent was 5, and factor was 2.


# Fusing, tiling, and parallelizing.

    # We saw in the previous lesson that we can parallelize
    # across a variable. Here we combine it with fusing and
    # tiling to express a useful pattern - processing tiles in
    # parallel.

    # This is where fusing shines. Fusing helps when you want to
    # parallelize across multiple dimensions without introducing
    # nested parallelism. Nested parallelism (parallel for loops
    # within parallel for loops) is supported by Halide, but
    # often gives poor performance compared to fusing the
    # parallel variables into a single parallel for loop.

    x, y = Var(), Var()
    gradient=Func("gradient_fused_tiles")
    gradient[x, y] = x + y

    # First we'll tile, then we'll fuse the tile indices and
    # parallelize across the combination.
    x_outer, y_outer, x_inner, y_inner, tile_index=Var(), Var(), Var(), Var(), Var()
    gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, 2, 2)
    gradient.fuse(x_outer, y_outer, tile_index)
    gradient.parallel(tile_index)

    # The scheduling calls all return a reference to the Func, so
    # you can also chain them together into a single statement to
    # make things slightly clearer:
    #
    # gradient
    #     .tile(x, y, x_outer, y_outer, x_inner, y_inner, 2, 2)
    #     .fuse(x_outer, y_outer, tile_index)
    #     .parallel(tile_index)


    print "Evaluating gradient tiles in parallel\n"
    output = gradient.realize(4, 4)

    # The tiles should occur in arbitrary order, but within each
    # tile the pixels will be traversed in row-major order.

    print "Equivalent (serial) Python:\n"
    # This outermost loop should be a parallel for loop, but that's hard in C.
    for tile_index in xrange(4):
        y_outer = tile_index / 2
        x_outer = tile_index % 2
        for y_inner in xrange(2):
            for x_inner in xrange(2):
                y = y_outer * 2 + y_inner
                x = x_outer * 2 + x_inner
                print "Evaluating at ", x, y, "->", x + y
    print "\n\n"

# Putting it all together.

    # Are you ready? We're going to use all of the features above now.
    x, y = Var(), Var()
    gradient_fast=Func("gradient_fast")
    gradient_fast[x, y] = x + y

    # We'll process 256x256 tiles in parallel.
    x_outer, y_outer, x_inner, y_inner, tile_index= Var(), Var(), Var(), Var(), Var()
    gradient_fast \
        .tile(x, y, x_outer, y_outer, x_inner, y_inner, 256, 256) \
        .fuse(x_outer, y_outer, tile_index) \
        .parallel(tile_index)

    # We'll compute two scanlines at once while we walk across
    # each tile. We'll also vectorize in x. The easiest way to
    # express this is to recursively tile again within each tile
    # into 4x2 subtiles, then vectorize the subtiles across x and
    # unroll them across y:
    x_inner_outer, y_inner_outer, x_vectors, y_pairs=Var(), Var(), Var(), Var()
    gradient_fast \
        .tile(x_inner, y_inner, x_inner_outer, y_inner_outer, x_vectors, y_pairs, 4, 2) \
        .vectorize(x_vectors) \
        .unroll(y_pairs)

    # Note that we didn't do any explicit splitting or
    # reordering. Those are the most important primitive
    # operations, but mostly they are buried underneath tiling,
    # vectorizing, or unrolling calls.

    # Now let's evaluate this over a range which is not a
    # multiple of the tile size.

    # If you like you can turn on tracing, but it's going to
    # produce a lot of printfs. Instead we'll compute the answer
    # both in Python and Halide.
    result = gradient_fast.realize(800, 600)

    print "Equivalent Python...(no printout) \n"
    for tile_index in xrange( 4 * 3 ):
        y_outer = tile_index / 4
        x_outer = tile_index % 4
        for y_inner_outer in xrange(256/2):
            for x_inner_outer in xrange(256/4):
                # We're vectorized across x
                x = min(x_outer * 256, 800-256) + x_inner_outer*4
                x_vec = numpy.array([x + 0,
                                     x + 1,
                                     x + 2,
                                     x + 3])

                # And we unrolled across y (next 2 paragraphs)
                y_base = min(y_outer * 256, 600-256) + y_inner_outer*2

                # y_pairs = 0
                y = y_base + 0
                y_vec = numpy.array([y, y, y, y])
                val = numpy.array([x_vec[0] + y_vec[0],
                                   x_vec[1] + y_vec[1],
                                   x_vec[2] + y_vec[2],
                                   x_vec[3] + y_vec[3]])
                # y_pairs = 1
                y = y_base + 1
                y_vec = numpy.array([y, y, y, y])
                val = numpy.array([x_vec[0] + y_vec[0],
                                   x_vec[1] + y_vec[1],
                                   x_vec[2] + y_vec[2],
                                   x_vec[3] + y_vec[3]])


    # Note that in the Halide version, the algorithm is specified
    # once at the top, separately from the optimizations, and there
    # aren't that many lines of code total. Compare this to the C
    # version. There's more code (and it isn't even parallelized or
    # vectorized properly). More annoyingly, the statement of the
    # algorithm (the result is x plus y) is buried in multiple places
    # within the mess. This C code is hard to write, hard to read,
    # hard to debug, and hard to optimize further. This is why Halide
    # exists.

    print "Success!\n"
    return 0

#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()


