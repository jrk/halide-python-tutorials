import imageIO
reload(imageIO)

def main():
    im=imageIO.imread('rgb.png')
    lumi=im[:,:,1] #I'm lazy, I'll just use green
    smallLumi=lumi[0:5, 0:5]

    # Replace if False: by if True: once you have implement the required functions. 
    # Exercises:
    if False:
        outputNP, myFunc=smoothGradientNormali
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'normalizedGradient.png')
    if False:
        outputNP, myFunc=smoothGradientNormali
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'rgbWave.png')
    if False: 
        outputNP, myFunc=a11.sobel(lumi)
        imageIO.imwrite(outputNP, 'sobelMag.png')
        print ' Dimensionality of Halide Func:', myFunc.dimensions()

    if False: 
        L=pythonCodeForBoxSchedule5(smallLumi)
        print L
    if False: 
        L=pythonCodeForBoxSchedule6(smallLumi)
        print L
    if False: 
        L=pythonCodeForBoxSchedule7(smallLumi)
        print L
