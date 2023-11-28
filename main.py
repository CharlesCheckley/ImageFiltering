import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import dist


'''
2D Convolution function which then applies an image 
filtering technique. 

Inputs: 
image: a greyscale image. 
kernelSize: the size of the window. Must be square and odd.
filter: the filter to apply to the window.

Output: 
A filtered image

'''
def convolution(image, kernelSize, filter):
    #sets iH image height and iW image width to their values 
    # obtained from the shape command
    imageHeight, imageWidth = image.shape[:2]
    
    #so that convolution can act on all parts of the image padding is added
    #in the from of 0s. 
    #determine the width of the padding required 
    pad = (kernelSize - 1) // 2

    #pad the image on all sides with padding of width 'pad'
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value = 0)

    #preallocate the output image array 
    output = np.zeros((imageHeight, imageWidth), dtype="float32")

    #loop through every pixel in the original (not the padded) image
    for y in range(pad, imageHeight + pad ): 
        for x in range(pad, imageWidth + pad):
            window = image[y - pad : y + pad + 1, x - pad : x + pad + 1]
            #applies a filter function to the window
            k = filter(window)
            output[y - pad, x - pad] = k

    output = output.astype("uint8")
	# return the output image
    return output


'''
Filter functions: each filter acts on a square window with odd side length
'''

#mean filter
def mean(window): 
    return ((window * 1/(window.shape[0]*window.shape[1])).sum())

#sharpening filter
def sharpen(window): 
    kernel = np.array((
	[-1, -1, -1],
	[-1, 6, -1],
	[-1, -1, -1]))

    return (window * kernel).sum()


#creates a 2D Gaussian Kernel
def makeGKernel(length, sigma):
    
    #generate a 1D Gaussian Distribution
    x = np.linspace(-length // 2, length // 2, length)
    gauss = np.exp(-0.5 * np.square(x) / np.square(sigma))

    #Use the product of 2x the 1D Gaussian Distribution 
    #to make the 2D Gaussian 
    kernel = np.outer(gauss, gauss)
    return kernel 



#Gaussian Low Pass filter
#requires a Guassian Kernel created by the function makeGKernel
def GaussianLowPass(window):
    kernel = makeGKernel(window.shape[1], 1)    
    output =  (window * kernel).sum()
    return output


#median filter
def median(window):
    #numpy library for efficient sorting
    sorted = np.sort(window, axis = None)
    medianValue = sorted.size // 2
    return sorted[medianValue]


#adaptive weighted median filter
def AdaptedWeightedMedian(window, centralWeight = 100, constant = 10 ):
    #preallocate numpy array for weight values
    weights = np.zeros((window.shape[0],window.shape[1]))

    #calculate the position of the central value in the window
    middle = [(window.shape[0] - 1)/2, (window.shape[1] - 1)/2] 

    #get mean and standard deviation values of the window
    #fetching them here avoids calculating them every time in the for loop
    meanValue = mean(window)
    std = np.std(window)

    #checks for a mean window value of 0 
    #speeds up calculation and avoids divide by 0 errors. 
    if meanValue == 0: 
        medianValue = 0
        return medianValue

    #calculate distance of each element from the centre of the window
    for y in range(0, window.shape[0]):
        for x in range(0, window.shape[1]):
            distance = dist(middle, [y,x])
            #calculate weight according to given formula
            weights[y,x] = int(round(centralWeight - ((constant * distance * std) / meanValue)))
 

    weightMedianValue = weights.sum() // 2
    window = window.flatten()
    weights = weights.flatten()
    stacked = np.stack((window, weights))
    stacked = stacked[:, stacked[0, :].argsort()]
    
    for i in range(0, stacked.shape[1]): 
        if weightMedianValue > 0:
            value = stacked[1,i]
            
            weightMedianValue = weightMedianValue - value
            
        else: 
            break 
    medianValue = stacked[0,i]
    return medianValue


    
#import one of the two sample images
img = cv2.imread('C:\\Users\\charl\\OneDrive - University of Bath\\Year 4\\Image Processing\\NZjers1.png' , 0)
#or
#img = cv2.imread('C:\\Users\\charl\\OneDrive - University of Bath\\Year 4\\Image Processing\\foetus.png', 0)

#example usage calls


output = convolution(img, 5, GaussianLowPass)


edges = cv2.Canny(image=output, threshold1=100, threshold2=200) # Canny Edge Detection



plt.imshow(edges, cmap = 'gray')
plt.show()

