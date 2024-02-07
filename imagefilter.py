from PIL import Image
import numpy as np
import math
from scipy import signal
import cv2
import time

'''
boxfilter(n)
return:  box filter of size n by n

n is odd and filter should be a Numpy array
'''

def boxfilter(n):
    assert n > 0 and n%2 == 1, "Dimension must be odd" #True, nothing happens; False, print message
    return 1/(math.pow(n,2))*np.ones((n,n))

# test
# print(boxfilter(5))
# print(boxfilter(4))
    
'''
Formula for the Gaussian

gauss1d(sigma)
return: a 1D Gaussian filter for a given value of sigma
'''

def gauss1d(sigma):
    # length of filter with length 6 times sigma rounded to next odd integer
    filter_length = math.ceil(6 * sigma)
    # print(filter_length)
    if filter_length % 2 != 1:
        filter_length += 1
        # print(filter_length)
        # print("ths is running")
        
        
    # generate 1D array of x values
    x = np.arange(-1*filter_length//2 + 1, filter_length//2+1)
    # print(x)
    
    # Value of filter = Gaussian function: exp(-x^2/(2*sigma^2)
    gauss_filter = np.exp(-np.square(x) / (2 * np.square(sigma)))
    # print("debug")
    # print(gauss_filter)
    
    #Formula for the Gaussian ignores constant factor -> normalize the values in the filter so that they sum to 1
    gauss_filter = gauss_filter/(np.sum(gauss_filter))
    
    return gauss_filter

# test
# sigma_values = [0.3, 0.5, 1, 2]

# for sigma in sigma_values:
#     gaussian_filter = gauss1d(sigma)
#     print(gaussian_filter, "\n") 


'''
gauss2d(sigma) 
return: 2D Gaussian filter for a given value of sigma

filter should be a 2D array
'''

def gauss2d(sigma):
    # compute gaussian 1d
    gauss_1d = gauss1d(sigma)
    
    # compute gaussian 2d; convert 1d array with f[np.newaxis] and solve with convolve2d
    gauss_2d = signal.convolve2d(gauss_1d[np.newaxis], np.transpose(gauss_1d[np.newaxis]), mode = 'full', boundary = 'fill', fillvalue = 0)
    return gauss_2d


# print(gauss2d(0.5))
# print(gauss2d(1))


'''
convolve2d_manual(array, filter) 


takes in an image (stored in array) and a filter
performs convolution to the image with zero paddings (thus, the image sizes of input and output are the same)

Used the following source to help with my understanding
Citation: https://stackoverflow.com/questions/63036809/how-do-i-use-only-numpy-to-apply-filters-onto-images

'''

def convolve2d_manual(array, filter):
   # image dimensions
   img_rows = array.shape[0]
   img_columns = array.shape[1]
   
   # filter dimensions 
   filter_rows = filter.shape[0]
   filter_columns = filter.shape[1]
   
   # calculate the amount of padding needed and pad image
   padding_top = filter_rows // 2
   padding_bottom = filter_rows - padding_top - 1
   padding_left = filter_columns // 2
   padding_right = filter_columns - padding_left - 1

   pad_img = np.pad(array, ((padding_top, padding_bottom), (padding_left, padding_right)), mode='constant')
   
   # initialize the output with the same shape as the original image (initialized with zeros)
   output_image = np.zeros((img_rows, img_columns))
   
   # convolution 
   for row in range(img_rows):
       for column in range(img_columns):
           # determine the dimensions of the neighborhood
           neighborhood = pad_img[row : row + filter_rows, column : column + filter_columns]
           # output = summation -> filter (i, j) * array (x-i, y-i) (neighborhood)
           output_image[row, column] = np.sum(np.multiply(filter, neighborhood))
           
   return output_image

'''
gaussconvolve2d_manual(array,sigma) 
return: 2D array

applies Gaussian convolution to a 2D array for the given value of sigma
1. generating a filter with gauss2d
2. apply it to the array with convolve2d_manual(array, filter)

'''
def gaussconvolve2d_manual(array,sigma):
    gauss2d_filter = gauss2d(sigma)
    gauss_convolve2d = convolve2d_manual(array, gauss2d_filter)
    
    
    return gauss_convolve2d

'''
Apply gaussconvolve2d_manual to the image of the dog
Convert it to a greyscale numpy array and run gaussconvolve2d
'''
# open and display the dog image
dog_img = Image.open('images/0b_dog.bmp')
dog_img.save('dog0b.png','PNG')
dog_img.show()

# convert the image to a black and white "luminance" greyscale image
grey_dog_img = dog_img.convert('L')

# convert the image to a numpy array (for subsequent processing)
grey_dog_array = np.asarray(grey_dog_img, dtype=np.float32)

# apply filter
filtered_grey_dog_array = gaussconvolve2d_manual(grey_dog_array, 3)

# save, open and display the filtered dog image
filtered_grey_dog_img = Image.fromarray(filtered_grey_dog_array.astype('uint8'))
filtered_grey_dog_img.save('filtered_dog.png','PNG')
filtered_grey_dog_img.show()


'''
gaussconvolve2d_scipy(array,sigma)
return: 2D array

1. generate a filter with gauss2d
2. apply it to the array with signal.convolve2d(array,filter,'same')
'''

def gaussconvolve2d_scipy(array,sigma):
    gauss2d_filter = gauss2d(sigma)
    gauss_convolve2d_scipy = signal.convolve2d(array,gauss2d_filter,'same')
    
    return gauss_convolve2d_scipy

'''
Create a blurred version of the one of the paired images and filter each of the three color channels (RGB) separately

Used the following source to help with my understanding
citation: https://stackoverflow.com/questions/57398643/how-to-extract-individual-channels-from-an-rgb-image
          https://www.educative.io/answers/splitting-rgb-channels-in-python
    
'''
sigma = 10

# convert .bmp file to .png. Save the image
dog0b_bmp = Image.open('images/0b_dog.bmp')
dog0b_bmp.save('dog0b.png','PNG')
dog0b_bmp.show()

# convert the image to a numpy array (for subsequent processing)
dog0b_array = np.asarray(dog0b_bmp, dtype = np.float32)

# extract colour channel (rgb)
dog_red_channel = dog0b_array[:, :, 0]
dog_green_channel = dog0b_array[:, :, 1]
dog_blue_channel = dog0b_array[:, :, 2]

# filter each colour with gaussconvolve2d_scipy function
dog_blurred_red = gaussconvolve2d_scipy(dog_red_channel, sigma)
dog_blurred_green = gaussconvolve2d_scipy(dog_green_channel, sigma)
dog_blurred_blue = gaussconvolve2d_scipy(dog_blue_channel, sigma)

# Merge the colour channels back to each other
dog0b_merged_array = np.stack([dog_blurred_red, dog_blurred_green, dog_blurred_blue], axis=-1)

# save and show filtered image
dog0b_filtered  = Image.fromarray(dog0b_merged_array.astype('uint8'))
dog0b_filtered.save('dog0b_filtered.png','PNG')
dog0b_filtered.show()

'''
High frequency filtered image = Compute a low frequency Gaussian filtered image and then subtracting it from the original

citation: https://stackoverflow.com/questions/57398643/how-to-extract-individual-channels-from-an-rgb-image
          https://www.educative.io/answers/splitting-rgb-channels-in-python
'''

# convert .bmp file to .png. Save the image
cat0a_bmp = Image.open('images/0a_cat.bmp')
cat0a_bmp.save('cat0a.png','PNG')
cat0a_bmp.show()

# convert the image to a numpy array (for subsequent processing)
cat0a_array = np.asarray(cat0a_bmp, dtype = np.float32)
    
# extract colour channel (rgb)
cat_red_channel = cat0a_array[:, :, 0]
cat_green_channel = cat0a_array[:, :, 1]
cat_blue_channel = cat0a_array[:, :, 2]

# filter each colour with gaussconvolve2d_scipy function (low frequency)
cat_blurred_red = gaussconvolve2d_scipy(cat_red_channel, sigma)
cat_blurred_green = gaussconvolve2d_scipy(cat_green_channel, sigma)
cat_blurred_blue = gaussconvolve2d_scipy(cat_blue_channel, sigma)

    
# high frequency filtered image = Low frequency Gaussian filtered image - original
cat_red_highfreq = cat_red_channel - cat_blurred_red
cat_green_highfreq = cat_green_channel - cat_blurred_green
cat_blue_highfreq = cat_blue_channel - cat_blurred_blue

# Merge the colour channels back to each other
cat0a_merged_array = np.stack([cat_red_highfreq, cat_green_highfreq, cat_blue_highfreq], axis=-1)

# visualize by adding 128 (rescale to the range [0,255] -> rgb colour scale)
cat0a_vis_highfreq = cat0a_merged_array + 128


# save and show filtered image
cat0a_filtered  = Image.fromarray(cat0a_vis_highfreq.astype('uint8'))
cat0a_filtered.save('dog0b_filtered.png','PNG')
cat0a_filtered.show()

'''
Create hybrid images

citation: https://stackoverflow.com/questions/57398643/how-to-extract-individual-channels-from-an-rgb-image
          https://www.educative.io/answers/splitting-rgb-channels-in-python

'''

# filters the images with gaussconvolve2d_scipy
# returns: gaussian filtered version of the original image and original image (both in the form of an array)

# loads the image and extracts the colour channel
def image_helper(image_path):    
    # load the image
    img_bmp = Image.open(image_path)
    
    # convert the image to a numpy array (for subsequent processing)
    img_array = np.asarray(img_bmp, dtype = np.float32)

    # extract colour channel (rgb)
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2]
    
    return red_channel, green_channel, blue_channel

def gauss_filter(sigma, image_path):
    red_channel, green_channel, blue_channel = image_helper(image_path)

    # filter each colour with gaussconvolve2d_scipy function
    blurred_red = gaussconvolve2d_scipy(red_channel, sigma)
    blurred_green = gaussconvolve2d_scipy(green_channel, sigma)
    blurred_blue = gaussconvolve2d_scipy(blue_channel, sigma)
    
    return blurred_red, blurred_green, blurred_blue

# filters the images with high frequency
# returns: high frequency filtered image (in the form of an array)
def highfreq_filter(sigma, image_path):
    red_channel, green_channel, blue_channel = image_helper(image_path)
    blurred_red, blurred_green, blurred_blue = gauss_filter(sigma, image_path)
    
    # high frequency filtered image = Low frequency Gaussian filtered image - original
    red_highfreq = red_channel - blurred_red
    green_highfreq = green_channel - blurred_green
    blue_highfreq = blue_channel - blurred_blue
    
    return red_highfreq, green_highfreq, blue_highfreq

# merges two images together to create a hybrid image
# returns: hybrid image
def hybrid_img(sigma, image_path_A, image_path_B, new_file_name):
    blurred_red, blurred_green, blurred_blue = gauss_filter(sigma, image_path_A)
    
    red_highfreq, green_highfreq, blue_highfreq = highfreq_filter(sigma, image_path_B)
    
    # Merge the two images together
    merged_array = np.stack([blurred_red + red_highfreq, blurred_green + green_highfreq, blurred_blue + blue_highfreq], axis=-1)
    
    # Clam the values in range [0,255] to remove noise
    merged_array = np.clip(merged_array, 0, 255)

    # save and show filtered image
    hybrid_filtered = Image.fromarray(merged_array.astype('uint8'))
    hybrid_filtered.save(new_file_name, 'PNG')
    hybrid_filtered.show()
    

hybrid_img(3, 'images/0b_dog.bmp', 'images/0a_cat.bmp', 'hybrid_cat_dog3.png')
hybrid_img(3, 'images/1b_motorcycle.bmp', 'images/1a_bicycle.bmp', 'hybrid_bike_motocycle3.png')
hybrid_img(3, 'images/2b_marilyn.bmp', 'images/2a_einstein.bmp', 'hybrid_marilyn_einstein3.png')
hybrid_img(3, 'images/3b_submarine.bmp', 'images/3a_fish.bmp', 'hybrid_submarine_fish3.png')
hybrid_img(3, 'images/4b_plane.bmp', 'images/4a_bird.bmp', 'hybrid_plane_bird3.png')

