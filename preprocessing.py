import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def pathProcessing (path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']  # Add more extensions if needed
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(path, ext)))

    return image_paths

def readImage (paths):
    images=[]

    for path in paths:
        image= cv2.imread(path,0)
        if image is not None:
            images.append(image)
        else:
            print(f"Failed to read image at path: {path}")

    return images

def gaussianBlur(images):
    blurredImages=[]
    for image in images:
        kernel_size = (5, 5)
        sigma_x = 1
        blurredImage = cv2.GaussianBlur(image, kernel_size, sigma_x)
        blurredImages.append(blurredImage)

    return blurredImages

def handler(imageFolders):
    #import image
    readyPaths= pathProcessing(imageFolders)
    pureImages= readImage(readyPaths)

    processingImages= pureImages
    #blur image
    bluredImages= gaussianBlur(processingImages)
    
    #trackbars
    #canny edge detect
    #dilate 
    #erode
    #getCOntour