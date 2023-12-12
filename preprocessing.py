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
        sigma_x = 3
        blurredImage = cv2.GaussianBlur(image, kernel_size, sigma_x)
        blurredImages.append(blurredImage)

    return blurredImages

def cannyEdgeDetect(images):
    edgedImages = []

    for image in images:
        # Konversi citra ke skala abu-abu

        # Deteksi tepi menggunakan metode Canny
        edges = cv2.Canny(image, 50, 150)

        # Tambahkan citra dengan tepi ke dalam daftar
        edgedImages.append(edges)

    return edgedImages

def adaptiveThreshold(images):
    thresholdedImages = []

    for image in images:

        # Apply adaptive thresholding
        adaptive_threshold = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2   # Constant C
        )

        thresholdedImages.append(adaptive_threshold)

    return thresholdedImages

def dilation(images):
    sharpenedImages = []
    kernel = np.ones((5,5))
    for image in images:


      dilate = cv2.dilate(image, kernel, 2)

      sharpenedImages.append(dilate)
    return sharpenedImages

def erode(images):
    sharpenedImages = []
    kernel = np.ones((5,5))
    for image in images:


      dilate = cv2.erode(image, kernel, 2)

      sharpenedImages.append(dilate)
    return sharpenedImages

def contour(pureImages, processedImages):
    contourAjah = []
    number = 0
    for image in processedImages:
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            # Buat gambar kosong untuk menampilkan kotak pembatas
            kosongan = np.zeros_like(pureImages[number])

            # Temukan kontur terbesar
            largestContour = max(contours, key=cv2.contourArea)

            # Dapatkan kotak pembatas dari kontur terbesar
            x, y, w, h = cv2.boundingRect(largestContour)

            # Gambar kotak pembatas pada citra asli dan citra kosongan
            cv2.rectangle(pureImages[number], (x, y), (x + w, y + h), (255, 255, 255), 3)
            cv2.rectangle(kosongan, (x, y), (x + w, y + h), (255, 255, 255), 3)

            # Tambahkan citra kosongan dengan kotak pembatas ke dalam daftar
            contourAjah.append(kosongan)
        number += 1
    return contourAjah


def handler(imageFolders):
    #import image
    readyPaths= pathProcessing(imageFolders)
    pureImages= readImage(readyPaths)
    
    processingImages= pureImages
    #blur image
    bluredImages= gaussianBlur(processingImages)
    bluredImages= gaussianBlur(bluredImages)
    bluredImages= gaussianBlur(bluredImages)
    bluredImages= gaussianBlur(bluredImages)
    bluredImages= gaussianBlur(bluredImages)
    bluredImages= gaussianBlur(bluredImages)
    
    #adaptiveThresholded
    thresholdedImages=adaptiveThreshold(bluredImages)
    #canny edge detect
    edgedImages=cannyEdgeDetect(thresholdedImages)

    #dilate 
    dilatedImages=dilation(edgedImages)
    #erode
    erodedImages=erode(dilatedImages)
    #getCOntour
    kuntur=contour(pureImages,erodedImages)

    dilatedKuntur=dilation(kuntur)

    desired_size = (300, 200)

    # Lakukan resizing menggunakan cv2.resize
    resized_image = cv2.resize(dilatedKuntur[11], desired_size)
    
    cv2.imshow('Image', resized_image)

    # Tunggu hingga pengguna menekan tombol apa pun (0 menandakan bahwa kita menunggu waktu tak terbatas)
    cv2.waitKey(0)

    # Tutup jendela setelah tombol ditekan
    cv2.destroyAllWindows()

handler("Dataset")
