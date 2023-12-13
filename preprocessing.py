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
        image= cv2.imread(path)
        if image is not None:
            images.append(image)
        else:
            print(f"Failed to read image at path: {path}")

    return images

def grayImages( images):
    grayed=[]
    for image in images:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        grayed.append(gray)

    return grayed

def gaussianBlur(images):
    blurredImages=[]
    for image in images:
        kernel_size = (5, 5)
        sigma_x = 9
        blurredImage = cv2.GaussianBlur(image, kernel_size, sigma_x)
        blurredImages.append(blurredImage)

    return blurredImages

def cannyEdgeDetect(images,th1,th2):
    edgedImages = []

    for image in images:
        # Konversi citra ke skala abu-abu

        # Deteksi tepi menggunakan metode Canny
        edges = cv2.Canny(image,th1, th2)

        # Tambahkan citra dengan tepi ke dalam daftar
        edgedImages.append(edges)

    return edgedImages

def convolution2d(image,kernel,stride,padding):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Pad the image with zeros to account for the kernel size.
    padded_image = np.zeros((image_height + kernel_height - 1, image_width + kernel_width - 1))
    padded_image[kernel_height // padding:image_height + kernel_height // padding, kernel_width // padding:image_width + kernel_width // padding] = image

    # Convolve the image with the kernel.
    convolved_image = np.zeros((image_height, image_width))
    for i in range(image_height):
      for j in range(image_width):
        convolved_image[i, j] = np.sum(padded_image[i*stride:i*stride + kernel_height, j*stride:j*stride + kernel_width] * kernel)

    return convolved_image

def emboss(images):
    embossed=[]
    for image in images:
        kernel = np.array([[-2, -1, 0],
                    [-1, 1, 1],
                    [0, 1, 2]])

        convolved_image = convolution2d(image, kernel, 1, 2)

        embossed.append(convolved_image)
    return embossed

def topHat(images):
    topHatImages = []

    for image in images:
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        topHatImages.append(top_hat)

    return topHatImages

def opening(images):
    opened=[]

    for image in images:
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        opened.append(opening)

    return opened


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

def mapp(h):
    h=h.reshape((4,2))
    hnew=np.zeros((4,2),dtype=np.float32)

    add = h.sum(1)
    hnew[0]=h[np.argmin(add)]
    hnew[2]=h[np.argmax(add)]

    diff=np.diff(h,axis=1)
    hnew[1]=h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def contourIndia(originaImages,images):
    flipped=[]
    number=0
    for image in images:
        image2,contours,hierarchy=cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours,key=cv2.contourArea,reverse=True)

        for c in contours:
            p=cv2.arcLength(c,True)
            approx= cv2.approxPolyDP(c,0.02*p,True)
            if len(approx)==4:
                target=approx
                break
        approx=mapp(target)
        pts=np.float32([[0,0],[180,0],[180,180],[0,180]])
        op=cv2.getPerspectiveTransform(approx,pts)
        dst=cv2.warpPerspective(originaImages[number],op,(180,180))
        flipped.append(dst)
        number+=1
    return flipped


def display_images(images, titles):
    # Menghitung jumlah baris dan kolom
    rows = len(images) // 2 + len(images) % 2
    cols = 2
    dim =180
    # Membuat frame kosong dengan ukuran yang sesuai
    frame = np.zeros((rows * dim, cols * dim), dtype=np.uint8)

    for i, (img, title) in enumerate(zip(images, titles)):
        # Resizing gambar menjadi (500, 500)
        resized_img = cv2.resize(img, (dim, dim))

        # Menentukan baris dan kolom untuk menempatkan gambar
        row = i // cols
        col = i % cols

        # Menempatkan gambar pada frame
        frame[row * dim:(row + 1) * dim, col * dim:(col + 1) * dim] = resized_img

        # Menambahkan judul di bawah gambar
        cv2.putText(frame, title, (col * dim, (row + 1) * dim - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)

    # Menampilkan frame yang berisi semua gambar
    cv2.imshow('Image Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def handler(imageFolders):
    #import image
    readyPaths= pathProcessing(imageFolders)
    pureImages= readImage(readyPaths)
    
    processingImages= pureImages
    #blur image
    bluredImages= emboss(processingImages)

    #adaptiveThresholded
    segmented= adaptiveThreshold(bluredImages)
    #erode
    opened= opening(segmented)

    #getCOntour
    kuntur=contour(pureImages,opened)

    dilatedKuntur=dilation(kuntur)
    number=5

    # Lakukan resizing menggunakan cv2.resize
    images_to_display = [pureImages[number], bluredImages[number], segmented[number], opened[number], dilatedKuntur[number]]
    titles = ['Pure Image', 'Blurred Image', 'Segmented Image', 'Opened Image', 'Dilated Contour']

    display_images(images_to_display,titles)
    

def handler2(path):

    processed= pathProcessing(path)

    pureImages=readImage(processed)

    grayedImages= grayImages(pureImages)

    blurredImages=gaussianBlur(grayedImages)

    edged= cannyEdgeDetect(blurredImages,50,100)

    flip= contourIndia(pureImages,edged)

    number=5

    # Lakukan resizing menggunakan cv2.resize
    images_to_display = [pureImages[number], grayedImages[number], blurredImages[number], edged[number], flip[number]]
    titles = ['Pure Image', 'Blurred Image', 'Segmented Image', 'Opened Image', 'Dilated Contour']

    display_images(images_to_display,titles)

handler2('Dataset')