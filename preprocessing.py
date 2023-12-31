import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils.contours import sort_contours

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

def brighten_image(images, alpha=0.5, beta=10):
    bright=[]
    for image in images:
        brightened_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        bright.append(brightened_image)

    return bright

def otsuThresholding(images):
    # List to store the thresholded (binary) images
    binary_images = []

    for image in images:
        # Convert the image to grayscale if it's not already
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Apply Otsu's thresholding
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Append the binary image to the list
        binary_images.append(binary_image)

    return binary_images

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

def sharpen(images):
    sharpened_images = []
    for image in images:
        kernel = np.array([[0, -1, 0],
                           [-1,  5, -1],
                           [0, -1, 0]])

        # Assuming image is a NumPy array
        convolved_image = cv2.filter2D(image, -1, kernel)

        sharpened_images.append(convolved_image)

    return sharpened_images

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

import cv2

import cv2
import numpy as np

def contour(pureImages, images):
    contourr = []
    number = 0
    for image in images:
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_with_contours = np.copy(pureImages[number])


        mx = (0,0,0,0)      # biggest bounding box so far
        mx_area = 0
        for cont in contours:
            x,y,w,h = cv2.boundingRect(cont)
            area = w*h
            if area > mx_area:
                mx = x,y,w,h
                mx_area = area
        x,y,w,h = mx
        croped=img_with_contours[y:y+h,x:x+w]
        # Draw contours on the copy of the original image
        contourr.append(croped)

        number += 1

    return contourr

def invert(images):
    inverted_images = []
    for image in images:
        inverted_image = 255 - image
        inverted_images.append(inverted_image)
    return inverted_images

def biggestContour(pureImages, processedImages):
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
    dim = 180
    # Membuat frame kosong dengan ukuran yang sesuai
    frame = np.zeros((rows * dim, cols * dim, 3), dtype=np.uint8)

    for i, (img, title) in enumerate(zip(images, titles)):
        # Resizing gambar menjadi (dim, dim)
        resized_img = cv2.resize(img, (dim, dim))

        # Menentukan baris dan kolom untuk menempatkan gambar
        row = i // cols
        col = i % cols

        # Menempatkan gambar pada frame
        frame[row * dim:(row + 1) * dim, col * dim:(col + 1) * dim] = resized_img

        # Menambahkan judul di bawah gambar
        cv2.putText(frame, title, (col * dim, (row + 1) * dim - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Menampilkan frame yang berisi semua gambar
    cv2.imshow('Image Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_image(image, size=(200, 200)):
    return cv2.resize(image, size)

def displayImages(images, titles):
    """
    Display a list of images with corresponding titles using Matplotlib.

    Parameters:
    - images: List of images to be displayed.
    - titles: List of titles for each image.
    """
    num_images = len(images)

    # Create subplots based on the number of images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        # Resize the image to 200x200 pixels
        resized_image = resize_image(images[i], size=(200, 200))

        # If the image is in grayscale, use cmap='gray'
        cmap = 'gray' if len(resized_image.shape) == 2 else None

        axes[i].imshow(resized_image, cmap=cmap)
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.show()

def findContours(img):
    conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sort_contours(conts, method='left-to-right')[0]

    return conts

def kunturisasi(images):
    terbaca=[]
    for image in images:
        show=[]
        kuntul=findContours(image)
        min_w, max_w = 30, 160
        min_h, max_h = 34, 140
        img_copy = image.copy() # original image for plotting countour result
        filtered_conts = []

        for c in kuntul:
            (x, y, w, h) = cv2.boundingRect(c) # find bounding box based on contour
            if(w >= min_w and w <= max_w) and (h >= min_h and h <= max_h): # if pixel follow this rule, it consider as char
                filtered_conts.append(c)
                roi = image[y:y+h, x:x+w] # get region of interest for char
                thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                show.append(thresh) # check

                # Build bounding box on original image
                cv2.rectangle(img_copy, (x,y), (x+w, y+h), (255,0,0), 2)
                gettingShit(show)
def handler3(path):
    processed= pathProcessing(path)

    pureImages=readImage(processed)

    processedImage=pureImages

    grayed=grayImages(processedImage)

    blured=gaussianBlur(grayed)

    canny=cannyEdgeDetect(blured,30,50)

    erosi=dilation(canny)

    opens=opening(erosi)
    opens=opening(opens)

    kuntur=contour(pureImages,opens)

    number=17

    images_to_display = [pureImages[number], grayed[number], blured[number], erosi[number],canny[number],opens[number],kuntur[number]]
    titles = ['Pure Image', 'Gray', 'Blurred','eroded','canny','opens','kuntur']

    return kuntur
    


def gettingShit(kuntur):
    display=[]
    titel=[]
    number=1

    for images in kuntur:
        display.append(images)
        titel.append(str(number))
        number+=1
    displayImages(display,titel)

def postCropProcessing(images):
    gray=grayImages(images)
    sharpenedImages=sharpen(gray)
    thresholding=adaptiveThreshold(sharpenedImages)
    invertion =invert(thresholding)
    erosi=erode(invertion)
    isinya=[erosi[2]]
    kunturisasi(isinya)

cropped=handler3('Dataset')
postCropProcessing(cropped)
##

def find_contours(img):
    conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sort_contours(conts, method='left-to-right')[0]

    return conts

def extract_roi(img, margin=2):
    roi = img[y - margin:y+h, x - margin:x + w + margin]
    return roi

def thresholding(img):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    print(thresh)
    return thresh

def resize_img(img, w, h):
    if w > h:
        resized = imutils.resize(img, width=28)
    else:
        resized = imutils.resize(img, height=28)

    # (w, h) = resized.shape
    (h, w) = resized.shape

    # Calculate how many pixels need to fill char image
    dX = int(max(0, 28 - w) / 2.0)
    dY = int(max(0, 28 - h) / 2.0)

    filled = cv2.copyMakeBorder(resized, top=dY, bottom=dY, right=dX, left=dX, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    filled = cv2.resize(filled, (28,28))

    return filled

def normalization(img):
    img = img.astype('float32') / 255.0 # convert to floating point
    img = np.expand_dims(img, axis=-1) # add depth
    return img

def process_box(gray, x, y, w, h):

    roi = extract_roi(gray)
    thresh = thresholding(roi)
    (h, w) = thresh.shape
    resized = resize_img(thresh, w, h)

    normalized = normalization(resized)

    return (normalized, (x, y, w, h))

def terakhir(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
    invertion = 255 - adaptive
    erode = cv2.erode(invertion, np.ones((1,1)))
    conts = find_contours(erode.copy())

    min_w, max_w = 30, 160
    min_h, max_h = 34, 140
    img_copy = img.copy() # original image for plotting countour result
    filtered_conts = []

    for c in conts:
        (x, y, w, h) = cv2.boundingRect(c) # find bounding box based on contour
        if(w >= min_w and w <= max_w) and (h >= min_h and h <= max_h): # if pixel follow this rule, it consider as char
            filtered_conts.append(c)
            roi = gray[y:y+h, x:x+w] # get region of interest for char
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # check

            # Build bounding box on original image
            cv2.rectangle(img_copy, (x,y), (x+w, y+h), (255,0,0), 2)
    
    detected_char = []

    for c in conts:
        (x, y, w, h ) = cv2.boundingRect(c)

        min_w, max_w = 30, 160
        min_h, max_h = 34, 140

        if(w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
            detected_char.append(process_box(gray, x, y, w, h))

    pixels = np.array([px[0] for px in detected_char], dtype='float32')

    # Get all box for detected char
    boxes = [box[1] for box in detected_char]
    
    model = load_model('/content/drive/MyDrive/custom_ocr.model')

    digits = '0123456789'
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWZYZ'
    char_list = digits + letters
    char_list = [ch for ch in char_list]

    preds = model.predict(pixels)

    for p in preds:
        char_idx = np.argmax(p)
        print(char_list[char_idx], end = " ")
    
    img_copy2 = img.copy()

    for(pred, (x, y, w, h)) in zip(preds, boxes):
        i = np.argmax(pred)
        char_detected = char_list[i]

        cv2.rectangle(img_copy2, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(img_copy2, char_detected, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

    return img_copy2