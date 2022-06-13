import cv2, os
import cython
import json


def find_interest_points(image,h,w):
    '''
    Input is a binary image. An Interest point is a point where it goes white to black or black to white
    :param image:
    :param h:
    :param w:
    :return:
    '''
    interest = []
    # loop through each pixel in the image
    for i in range(1,h-1):
        for j in range(1,w-1):
            if image[i][j] == 1:
                if image[i-1] ==  0 or image[i+1] == 0 or image[j-1] ==  0 or image[j+1]:
                    interest.append((i,j))
    return interest

def label_image(image):
    h = image.shape[0]
    w = image.shape[1]


def tiff_to_png(image_path):
    # convert image and return
    image = cv2.imread(image_path, 0)
    # define a threshold, 128 is the middle of black and white in grey scale
    thresh = 128
    # threshold the image
    img_binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_binary

def main():
    '''
    The main method takes user input for image path, and saves the json image in the same folder.
    :return:
    '''
    # get the path to the data
    print("Batch or One Image?")
    num_vals = input()
    #TODO: setup for batch
    if(num_vals == "Batch" or num_vals == "batch"):
        print("NEED TO IMPLEMENT FOR BATCH")

    print("Input image path:")
    path = input()
    # change to black and white
    image = tiff_to_png(path)
    cv2.imshow("input image", image)
    print("NUMBER OF CELLS")
    cell_count = input().int()
    print("left to right top to bottom cell labels:")
    labels = []
    for cell in range(0,cell_count):
        print("cell #" + cell + ":")
        labels.append(input())






if __name__ == '__main__':
    main()
