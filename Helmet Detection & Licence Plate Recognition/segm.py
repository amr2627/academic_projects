import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

import cv2
def extract_plate(img):  # the function detects and perfors blurring on the number plate.
    plate_img = img.copy()

    # Loads the data required for detecting the license plates from cascade classifier.
    plate_cascade = cv2.CascadeClassifier('./indian_license_plate.xml')

    # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3, minNeighbors=7)

    for (x, y, w, h) in plate_rect:
        a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))
        plate = plate_img[y + a:y + h - a, x + b:x + w - b, :]
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (51, 51, 255), 3)
    cv2.imwrite('image.jpg',plate)
    return plate

cap= cv2.VideoCapture('Bikevideo1.mp4')
found=True

while found:
    ret,frame=cap.read()
    if ret:
        cv2.imshow("Iterating through Frames to find best possible image",extract_plate(frame))

        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
    else:
        found = False
cv2.destroyAllWindows()




def find_contours1(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('image2.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(
                intX)

            char_copy = np.zeros((44, 24))
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
            plt.imshow(ii, cmap='gray')

            char = cv2.subtract(255, char)

            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)


    plt.show()
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)

    return img_res


def preprocess(image) :

    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('image2.jpg',img_binary_lp)

    char_list = find_contours1(dimensions, img_binary_lp)

    return char_list

img = cv2.imread('image.jpeg')
char = preprocess(img)

for i in range(1):
    plt.subplot(1, 10, i+1)
    plt.imshow(char[i], cmap='gray')
    plt.axis('off')
    def fix_dimension(img):
        new_img = np.zeros((28, 28, 3))
        for i in range(3):
            new_img[:, :, i] = img
        return new_img

    def show_results():
        dic = {}
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, c in enumerate(characters):
            dic[i] = c
        model = tf.keras.models.load_model("64x3-CNN.model")

        output = []
        for i, ch in enumerate(char):  # iterating over the characters
            img_ = cv2.resize(ch, (28, 28))
            img = fix_dimension(img_)
            img = img.reshape(1, 28, 28, 3)  # preparing image for the model
            y_ = model.predict_classes(img)[0]  # predicting the class
            character = dic[y_]  #
            output.append(character)  # storing the result in a list

        plate_number = ''.join(output)

        return plate_number

    print("Detected license plate is ",show_results())

