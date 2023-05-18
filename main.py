import time
import numpy as np
import cv2
import pytesseract
# import simpleaudio as sa
import serial
import time
import serial.tools.list_ports
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
from playsound import playsound

messagebox.showinfo("Instructions:", "1. Make sure the arduino box is connected to a port in your pc."
                    +"\n2. Connect your computer and mobile phone to the same wifi network.\n"
                    +"3. Download/Open IP Webcam app and type the ip that is shown on screen in the next inputs.")

portToUse = "COM6"
# Get a list of available serial ports
ports = serial.tools.list_ports.comports()
# Print the list of ports
for port in ports:
    print(port)
    if ("USB" in str(port)):
        portToUse = (str(port)).split(" - ")[0]
print(f"Using: {portToUse}")

# Set up serial connection with Arduino
# Replace 'COM3' with the name of your serial port
ser = serial.Serial(f'{portToUse}', 9600)
ser.timeout = 1  # Set timeout for reading from serial port


def getResizedImage(img, desired_height):
    # Get the original image size
    original_height, original_width = img.shape[:2]

    # Set the new image size
    new_width = int(desired_height * original_width / original_height)

    # Resize the image using the cv2.INTER_AREA parameter
    resized_img = cv2.resize(img, (new_width, desired_height), interpolation=cv2.INTER_AREA)
    return resized_img, new_width

def convexHullOfBiggestContour(threshImage):
    contours, hierarchy = cv2.findContours(threshImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour and its convex hull
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    if max_contour is not None:
        hull = cv2.convexHull(max_contour)

        # Create new image with black background
        img_out = np.zeros(threshImage.shape, dtype=np.uint8)

        # Fill convex hull with white on new image
        cv2.fillConvexPoly(img_out, hull, (255, 255, 255))

        return img_out

def getAngleToRotate(img, gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    height, width, channels = img.shape

    # detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=width/7, maxLineGap=10)

    # # get longest line
    # longest_line = 0
    # for i, line  in enumerate(lines): 
    #     x1, y1, x2, y2 = line[0] 
    #     line_length = np.sqrt( (x2-x1)**2 + (y2-y1)**2 ) 
    #     if line_length > longest_line: 
    #         longest_line = line_length 
    #         longest_line_index = i 
            
    # lines = [lines[longest_line_index]]
    # x1, y1, x2, y2 = lines[longest_line_index][0] 
    # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  #draw the longest line

    # draw lines on the original image and calculate angles
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        angles.append(angle)

    # print("TEST")
    # angles = [x for x in angles if x >= 45]
    # print(angles)
    # calculate the most common angle
    mode_angle = max(set(angles), key=angles.count)

    # rotateBy = abs(abs(mode_angle) - 90)
    # rotateBy = math.ceil(mode_angle)

    # sign = mode_angle/-mode_angle
    # rotateBy = (90-mode_angle) * -1

    if(abs(mode_angle) > 45):
        rotateBy = mode_angle + 90
    else:
        rotateBy = mode_angle

    return mode_angle,rotateBy

def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def getCroppedEndToEnd(verticalHull, img):
    contours, hierarchy = cv2.findContours(verticalHull, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the largest contour
    biggest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest_contour)

    # Crop the top and bottom third of the bounding box from the cropped image
    hfifth = int(h / 5)
    cropped = img[y:y+hfifth, x:x+w]
    cropped2 = img[y+4*hfifth:y+5*hfifth, x:x+w]
    return cropped,cropped2, w, h

def rotateClockwiseIfLandscape(img):
    height, width, channels = img.shape
    if width > height:
        # Rotate the image clockwise by 90 degrees
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return rotated_img

def squareImage(img):
    height, width, _ = img.shape

    # Determine the size of the square to make the image
    size = max(height, width)

    # Create a black square image of the determined size
    square_img = np.zeros((size, size, 3), np.uint8)

    # Calculate the offset to center the original image in the square image
    x_offset = (size - width) // 2
    y_offset = (size - height) // 2

    # Copy the original image into the center of the square image
    square_img[y_offset:y_offset+height, x_offset:x_offset+width] = img
    return square_img


# Create a dialog box to ask for the URL
root = tk.Tk()
root.withdraw()
urlOrCamIndex = simpledialog.askstring("URL or Camera Index", "Enter the URL or camera index:                 ", 
                                       initialvalue="http://192.168.#.#:8080/video")
cap = cv2.VideoCapture(urlOrCamIndex)

# check if camera opened successfully
if not cap.isOpened():
    print("Error opening video capture")

# ret, frame0 = cap.read()
# gray1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
gray1 = None
moneyValueDetected = None
threshVal = 132 #uvsi
threshVal = 149 #uvsi
threshVal = 147 #uvsi
threshVal = simpledialog.askinteger("UV threshold value", "Enter the UV threshold value for fake/real detection (default: 130):", initialvalue=130)
minimumValueVotes = 2
mainThreshold = 80
obj = {"200": 0, "20": 0, "500": 0, "50": 0, "1000": 0, "100": 0 }

while True:
    # Capture a frame
    ret, frame = cap.read()
    desiredHeight = 1000

    img = rotateClockwiseIfLandscape(img=frame)
    img,_ = getResizedImage(img=img, desired_height=desiredHeight)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray1 is None:
        gray1 = gray
    # _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(gray, 36, 255, cv2.THRESH_BINARY)
    convexHull = convexHullOfBiggestContour(threshImage=otsu)
    # motion detect Calculate absolute difference between frames
    # cv2.namedWindow("gray", cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow("gray", gray)
    # cv2.namedWindow("gray1", cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow("gray1", gray1)
    # cv2.waitKey(0)

    diff = cv2.absdiff(gray1, gray)
    # Apply threshold to detect motion
    threshOfDiff = cv2.threshold(diff, 80, 255, cv2.THRESH_BINARY)[1]
    # Apply dilation to fill in gaps
    # dilated = cv2.dilate(thresh, None, iterations=1)
    # Find contours of motion
    contours, _ = cv2.findContours(threshOfDiff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"countours len: {len(contours)}")
    motion = len(contours) > 50
    # Update previous frame

    gray1 = gray
    
    if(motion):
        moneyValueDetected = None
        obj = {"20": 0, "50": 0, "100": 0, "200": 0, "500": 0, "1000": 0 }
        ser.write(b'2')
        continue

    if(moneyValueDetected is None):
        # print("Detecting money value...")

        angleOriginal=0
        rotateBy=0
        try:
            angleOriginal, rotateBy = getAngleToRotate(img=img, gray=convexHull)    
            # print(f"angleOriginal: {angleOriginal}, rotateBy: {rotateBy}")
            rotatedHull=convexHull
            rotatedImage=img
            if(rotateBy != 0 and abs(rotateBy) != 90):
                rotatedHull = rotate(img=convexHull,angle=rotateBy)
                rotatedImage = rotate(img=img,angle=rotateBy)

            cropped, cropped2, w, h = getCroppedEndToEnd(img=rotatedImage, verticalHull=rotatedHull)
            cropped2 = rotate(img=cropped2, angle=180)

            # cv2.imshow("croppedO", cropp
            

            cropped = squareImage(cropped)
            cropped2 = squareImage(cropped2)
            cropped = rotate(img=cropped, angle=-90)
            cropped2 = rotate(img=cropped2, angle=-90)
            

            croppedGray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            croppedGray2 = cv2.cvtColor(cropped2, cv2.COLOR_BGR2GRAY)

            blockSize = 101
            c = 7
            
            blockSize = int(.36 * w)
            if(blockSize % 2 != 1):
                blockSize += 1
            c = int(.07 * blockSize)
            
            blockSize = 81
            c = 2
            blockSize = 151
            c = 2
            blockSize = 11
            c = 2
            blockSize = 27#357
            c = 7

            blockSize = int(.0756 * w)
            if(blockSize % 2 != 1):
                blockSize += 1
            # c = int(.0196 * blockSize)
            c = 7

            # print(f"blockSize: {blockSize} c: {c}")
            croppedAdaptiveThresh = cv2.adaptiveThreshold(croppedGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, c)
            croppedAdaptiveThresh2 = cv2.adaptiveThreshold(croppedGray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, c)
            
            hconcatCropped = np.concatenate((croppedAdaptiveThresh, croppedAdaptiveThresh2), axis=1)

            config = f'--psm 11 digits -l eng --user-patterns patterns.txt'
            text1 = pytesseract.image_to_string(hconcatCropped, config=config)

            # textDetected = text1.split('\n')[0]
            textDetected = text1
            # print(f"text detected: @{textDetected}@")

            # if textDetected in obj:
            #     obj[textDetected] += 1
            #     if obj[textDetected] == minimumValueVotes:
            #         moneyValueDetected = textDetected 
            #         print(f"mmmmmmmmmmoneyValue: {moneyValueDetected}")
            #         obj = {"20": 0, "50": 0, "100": 0, "200": 0, "500": 0, "1000": 0 }

            for key in reversed(obj):
                # print(key)
                if str(key) in str(textDetected):
                    obj[key] += 1
                    if obj[key] == minimumValueVotes:
                        moneyValueDetected = key 
                        print(f"MONEY VALUE: {moneyValueDetected}")
                        obj = {"20": 0, "50": 0, "100": 0, "200": 0, "500": 0, "1000": 0 }

            cv2.namedWindow("FRAME", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("FRAME", frame)
            cv2.namedWindow("Motion Detect", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("Motion Detect", threshOfDiff)
            cv2.namedWindow("Output", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("Output", img)
            cv2.namedWindow("Otsue", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("Otsue", otsu)
            cv2.namedWindow("Hull", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("Hull", convexHull)
            cv2.namedWindow("rotatedHull", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("rotatedHull", rotatedHull)
            cv2.namedWindow("rotatedImage", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("rotatedImage", rotatedImage)
            cv2.namedWindow("cropped", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("cropped", cropped)
            cv2.namedWindow("cropped2", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("cropped2", cropped2)
            cv2.namedWindow("croppedAdaptiveThresh", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("croppedAdaptiveThresh", croppedAdaptiveThresh)
            cv2.namedWindow("croppedAdaptiveThresh2", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("croppedAdaptiveThresh2", croppedAdaptiveThresh2)
            cv2.namedWindow("hconcatCropped", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("hconcatCropped", hconcatCropped)
        except Exception as e:
            print("error.", e)
    else:
        # print("Detecting UVSI...")
        converted_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(converted_img)
        _, Hthresh = cv2.threshold(h, threshVal, 255, cv2.THRESH_BINARY_INV)

        total_pixels = Hthresh.shape[0] * Hthresh.shape[1]
        # Count the number of non-zero pixels
        non_zero_pixels = cv2.countNonZero(Hthresh)
        # Check if all pixels are white
        if non_zero_pixels == total_pixels:
            # print("FAKE!!!")
            ser.write(b'1')
        else:
            # print("REAL!!!")
            ser.write(b'0')
            # response = ser.readline().decode().strip()

        if(moneyValueDetected == "20"):
            wavFileName = "Ptwenty.wav"
        elif(moneyValueDetected == "50"):
            wavFileName = "Pfifty.wav"
        elif (moneyValueDetected == "100"):
            wavFileName = "POneHundred.wav"
        elif (moneyValueDetected == "200"):
            wavFileName = "PTwoHundred.wav"
        elif (moneyValueDetected == "500"):
            wavFileName = "PFiveHundred.wav"
        elif (moneyValueDetected == "1000"):
            wavFileName = "PFiveThousand.wav"
        playsound(wavFileName)

    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for i in range(20):
        success = cap.grab()
        # print(f"success {i}")
    # cv2.waitKey(0)
    # time.sleep(1)

# Release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()

