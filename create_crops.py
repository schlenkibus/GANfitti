import re
from tkinter import *
from PIL import Image, ImageTk
import glob
import os
import sys
import getopt

from helper import *

labelDirectoryPath = ""
outputPath = ""
imagePath = "" 
#parse arguments
try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:l:",["help","input=","output=","label="])
except getopt.GetoptError:
    print("create_crops.py -i <input> -o <output> -l <label>")
    sys.exit(2) 

for opt, arg in opts:
    if opt in ('-h', "--help"):
        print("create_crops.py -i <input> -o <output> -l <label>")
        sys.exit(1)
    elif opt in ("-i", "--input"):
        imagePath = arg
    elif opt in ("-o", "--output"):
        outputPath = arg
    elif opt in ("-l", "--label"):
        labelDirectoryPath = arg

#find all files in lable directory
labelFiles = glob.glob(labelDirectoryPath + "/*.txt")
labelNames = getBaseNames(labelFiles)
rawLabelNames = dropExtension(labelNames)
imageFiles = getFilesRecursively(imagePath)
imageNames = getBaseNames(imageFiles)
rawImageNames = dropExtension(imageNames)
print(f"num images: {len(imageFiles)}")

#count lines in label files
labelLines = []
for labelFile in labelFiles:
    with open(labelFile, "r") as f:
        labelLines.append(len(f.readlines()))

#accumulate lines
totalLines = 0
for labelLine in labelLines:
    totalLines += labelLine

print(f"total files: {str(len(labelFiles))} with {str(totalLines)} lines")

#find imageFile based on labelFile basename
for iter, labelFile in enumerate(rawLabelNames):
    imagePath = getPathForName(labelFile, imageFiles)

    labelLines = []
    labelFile = labelFiles[iter]
    with open(labelFile, "r") as f:
        for line in f:
            labelLines.append(line)

    image = Image.open(imagePath)
    size = image.size

    for it, line in enumerate(labelLines):
        #split into startX,startY,endX,endY
        line = line.split(",")
        startX = int(float(line[0]) * size[0])
        startY = int(float(line[1]) * size[1])
        endX = int(float(line[2]) * size[0])
        endY = int(float(line[3]) * size[1])
        print(f"{labelFile}:{it} {startX} {startY} {endX} {endY}")
        #crop image
        croppedImage = image.crop((startX, startY, endX, endY))
        croppedImage.save(outputPath + "/" + rawLabelNames[iter] + "_" + str(it) + ".jpg")

    print(str(iter) + ": " + labelFile + " -> " + imagePath)
