import re
import shutil
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
    print("create_yolov5_dataset.py -i <input> -o <output> -l <label>")
    sys.exit(2) 

for opt, arg in opts:
    if opt in ('-h', "--help"):
        print("create_yolov5_dataset.py -i <input> -o <output> -l <label>")
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

print(outputPath)

ratioTrain = 0.7
ratioVal = 0.15
ratioTest = 0.15

#create directory structure: train, val, test with images and labels each
trainPath = outputPath + "/train"
valPath = outputPath + "/val"
testPath = outputPath + "/test"

if not os.path.exists(trainPath):
    os.mkdir(trainPath)

if not os.path.exists(valPath):
    os.mkdir(valPath)

if not os.path.exists(testPath):
    os.mkdir(testPath)

for p in [trainPath, valPath, testPath]:
    if not os.path.exists(p + "/images"):
        os.mkdir(p + "/images")
    
    if not os.path.exists(p + "/labels"):
        os.mkdir(p + "/labels")

#find imageFile based on labelFile basename
for iter, labelFile in enumerate(rawLabelNames):
    imagePath = getPathForName(labelFile, imageFiles)

    labelLines = []
    labelFile = labelFiles[iter]
    with open(labelFile, "r") as f:
        for line in f:
            print(line)
            if line.startswith("0,0,1,1"):
                print("dropped line")
                continue
            labelLines.append(line)

    if len(labelLines) == 0:
        print(f"dropped label file {labelFile}")
        continue

    #clean up label lines
    for it, line in enumerate(labelLines):
        labelLines[it] = "0 " + line
        labelLines[it] = labelLines[it].replace(",", " ")

    totalLabelFiles = len(rawLabelNames)
    trainLabelFiles = int(totalLabelFiles * ratioTrain)
    valLabelFiles = int(totalLabelFiles * ratioVal)
    testLabelFiles = int(totalLabelFiles * ratioTest)

    isTrainLabelFile = iter < trainLabelFiles
    isValLabelFile = iter >= trainLabelFiles and iter < (trainLabelFiles + valLabelFiles)
    isTestLabelFile = iter >= (trainLabelFiles + valLabelFiles)

    if isTrainLabelFile:
        labelFile = trainPath + "/labels/" + labelNames[iter]
        imageFile = trainPath + "/images/" + imageNames[iter]
        #save files
        with open(labelFile, "w") as f:
            for line in labelLines:
                f.write(line)
        shutil.copy(imagePath, imageFile)
        print(f"train {labelFile} -> {imageFile}")

    if isValLabelFile:
        labelFile = valPath + "/labels/" + labelNames[iter]
        imageFile = valPath + "/images/" + imageNames[iter]
        #save files
        with open(labelFile, "w") as f:
            for line in labelLines:
                f.write(line)
        shutil.copy(imagePath, imageFile)
        print(f"val {labelFile} -> {imageFile}")

    if isTestLabelFile:
        labelFile = testPath + "/labels/" + labelNames[iter]
        imageFile = testPath + "/images/" + imageNames[iter]
        #save files
        with open(labelFile, "w") as f:
            for line in labelLines:
                f.write(line)
        shutil.copy(imagePath, imageFile)
        print(f"test {labelFile} -> {imageFile}")