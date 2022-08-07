import re
from tkinter import *
from PIL import Image, ImageTk
import glob
import os
import sys
import getopt
import json

from helper import *


imageDirectoryPath = ""
labelOutPath = ""

#parse arguments
try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["help","input_images=","output="])
except getopt.GetoptError:
    print("create_text_labels_editor.py -i <input_images> -o <output>")
    sys.exit(2)

for opt, arg in opts:
    if opt in ('-h', "--help"):
        print("create_text_labels_editor.py -i <input_images> -o <output>")
        sys.exit(1)
    elif opt in ("-i", "--input_images"):
        imageDirectoryPath = arg
    elif opt in ("-o", "--output"):
        labelOutPath = arg

print(imageDirectoryPath)
print(labelOutPath)

#Images like DSC0000_0.jpg
#text labels saved in outDir DSC0000_0.json
#contents:
#[
# {
#   "text": {molt},
#   "outline": {color},
#   "second-outline": {color},
#   "fill": {color},
#   "highlights": {color},
#   "style": {bubble,block,edgy,abstract,toy,wildstyle,masterpiece,tag},
#   "background-color": {color},
#   "background": {brick, concrete, wood, truck, electrical_box, train},
#   "character": {cat}
# }
# ]

#update/save metadata file where all used values for keys are stored


images = getFilesRecursively(imageDirectoryPath, ".jpg")
images.sort()
baseNameImages = getBaseNames(images)
baseNameImagesWithoutExtension = dropExtension(baseNameImages)
print(baseNameImagesWithoutExtension)

def getNextImageToLabel():
    textLabels = getFilesRecursively(labelOutPath, ".json")
    baseNameTextLabels = getBaseNames(textLabels)
    baseNameTextLabelsWithoutExtension = dropExtension(baseNameTextLabels)
    unprocessed_images = list(set(baseNameImagesWithoutExtension) - set(baseNameTextLabelsWithoutExtension))
    unprocessed_images.sort()
    unprocessed_images = addExtensionToFiles(unprocessed_images, "jpg")
    print(len(unprocessed_images))
    if len(unprocessed_images) > 0: 
       return unprocessed_images[0]
    else:
        return None

def getNextImageToLabelPath():
    nextImageTolabel = getNextImageToLabel()
    fullPath = os.path.join(imageDirectoryPath, nextImageTolabel)
    print(fullPath)
    return fullPath

def getCurrentLabelOutPath():
    nextImageToLabel = dropExtension_file(getNextImageToLabel())
    fullPath = os.path.join(labelOutPath, nextImageToLabel)
    fullPath += ".json"
    print(fullPath)
    return fullPath
    
def getNumDone():
    labels = getFilesRecursively(labelOutPath, ".json")
    return len(labels)

def getNumTotal():
    images = getFilesRecursively(imageDirectoryPath, ".jpg")
    return len(images)

from flask import Flask
from flask.helpers import url_for
from flask.json import JSONEncoder
from flask import send_file, request, globals, redirect
from flask.templating import render_template
from flask.views import View
import sys, os, uuid, copy
import multiprocessing, subprocess
import helper as FlaskHelpers

app = Flask(__name__)

@app.route('/current-image')
def currentImage():
    return send_file(getNextImageToLabelPath())

@app.route('/', methods=['GET', 'POST'])
def default():
    return render_template("graff-label-editor.html", total_avail = getNumTotal(), done = getNumDone())
	
@app.route('/set-label', methods = ['POST'])
def upload_label():
    if request.method == 'POST':
        app.logger.info(request.form)
        dText = request.form["text"]
        dBackground = request.form["background"]
        dFill = request.form["fill"]
        dOutline = request.form["outline"]
        dStyle = request.form["style"]
        dHighlights = request.form["highlights"]

        data = {
            "text": dText,
            "background": dBackground,
            "fill": dFill,
            "outline": dOutline,
            "style": dStyle,
            "highlights": dHighlights
        }

        with open(getCurrentLabelOutPath(), "w") as f:
            json.dump(data, f)

    return redirect('/')

app.run(host='0.0.0.0', debug=True)

exit(2)

def loadLabels():
    global labelNames
    global labelFiles
    global rawLabelNames
    print(labelOutPath)
    labelFiles = getFilesRecursively(labelOutPath, ".txt")
    labelNames = getBaseNames(labelFiles)
    rawLabelNames = dropExtension(labelNames)
    #check if all files are unique
    if len(rawLabelNames) != len(set(rawLabelNames)):
        print("ERROR: Labels are not unique")
        sys.exit(1)

    #count number of lines in each labelFile
    labelCount = 0
    for labelFile in labelFiles:
        with open(labelFile, "r") as f:
            labelCount += len(f.readlines())
    
    #print labelCounts
    print(f"Found {len(rawLabelNames)} labeled files with {labelCount} labels")

imagePaths = getFilesRecursively(imageDirectoryPath)
imageNames = getBaseNames(imagePaths)
rawImageNames = dropExtension(imageNames)
loadLabels()

imageBuffer = None
photoImageBuffer = None
currentSize = None
currentImageId = None
currentImageName = None
currentLabelBuffer = []

crossHairH = None
crossHairV = None

#check if all files are unique
if len(rawImageNames) != len(set(rawImageNames)):
    print("ERROR: Images are not unique")
    sys.exit(1)

#check if all files are unique
if len(rawLabelNames) != len(set(rawLabelNames)):
    print("ERROR: Labels are not unique")
    sys.exit(1)

print(f"Found {len(rawImageNames)} images and {len(rawLabelNames)} labels")

#selectionState
startPosition = None
rectID = None

def pressed(event):
    global startPosition
    startPosition = (event.x, event.y)
    print(f"pressed {event.x}, {event.y}")


def released(event):
    global startPosition
    if startPosition == None:
        return

    global currentLabelBuffer
    startX = max(startPosition[0] / currentSize[0], 0)
    startY = max(startPosition[1] / currentSize[1], 0)
    endX = min(event.x / currentSize[0], 1)
    endY = min(event.y / currentSize[1], 1)

    currentLabelBuffer.append(f"{min(startX, endX)},{min(startY, endY)},{max(endX, startX)},{max(endY, startY)}") 
    print(f"added label: {currentLabelBuffer[-1]}")

    global rectID
    global canvas

    if rectID:
        canvas.delete(rectID)

def drag(event):
    global canvas
    global rectID
    global startPosition

    mouseMove(event)

    if rectID:
        canvas.delete(rectID)
    rectID = canvas.create_rectangle(startPosition[0], startPosition[1], event.x, event.y)

def key(event):
    print(f"key pressed: {event.char}")
    
    if event.char == "f":
        saveFullLabel()
    elif event.char == "n":
        loadNextImage()
        

def saveLabelBuffer(imageName):
    global currentLabelBuffer
    global labelOutPath

    #save labels
    labelPath = os.path.join(labelOutPath, dropExtension_file(imageName) + ".txt")
    print(labelPath)
    with open(labelPath, "w") as f:
        for label in currentLabelBuffer:
            f.write(label + "\n")

    currentLabelBuffer = []
    return None

def loadNextImage():
    global imageNames
    global labelNames

    global imageBuffer
    global photoImageBuffer
    global currentSize
    global canvas
    global currentImageId
    global currentImageName

    #save current labels
    if currentImageName != None:
        saveLabelBuffer(currentImageName)
        loadLabels()

    #find imageName that is not in labelNames
    nextImage = None
    for imageName in rawImageNames:
        if imageName not in rawLabelNames:
            nextImage = imageName
            break
    
    if nextImage == None:
        print("ERROR: No images left to label")
        sys.exit(1)
    else:
        print(f"Next image: {nextImage}")

    #load image

    if currentImageId != None:
        canvas.delete(currentImageId)

    imagePath = getPathForName(nextImage, imagePaths)
    if imagePath == None:
        print("ERROR: Could not find image")
        sys.exit(1)

    currentImageName = getBaseName(imagePath)
    imageBuffer = Image.open(imagePath)
    imageBuffer = imageBuffer.resize((1200,900))
    photoImageBuffer = ImageTk.PhotoImage(imageBuffer)
    currentSize = imageBuffer.size
    currentImageId = canvas.create_image(0, 0, image=photoImageBuffer, anchor=NW)
    canvas.pack()

def saveFullLabel():
    global currentLabelBuffer
    currentLabelBuffer.append(f"{0},{0},{1},{1}")
    loadNextImage()

def mouseMove(event):
    global canvas
    global crossHairH
    global crossHairV
    print(f"mouse move: {event.x}, {event.y}")

    if crossHairH:
        canvas.delete(crossHairH)
    
    if crossHairV:
        canvas.delete(crossHairV)
    
    crossHairH = canvas.create_line(event.x, 0, event.x, canvas.winfo_height(), fill="red")
    crossHairV = canvas.create_line(0, event.y, canvas.winfo_width(), event.y, fill="red")

root = Tk()
root.title("GANfitti AOI-Tool")
root.geometry("1200x900")
canvas = Canvas(root, width=1200, height=800)
canvas.bind("<Button-1>", pressed)
canvas.bind("<ButtonRelease-1>", released)
canvas.bind("<B1-Motion>", drag)
canvas.bind("<Motion>", mouseMove)
root.bind("<Key>", key)
next = Button(root, text="Save Labels and load Next (N)", command=loadNextImage)
fullImage = Button(root, text="Save Full Image Label and load Next (F)", command=saveFullLabel)
next.pack()
fullImage.pack()
canvas.pack()

loadNextImage()

root.mainloop()