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

#parse
try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:",["help","input="])
except getopt.GetoptError:
    print("cleanup_dataset.py -i <input>")
    sys.exit(2)

for opt, arg in opts:
    if opt in ('-h', "--help"):
        print("cleanup_dataset.py -i <input>")
        sys.exit(1)
    elif opt in ("-i", "--input"):
        labelDirectoryPath = arg

#find all files in lable directory
labelFiles = getFilesRecursively(labelDirectoryPath, ".txt")

#print numlabels
numLabels = len(labelFiles)
print(f"num labels: {numLabels}")

for labelFile in labelFiles:
    with open(labelFile, "r") as f:
        lines = f.readlines()
    
    oldNumLines = len(lines)

    #cleanup according to yolov5 format
    for iter, line in enumerate(lines):    
        if not line.startswith("0 "):
            lines[iter] = "0 " + line

        lines[iter] = lines[iter].replace(",", " ")

    #overwrite file
    with open(labelFile, "w") as f:
        f.writelines(lines)

    #count number of lines in file
    with open(labelFile, "r") as f:
        numLines = len(f.readlines())
        #assert that old and new number of lines are the same
        assert oldNumLines == numLines