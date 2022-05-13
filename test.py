import os
import sys

latest = 0
path = "/tmp"
directoryName = None

for file in os.listdir(path):
    if os.path.isdir(os.path.join(path, file)):
        fullPath = os.path.join(path, file)
        timestamp = os.path.getmtime(fullPath)
        if latest < timestamp:
            latest = timestamp
            directoryName = file

print("Newest with name: " + directoryName)