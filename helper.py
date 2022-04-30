import os

def getFilesRecursively(directory, extension=".JPG"):
    ret = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                ret.append(os.path.join(root, file))
    return ret

def getBaseNames(paths):
    ret = []
    for path in paths:
        ret.append(os.path.basename(path))
    return ret

def getBaseName(path):
    return os.path.basename(path)

def getPathForName(name, paths):
    for path in paths:
        if os.path.basename(path) == name or os.path.splitext(os.path.basename(path))[0] == name:
            return path
    return None

def dropExtension(files):
    ret = []
    for file in files:
        ret.append(os.path.splitext(file)[0])
    return ret

def dropExtension_file(file):
    return os.path.splitext(file)[0]