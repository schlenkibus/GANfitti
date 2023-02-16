import json
import os
import sys
import getopt
import re

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:o:p:", ["inputdir=", "outputdir=", "picturedir="])
except getopt.GetoptError:
    print('create_text_dataset.py -i <inputdir> -o <outputdir> -p <picturedir>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('create_text_dataset.py -i <inputdir> -o <outputdir>')
        sys.exit()
    elif opt in ("-i", "--inputdir"):
        inputdir = arg
    elif opt in ("-p", "--picturedir"):
        picturedir = arg
    elif opt in ("-o", "--outputdir"):
        outputdir = arg

placeholder_keys = ['STYLE', 'TEXT', 'HIGHLIGHTS', 'OUTLINE', 'FILL', 'BACKGROUND']

base_prompt = "a graffiti on a BACKGROUND with the word TEXT written in FILL letters with a OUTLINE outline and HIGHLIGHTS details in a STYLE style"

if not all(key in base_prompt for key in placeholder_keys):
    print('The base prompt must have the following placeholder_keys: style, text, highlights, outline, fill, background')
    sys.exit(2)

import unicodedata

def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")

for filename in os.listdir(inputdir):
    if filename.endswith(".json"):
        with open(inputdir + filename, 'r') as f:
            data = json.load(f)

            newprompt = base_prompt

            #remove the words before and after 'HIGHLIGHTS' from the prompt if we have no highlights set
            if data["highlights"] == "":
                newprompt = re.sub(r' \w+ HIGHLIGHTS \w+', '', newprompt)

            for key in placeholder_keys:
                if(key == 'BACKGROUND'):
                    if data[key.lower()] == "brick":
                        newprompt = newprompt.replace(key, "brick wall")
                        continue

                newprompt = newprompt.replace(key, data[key.lower()])

            #copy file from picture dir with the same basename as the json
            if not os.path.exists(outputdir + filename):
                baseName = os.path.basename(inputdir + filename)
                inFile = picturedir + baseName
                inFile = inFile.replace('.json', '.jpg')
                cleanOutName = re.sub('(\W+)','_', newprompt)
                outFile = outputdir + cleanOutName + u'.jpg'
                print(inFile)
                print(outFile)
                if os.path.exists(inFile):
                    os.system('cp ' + inFile + ' ' + outFile)
                else:
                    print('File does not exist: ' + inFile)
