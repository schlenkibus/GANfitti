import json
import os
import sys
import getopt
import re

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:o:p:", ["inputdir=", "outputdir="])
except getopt.GetoptError:
    print('create_text_dataset.py -i <inputdir> -o <outputdir>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('create_text_dataset.py -i <inputdir> -o <outputdir>')
        sys.exit()
    elif opt in ("-i", "--inputdir"):
        inputdir = arg
    elif opt in ("-o", "--outputdir"):
        outputdir = arg

placeholder_keys = ['STYLE', 'TEXT', 'HIGHLIGHTS', 'OUTLINE', 'FILL', 'BACKGROUND']

base_prompt = "a graffiti on a BACKGROUND with the word TEXT written in FILL letters with a OUTLINE outline and HIGHLIGHTS details in a STYLE style"

if not all(key in base_prompt for key in placeholder_keys):
    print('The base prompt must have the following placeholder_keys: style, text, highlights, outline, fill, background')
    sys.exit(2)

with open(outputdir + "/prompts.txt", "w") as output_file:
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

                output_file.write((newprompt + "\n").encode('utf8'))
