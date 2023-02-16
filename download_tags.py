import requests

#outdir = ./tags

urls = [
    #see used tag urls txt
]


#create out dir
import os
if not os.path.exists(os.curdir + "/tags"):
    os.mkdir(os.curdir + "/tags")

for url in urls:
    #create request
    response = requests.get(url)
    #print content
    print(response.content)

    #collect all strings in response.content that end with .jpg and are delmited by ""
    decoded = response.content.decode('utf-8')
    splits = decoded.split('\"')
    print(splits)
    #find all splits that end with .jpg
    jpgs = [x for x in splits if x.endswith('.jpg')]

    #download all jps from base url and appended jpgs
    for jpg in jpgs:
        print(f"downloading: {jpg}")
        jpgurl = url + jpg
        response = requests.get(jpgurl)
        
        if not os.path.exists(os.curdir + "/tags/" + jpg):
            with open(os.curdir + "/tags/" + jpg, 'wb') as f:
                f.write(response.content)