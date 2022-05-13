#!/bin/bash

for entry in "$1"/*
do
  convert $entry -colorspace sRGB -type truecolor $entry
  convert $entry -resize 1024x1024 -gravity center -background "rgb(0,0,0)" -extent 1024x1024 +profile "*" $entry
  echo "$entry"
done
