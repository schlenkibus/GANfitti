#!/bin/bash

for entry in "$1"/*
do
  convert $entry -colorspace sRGB -type truecolor $entry
  convert $entry -resize 512x512 -gravity center -background "rgb(0,0,0)" -extent 512x512 +profile "*" $entry
  echo "$entry"
done
