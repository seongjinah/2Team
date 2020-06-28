#!/bin/bash

pip install numpy
pip install django
pip install nltk

if [ ! -d templates ]; then
    mkdir templates
fi

if [ ! -d static ]; then
    mkdir -p static/css
fi

cp -f 1.html templates
cp -f 2-1.html templates
cp -f 2-2.html templates
cp -f 3-1.html templates
cp -f 3-2.html templates
cp -f 4-1.html templates
cp -f 4-2.html templates
cp -f 7.html templates

cp -f 1.css static/css
cp -f 2-1.css static/css
cp -f 2-2.css static/css
cp -f 3-1.css static/css
cp -f 3-2.css static/css
cp -f 4-1.css static/css
cp -f 4-2.css static/css
cp -f 7.css static/css

$HOME/elasticsearch-7.6.2/bin/elasticsearch -d

flask run 
