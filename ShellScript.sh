#!/bin/bash

$HOME/elasticsearch-7.6.2/bin/elasticsearch -d

pip install numpy
pip install django
pip install nltk

if [ ! -d templates ]; then
    mkdir templates
fi

if [ ! -d static ]; then
    mkdir -p static/css
fi

cp -f $HOME/2Team/1.html templates
cp -f $HOME/2Team/2-1.html templates
cp -f $HOME/2Team/2-2.html templates
cp -f $HOME/2Team/3-1.html templates
cp -f $HOME/2Team/3-2.html templates
cp -f $HOME/2Team/4-1.html templates
cp -f $HOME/2Team/4-2.html templates
cp -f $HOME/2Team/7.html templates

cp -f $HOME/2Team/1.css static/css
cp -f $HOME/2Team/2-1.css static/css
cp -f $HOME/2Team/2-2.css static/css
cp -f $HOME/2Team/3-1.css static/css
cp -f $HOME/2Team/3-2.css static/css
cp -f $HOME/2Team/4-1.css static/css
cp -f $HOME/2Team/4-2.css static/css
cp -f $HOME/2Team/7.css static/css

cp -f $HOME/2Team/app.py $HOME
cp -f $HOME/2Team/textfile.txt $HOME

flask run 
