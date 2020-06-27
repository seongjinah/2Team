#!/usr/bin/python

from flask import Flask
from flask import render_template
from flask import request
from bs4 import BeautifulSoup
from numpy import dot
import numpy.linalg as npl
import requests
import time
import nltk
import json
import sys
from elasticsearch import Elasticsearch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import math

app = Flask(__name__)

es_host = "0.0.0.0"
es_port = "9200"

word_d = {}


def compute_tf(s):
    bow = set()
    wordcount_d = {}

    tokenized = word_tokenize(s)
    for tok in tokenized:
        if tok not in wordcount_d.keys():
            wordcount_d[tok] = 0
        wordcount_d[tok] += 1
        bow.add(tok)

    tf_d = {}
    for word, count in wordcount_d.items():
        tf_d[word] = float(count) / len(bow)

    return tf_d


def compute_idf():
    Dval = len(content_list)
    bow = set()

    for cl in content_list:
        tokenized = word_tokenize(cl)
        for tok in tokenized:
            bow.add(tok)

    idf_d = {}
    for t in bow:
        cnt = 0
        for s in content_list:
            if t in word_tokenize(s):
                cnt += 1
        idf_d[t] = math.log(Dval / (cnt + 1))

    return idf_d


def make_word_d():
    for i in range(len(content_list)):
        tokenized = word_tokenize(content_list[i])
        for word in tokenized:
            if word not in word_d.keys():
                word_d[word] = 0
            word_d[word] += 1


def make_vector(i):
    v = []
    s = content_list[i]
    tokenized = word_tokenize(s)
    for w in word_d.keys():
        val = 0
        for t in tokenized:
            if t == w:
                val += 1
        v.append(val)
    return v


#######################################################################################
# main

nltk.download("stopwords")
nltk.download("punkt")

swlist = []
for sw in stopwords.words("english"):
    swlist.append(sw)

web_list = []
content_list = []
es = Elasticsearch([{'host': es_host, 'port': es_port}], timeout=30)


@app.route('/')
def start():
    return render_template('1.html')


@app.route('/input1')
def input1():
    return render_template('2-1.html')


@app.route('/input2')
def input2():
    return render_template('2-2.html')

@app.route('/home')
def home():
    return render_template('1.html')


@app.route('/one_url', methods=['GET', 'POST'])
def one_url():
    error = None
    value = 0
    address = request.form['onename']

    start = time.time()  # 시작시간

    resq = requests.get(address)
    soup = BeautifulSoup(resq.content, 'html.parser')

    p = soup.find_all('p')
    h1 = soup.find_all('h1')
    h2 = soup.find_all('h2')
    h3 = soup.find_all('h3')
    h4 = soup.find_all('h4')
    h5 = soup.find_all('h5')
    h6 = soup.find_all('h6')
    li = soup.find_all('li')

    s = ''

    for i in p:
        s += i.text.replace('\n', '').strip() + ' '

    for i in h1:
        s += i.text.replace('\n', '').strip() + ' '

    for i in h2:
        s += i.text.replace('\n', '').strip() + ' '

    for i in h3:
        s += i.text.replace('\n', '').strip() + ' '

    for i in h4:
        s += i.text.replace('\n', '').strip() + ' '

    for i in h5:
        s += i.text.replace('\n', '').strip() + ' '

    for i in h6:
        s += i.text.replace('\n', '').strip() + ' '

    for i in li:
        s += i.text.replace('\n', '').strip() + ' '

    if s == '':
        value = value + 1

    # 특수문자 제거
    s = re.sub('[\[\]\/!@#$%^&*().,:]', ' ', s)

    # 단어 개수 세기
    crawling_num = len(s.split())

    # 시간 계산
    crawling_time = round(time.time() - start, 3)

    doc = {}
    doc = {
        'url': address,
        'word_num': crawling_num,
        'time': crawling_time,
    }

    res = es.index(index='one_url', doc_type='data', id='1', body=doc)

    if value == 0:
        k = 0
    else:
        k = 1

    return render_template('3-1.html', address=address, num=crawling_num, time=crawling_time, k=k)


@app.route('/textfile', methods=['GET', 'POST'])
def textfile():
    error = None
    web_list = []
    content_list = []
    FileName = request.form['FileName']
    with open(FileName, 'r') as f:
        while 1:
            line = f.readline().split()
            if not line: break
            web_list.extend(line)

    multi = False
    if (len(web_list) != len(set(web_list))):
        multi = True

    lines = []
    for line in web_list:
        if line not in lines:
            lines.append(line)

    time_list = []
    num_list = []
    value = 0

    for j in range(len(lines)):
        start = time.time()

        address = lines[j]

        res = requests.get(address)
        soup = BeautifulSoup(res.content, 'html.parser')

        p = soup.find_all('p')
        h1 = soup.find_all('h1')
        h2 = soup.find_all('h2')
        h3 = soup.find_all('h3')
        h4 = soup.find_all('h4')
        h5 = soup.find_all('h5')
        h6 = soup.find_all('h6')
        li = soup.find_all('li')

        s = ''

        for i in p:
            s += i.text.replace('\n', '').strip() + ' '

        for i in h1:
            s += i.text.replace('\n', '').strip() + ' '

        for i in h2:
            s += i.text.replace('\n', '').strip() + ' '

        for i in h3:
            s += i.text.replace('\n', '').strip() + ' '

        for i in h4:
            s += i.text.replace('\n', '').strip() + ' '

        for i in h5:
            s += i.text.replace('\n', '').strip() + ' '

        for i in h6:
            s += i.text.replace('\n', '').strip() + ' '

        for i in li:
            s += i.text.replace('\n', '').strip() + ' '
        if s == '':
            value = value + 1

        # 특수문자 제거
        s = re.sub('[\[\]\/!@#$%^&*().,:]', ' ', s)

        # 단어수 계산
        num_list.append(len(s.split()))

        # 시간 계산
        time_list.append(round(time.time() - start, 3))

        # s에서 stopwords 제외 -> s_list
        s_list = []
        tokenized = word_tokenize(s)
        for tok in tokenized:
            if tok not in swlist:
                s_list.append(tok)

        # 깔끔한 s(s_list)를 content_list에 저장
        s = ' '.join(s_list)
        content_list.append(s)

    # tf-idf 계산
    idf_d = compute_idf()
    tfidf_word_list = []
    tfidf_value_list = []

    for cl in content_list:
        tf_d = compute_tf(cl)

        sort_tf_d = sorted(tf_d.items(), reverse=True, key=lambda item: item[1])
        top_10_set = list(sort_tf_d[:10])
        top_10 = []
        tf_10 = []
        for t10 in top_10_set:
            top_10.append(t10[0])
            tf_10.append(round(t10[1], 3))
        tfidf_word_list.append(top_10)
        tfidf_value_list.append(tf_10)

    # Cosine Similarity
    make_word_d()
    vector = []
    cosine_web_list = []
    cosine_value_list = []
    index_name = []

    for i in range(len(content_list)):
        vector.append(make_vector(i))

    for i in range(len(content_list)):
        cosine = {}
        for j in range(len(content_list)):
            if i == j: continue
            dotpro = dot(vector[i], vector[j])
            cossimil = dotpro / (npl.norm(vector[i]) * npl.norm(vector[j]))
            cosine[j] = cossimil
        cosine_d = sorted(cosine.items(), reverse=True, key=lambda item: item[1])
        top_3_set = list(cosine_d[:3])
        top_3 = []
        cs_3 = []
        for t3 in top_3_set:
            top_3.append(lines[t3[0]])
            cs_3.append(round(t3[1],3))
        cosine_web_list.append(top_3)
        cosine_value_list.append(cs_3)

    for j in range(len(lines)):
        doc = {}
        doc = {
            'url': lines[j],
            'word_num': num_list[j],
            'time': time_list[j],
            'tf-idf': tfidf_word_list[j],
            'tfidf-value': tfidf_value_list[j],
            'cosine-similarity': cosine_web_list[j],
            'cosine_value': cosine_value_list[j]
        }

        name = 'text_url' + str(j)
        res = es.index(index=name, doc_type='data', id=j, body=doc)
        index_name.append(name)

    if value == 0:
        k = 0
    else:
        k = 1

    return render_template('3-2.html', filename=FileName, url_list=lines, num_list=num_list, time_list=time_list, k=k,
                           multi=multi, name=index_name)


@app.route('/tf_idf', methods=['GET', 'POST'])
def tf_idf():
    error = None
    name = request.form['name']
    tf_list = []
    tf_value = []

    results = es.search(index=name, body={'from': 0, 'size': 100})

    for result in results['hits']['hits']:
        tf_list.extend(result['_source']['tf-idf'])
        tf_value.extend(result['_source']['tfidf-value'])
        url = result['_source']['url']
    return render_template('4-1.html', tf=tf_list, tf_value=tf_value, url=url)


@app.route('/cosine', methods=['GET', 'POST'])
def cosine():
    error = None
    name = request.form['name']
    cosine_list = []
    cosine_value = []

    results = es.search(index=name, body={'from': 0, 'size': 100})

    for result in results['hits']['hits']:
        cosine_list.extend(result['_source']['cosine-similarity'])
        cosine_value.extend(result['_source']['cosine_value'])
        url = result['_source']['url']
    return render_template('4-2.html', cosine=cosine_list, cosine_value=cosine_value, url=url)

