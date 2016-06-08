# Entity-Centric Coreference Resolution with Deep Learning

This repository contains code for training and running the neural coreference models decribed in ["Improving Coreference Resolution by Learning Entity-Level Distributed Representations"](http://cs.stanford.edu/people/kevclark/resources/clark-manning-acl16-improving.pdf), Kevin Clark and Christopher D. Manning, ACL 2016. 

### Requirements
Theano, numpy, and scikit-learn. It also uses a slightly modified version of keras 0.2; run `python setup.py install` in the modified_keras directory to install.

### Usage
#### Running an already-trained model
The easiest way of doing this is within Stanford's [CoreNLP](https://github.com/stanfordnlp/CoreNLP), where forward-prop for the models has been implemented in Java. Example usage: 
```
java -Xmx5g -cp stanford-corenlp.jar edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,mention,coref -coref.algorithm neural -file example_file.txt
```
You will need to fork the latest version from github and download the latest models from [here](http://nlp.stanford.edu/software/stanford-corenlp-models-current.jar).

#### Training your own model
1. Download pretrained word embeddings. We use 50 dimensional word2vec embeddings for English ([link](https://drive.google.com/open?id=0B5Y5rz_RUKRmdEFPcGIwZ2xLRW8)) and 64 dimenensional [polyglot](https://sites.google.com/site/rmyeid/projects/polyglot) embeddings for Chinese ([link](http://bit.ly/19bTKeS)) in our paper.
2. Run the [NeuralCorefDataExporter](https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/coref/neural/NeuralCorefDataExporter.java) class in the development version Stanford's CoreNLP. This does mention detection and feature extraction on the CoNLL data and then outputs the results as json.
3. Run run_all.py, preferably on a GPU. Training takes roughly 4 days on a GTX TITAN GPU.
