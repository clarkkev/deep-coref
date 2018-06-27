# Coreference Resolution with Deep Learning

This repository contains code for training and running the neural coreference models decribed in two papers:
* ["Deep Reinforcement Learning for Mention-Ranking Coreference Models"](http://cs.stanford.edu/people/kevclark/resources/clark-manning-emnlp2016-deep.pdf), Kevin Clark and Christopher D. Manning, EMNLP 2016.
* ["Improving Coreference Resolution by Learning Entity-Level Distributed Representations"](http://cs.stanford.edu/people/kevclark/resources/clark-manning-acl16-improving.pdf), Kevin Clark and Christopher D. Manning, ACL 2016.

[Hugging Face](https://huggingface.co/) built a [coreference system](https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30) based on this
  one with a [cool demo](https://huggingface.co/coref/). Their system is also [on
  github](https://github.com/huggingface/neuralcoref).

### Requirements
Theano, numpy, and scikit-learn. It also uses a slightly modified version of keras 0.2; run `python setup.py install` in the modified_keras directory to install.

### Usage
#### Running an already-trained model
The easiest way of doing this is within Stanford's [CoreNLP](https://github.com/stanfordnlp/CoreNLP), where forward-prop for the models has been implemented in Java. Example usage:
```bash
java -Xmx5g -cp stanford-corenlp-3.7.0.jar:stanford-corenlp-models-3.7.0.jar:* edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,mention,coref -coref.algorithm neural -file example_file.txt
```
See the [CorefAnnotator](http://stanfordnlp.github.io/CoreNLP/coref.html) page for more details.


#### Training your own model
Do the following to train and evaluate the neural mention-ranking model with reward rescaling (the highest scoring model from the papers).

1. Download the CoNLL training data from [here](http://conll.cemantix.org/2012/data.html).

2. Download pretrained word embeddings. We use 50 dimensional word2vec embeddings for English ([link](https://drive.google.com/open?id=0B5Y5rz_RUKRmdEFPcGIwZ2xLRW8)) and 64 dimenensional [polyglot](https://sites.google.com/site/rmyeid/projects/polyglot) embeddings for Chinese ([link](http://bit.ly/19bTKeS)) in our paper.

3. Run the [NeuralCorefDataExporter](https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/coref/neural/NeuralCorefDataExporter.java) class in the development version of Stanford's CoreNLP (you will need to fork from the [github](https://github.com/stanfordnlp/CoreNLP/)) using the [neural-coref-conll](https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/coref/properties/neural-english-conll.properties) properties file. This does mention detection and feature extraction on the CoNLL data and then outputs the results as json. The command is
```bash
java -Xmx2g -cp stanford-corenlp.jar:stanford-corenlp-models-3.7.0.jar:* edu.stanford.nlp.coref.neural.NeuralCorefDataExporter <properties-file> <output-path>
```

4. Run run_all.py, preferably on a GPU. Training takes roughly 7 days on a GTX TITAN GPU.

run_all.py also contains methods to train the other models from the papers.

Once a model is trained, you can use pairwise_learning.py to evaluate the model and output_utils.py to view its predictions.

#### Performance
Following the above instructions will replicate results from the 2016 EMNLP paper (~65.7 CoNLL F1 on the CoNLL 2012 English test set). However, we noticed that using rule-based mention filtering from Stanford's deterministic coreference system is significantly decreasing the score. Add ```coref.md.liberalMD=true``` to the properties file during feature extraction (step 3) to disable this filtering and achieve even better performance (~66.9 CoNLL F1 on the CoNLL 2012 English test set).
