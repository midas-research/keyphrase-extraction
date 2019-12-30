#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 09:21:38 2019

@author: r17935avinash
"""

#from flair.data import Sentence
#from flair.embeddings import CharacterCNNEmbeddings
#sentence = Sentence('Avinash is a bad boy')
#embedding = CharacterCNNEmbeddings()
#embedding.embed(sentence)
#for token in sentence:
#    print(token)
#    print(token.embedding)


from flair.data import Corpus
from flair.datasets import CONLL_03
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterCNNEmbeddings, FlairEmbeddings, FastTextEmbeddings
from typing import List

# 1. get the corpus
corpus: Corpus = CONLL_03(base_path='resources/tasks')
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('glove'),
    # FastTextEmbeddings(),
    # comment in this line to use character embeddings
     CharacterCNNEmbeddings(),

    # comment in these lines to use flair embeddings
   # FlairEmbeddings('news-forward'),
   # FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=200,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-ner',
              learning_rate=0.015,
              mini_batch_size=10,
              max_epochs=80)

# 8. plot weight traces (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_weights('resources/taggers/example-ner/weights.txt')
