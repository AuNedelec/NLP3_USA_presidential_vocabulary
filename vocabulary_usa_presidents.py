#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install gensim')
import os
import gensim
import spacy
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words


# In[ ]:


# specify the directory containing the speech files
speeches_dir = "Project_3_Word_Embeddings/president_speeches"
# get list of all speech files
files = sorted([file for file in os.listdir() if file [-4:]== '.txt'])
print(files)


# In[ ]:


# specify the directory containing the speech files
speeches_dir = "Project_3_Word_Embeddings/president_speeches"
# get list of all speech files
files = sorted([file for file in os.listdir() if file [-4:]== '.txt'])

# read each speech file
speeches = [read_file(file) for file in files]

# preprocess each speech
processed_speeches = process_speeches(speeches)

# merge speeches
all_sentences = merge_speeches(processed_speeches)

# view most frequently used words
most_freq_words = most_frequent_words(all_sentences)

# create gensim model of all speeches
all_prez_embeddings = gensim.models.Word2Vec(all_sentences, size =96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom
similar_to_freedom = all_prez_embeddings.most_similar('freedom', topn=20)

similar_to_family = all_prez_embeddings.most_similar('family', topn=20)

# get President Roosevelt sentences
roosevelt_sentences = get_president_sentences('franklin-d-roosevelt')

# view most frequently used words of Roosevelt
roosevelt = most_frequent_words(roosevelt_sentences)

# create gensim model for Roosevelt

roosevelt_embeddings = gensim.models.Word2Vec(roosevelt_sentences, size =96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for Roosevelt
roosevelt_sim_to_freedom = roosevelt_embeddings.most_similar('freedom', topn=20)

# get sentences of multiple presidents
rushmore_prez_sentences = get_presidents_sentences(['washington', 'jefferson', 'lincoln', 'theodore-roosevelt'])

# view most frequently used words of presidents

rushmore = most_frequent_words(rushmore_prez_sentences)

# create gensim model for the presidents
rushmore_embeddings = gensim.models.Word2Vec(rushmore_prez_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for presidents
freedom = rushmore_embeddings.most_similar('freedom', topn=20)
print(freedom)

