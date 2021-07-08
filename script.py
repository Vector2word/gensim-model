import os
import gensim
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words

# get list of all speech files
files = sorted([file for file in os.listdir() if file[-4:] == '.txt'])


# read each speech file
print(files)


# preprocess each speech
speeches = [read_file(item) for item in files]


# merge speeches
processed_speeches = process_speeches(speeches)
all_sentences = merge_speeches(processed_speeches)

# view most frequently used words
most_freq_words = most_frequent_words(all_sentences)
#print(most_freq_words)


# create gensim model of all speeches
all_prez_embeddings = gensim.models.Word2Vec(all_sentences, vector_size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom
similar_to_freedom = all_prez_embeddings.wv.most_similar("freedom", topn=20)
print(similar_to_freedom)





