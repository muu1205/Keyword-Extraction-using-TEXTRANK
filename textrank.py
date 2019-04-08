Text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."
#Text ="Chinese smartphone maker OnePlus has a good news for its fans. The OnePlus 6T McLaren Edition may soon be back on sale in India. According Amazon India's website, the sale of special edition will be ‘flagged off’ soon. The e-tailer, however, has not revealed the exact date when the smartphone will go on sale. Running on Snapdragon 845 and packing 10GB RAM and 256GB storage, OnePlus 6T McLaren Edition is the company's 'most-powerful' smartphone to launch so far."
# "Keirsten is a California native who quit a career in finance to travel the world. She’s been traveling for 5 years and is ‘killing it’. This girl has some amazing girl power! She works with top brands and sponsors from all over the globe. Keirsten is also a good writer, content creator, and extremely talented in photography. She is also part of the GoPro Family and has been featured on Huffington Post, Forbes, Matador Network, and many more! I have yet to meet this beauty but she's on my list! Such an inspiration."
import nltk
from nltk import word_tokenize
import string

#nltk.download('punkt')

text = word_tokenize(Text)

print("Tokenized Text: ")
print(text)

POS_tag = nltk.pos_tag(text)

print("\nTokenized Text with POS tags:")
print(POS_tag)

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

adjective_tags = ['JJ', 'JJR', 'JJS']

lemmatized_text = []

for word in POS_tag:
    if word[1] in adjective_tags:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0], pos="a")))
    else:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0])))  # default POS = noun

print("\nText tokens after lemmatization of adjectives and nouns: ")
print(lemmatized_text)

POS_tag = nltk.pos_tag(lemmatized_text)

print("\nLemmatized text with POS tags:")
print(POS_tag)

stopwords = []

wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW','NER']

for word in POS_tag:
    if word[1] not in wanted_POS:
        stopwords.append(word[0])

punctuations = list(str(string.punctuation))

stopwords = stopwords + punctuations

processed_text = []
for word in lemmatized_text:
    if word not in stopwords:
        processed_text.append(word)
print("\n",processed_text)

vocabulary = list(set(processed_text))
print("\n",vocabulary)

import numpy as np
import math

vocab_len = len(vocabulary)

weighted_edge = np.zeros((vocab_len, vocab_len), dtype=np.float32)

score = np.zeros((vocab_len), dtype=np.float32)
window_size = 3
covered_coocurrences = []

for i in range(0, vocab_len):
    score[i] = 1
    for j in range(0, vocab_len):
        if j == i:
            weighted_edge[i][j] = 0
        else:
            for window_start in range(0, (len(processed_text) - window_size)):

                window_end = window_start + window_size

                window = processed_text[window_start:window_end]

                if (vocabulary[i] in window) and (vocabulary[j] in window):

                    index_of_i = window_start + window.index(vocabulary[i])
                    index_of_j = window_start + window.index(vocabulary[j])

                    # index_of_x is the absolute position of the xth term in the window
                    # (counting from 0)
                    # in the processed_text

                    if [index_of_i, index_of_j] not in covered_coocurrences:
                        weighted_edge[i][j] += 1 / math.fabs(index_of_i - index_of_j)
                        covered_coocurrences.append([index_of_i, index_of_j])

inout = np.zeros((vocab_len),dtype=np.float32)

for i in range(0,vocab_len):
    for j in range(0,vocab_len):
        inout[i]+=weighted_edge[i][j]

MAX_ITERATIONS = 50
d = 0.85
threshold = 0.0001  # convergence threshold

for iter in range(0, MAX_ITERATIONS):
    prev_score = np.copy(score)

    for i in range(0, vocab_len):

        summation = 0
        for j in range(0, vocab_len):
            if weighted_edge[i][j] != 0:
                summation += (weighted_edge[i][j] / inout[j]) * score[j]

        score[i] = (1 - d) + d * (summation)

    if np.sum(np.fabs(prev_score - score)) <= threshold:  # convergence condition
        print("\nConverging at iteration " + str(iter) + "....")
        break

for i in range(0,vocab_len):
    print("Score of "+vocabulary[i]+": "+str(score[i]))

phrases = []

phrase = " "
for word in lemmatized_text:

    if word in stopwords:
        if phrase != " ":
            phrases.append(str(phrase).strip().split())
        phrase = " "
    elif word not in stopwords:
        phrase += str(word)
        phrase += " "

print("\nPartitioned Phrases (Candidate Keyphrases): ")
print(phrases)

unique_phrases = []

for phrase in phrases:
    if phrase not in unique_phrases:
        unique_phrases.append(phrase)

print("\nUnique Phrases (Candidate Keyphrases):")
print(unique_phrases)

for word in vocabulary:
    # print word
    for phrase in unique_phrases:
        if (word in phrase) and ([word] in unique_phrases) and (len(phrase) > 1):
            # if len(phrase)>1 then the current phrase is multi-worded.
            # if the word in vocabulary is present in unique_phrases as a single-word-phrase
            # and at the same time present as a word within a multi-worded phrase,
            # then I will remove the single-word-phrase from the list.
            unique_phrases.remove([word])

print("\nThinned Unique Phrases (Candidate Keyphrases):")
print(unique_phrases)

phrase_scores = []
keywords = []
for phrase in unique_phrases:
    phrase_score=0
    keyword = ''
    for word in phrase:
        keyword += str(word)
        keyword += " "
        phrase_score+=score[vocabulary.index(word)]
    phrase_scores.append(phrase_score)
    keywords.append(keyword.strip())

i=0
for keyword in keywords:
    print("Keyword: '"+str(keyword)+"', Score: "+str(phrase_scores[i]))
    i+=1


sorted_index = np.flip(np.argsort(phrase_scores),0)

keywords_num = 10

print("\nKeywords:")
generate=''
for i in range(0,keywords_num):
    print(str(keywords[sorted_index[i]])+", ",end="")
    generate=generate+str(keywords[sorted_index[i]])+' '

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=600, height=600,background_color='black',stopwords=stopwords,min_font_size=10).generate(generate)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()