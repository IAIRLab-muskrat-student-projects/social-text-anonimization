
import re

import re
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from itertools import chain
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = WordNetLemmatizer()

def preprocess(text, stopwords):
    text = re.sub(r'@\w+', ' ', text.lower())
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = text.split()
    words = filter(lambda x: x not in stopwords, words)
    return ' '.join(lemmatizer.lemmatize(x) for x in words)

# Получение синонимов для уникальных слов
def get_synonyms(uniq_words):
    all_synonyms = set()
    for word in uniq_words:
        synonyms = wordnet.synsets(word)
        all_synonyms.update(chain.from_iterable([word.lemma_names() for word in synonyms]))
    return all_synonyms

# Функция для генерации вероятностей замены слова
def exponential(x, R, u, sensitivity=1, epsilon=25.4):
    scores = [u(x, r) for r in R]
    probabilities = [np.exp(epsilon * score / (2 * sensitivity)) for score in scores]
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)
    return probabilities

def exponential_gen(x, R, u, sensitivity=1, epsilon=25.4):
    scores = [u(x, r) for r in R]
    probabilities = [np.exp(epsilon * score / (2 * sensitivity)) for score in scores]
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)
    return np.random.choice(R, 1, p=probabilities)

def kgram_overlap(word1, word2, k):
    a = set([word1[i:i+k] for i in range(0, len(word1) - k + 1)])
    b = set([word2[i:i+k] for i in range(0, len(word2) - k + 1)])
    inter = len(a.intersection(b))
    return inter / (len(a) + len(b) - inter)

def score(word1, word2, tfidf, word_similarities):
    idx1, idx2 = tfidf.vocabulary_[word1], tfidf.vocabulary_[word2]
    return word_similarities[idx1, idx2] - 0.3 * kgram_overlap(word1, word2, 2)

def anonymize_with_syntf(docs, stopwords, use_synsets=False, epsilon=25.4):
    # Preprocess documents
    processed_docs = [preprocess(doc, stopwords) for doc in docs]
    
    # Synonym handling
    if use_synsets:
        uniq_words = set(chain.from_iterable([doc.split() for doc in processed_docs]))
        synonyms = ' '.join(get_synonyms(uniq_words))
        docs_with_synonyms = [*processed_docs, synonyms]
    else:
        docs_with_synonyms = processed_docs

    # Compute TF-IDF
    tfidf = TfidfVectorizer()
    tfidf.fit(docs_with_synonyms)
    doc_vecs = tfidf.transform(processed_docs)
    doc_vecs = normalize(doc_vecs, norm='l1')
    words = tfidf.get_feature_names_out()

    # Compute word similarities
    word_vecs = np.random.rand(len(words), 300)  # Replace with real embeddings
    word_similarities = cosine_similarity(word_vecs, word_vecs)

    # Compute replacement probabilities
    word_replace_probs = []
    for word in tqdm(words):
        word_replace_probs.append(exponential(word, words, lambda x, y: score(x, y, tfidf, word_similarities)))
    word_replace_probs = np.array(word_replace_probs)

    # Anonymize documents
    anonymized_docs = []
    for idx, doc in enumerate(processed_docs):
        words_count = len(doc.split())
        words_ = np.random.choice(words, words_count, p=doc_vecs[idx].todense().tolist()[0])
        for i in range(words_count):
            word_idx = tfidf.vocabulary_[words_[i]]
            words_[i] = np.random.choice(words, 1, p=word_replace_probs[word_idx])[0]
        anonymized_docs.append(' '.join(words_))

    return anonymized_docs

        