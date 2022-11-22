import json
import os
import re
from nltk.tokenize import sent_tokenize
import math
from sentence_transformers import SentenceTransformer, util

class BM25:
    """
    Best Match 25.

    Parameters
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.

    doc_len_ : list[int]
        Number of terms per document. So [3] means the first
        document contains 3 terms.

    corpus_ : list[list[str]]
        The input corpus.

    corpus_size_ : int
        Number of documents in the corpus.

    avg_doc_len_ : float
        Average number of terms for documents in the corpus.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1

    def fit(self, corpus):
        """
        Fit the various statistics that are required to calculate BM25 ranking
        score using the corpus given.

        Parameters
        ----------
        corpus : list[list[str]]
            Each element in the list represents a document, and each document
            is a list of the terms.

        Returns
        -------
        self
        """
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            tf.append(frequencies)

            # compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def search(self, query):
        scores = [self._score(query, index) for index in range(self.corpus_size_)]
        return scores

    def _score(self, query, index):
        score = 0.0

        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)

        return score

def get_title(__file__: str) -> str:
    # get the title name of the file
    title = re.sub(r'\d+', '', re.sub(r'[_-]', ' ', __file__)).strip()
    return title

def get_transcript(__file__: str) -> str:
    # get the transcript data from json file
    directory_path = os.getcwd() + '/data/' + __file__
    transcript = ''
    with open(directory_path, 'r') as file:
        json_data = json.load(file)

    if json_data:
        transcript = json_data['results']['transcripts'][0]['transcript']
    return transcript

def get_times(__file__: str) -> dict:
    # get sentence times from json file
    directory_path = os.getcwd() + '/data/' + __file__
    times = ''
    with open(directory_path, 'r') as file:
        json_data = json.load(file)

    if json_data:
        times = json_data['results']['items']
    return times

def get_sentence_tokens(transcript: str) -> list:
    # get sentence tokens from transcript
    sentence_tokens = sent_tokenize(transcript)
    return sentence_tokens

def get_tokens(sentence: str) -> list:
    # get word tokens from sentence
    tokens = sentence.split()
    return tokens

def get_bm25_scores(query: str, corpus: list) -> list:
    tokenize_corpus = [get_tokens(sentence) for sentence in corpus]
    get_query = get_tokens(query)
    bm25 = BM25()
    bm25.fit(tokenize_corpus)
    scores = bm25.search(get_query)
    return scores

def get_bm25_topN(transcript:str, scores: list, n=10) -> list:
    zip_scores = zip(scores, get_sentence_tokens(transcript), )
    list_zip_scores = list(zip_scores)
    sort_scores = sorted(list_zip_scores, key=lambda x: x[0], reverse=True)
    topN_sentences = [t[1] for t in sort_scores]
    return topN_sentences[:n]

def get_bert_topN(title: str, sentences: str, n=5) -> list:
    model = SentenceTransformer('/models')
    sentence_embeddings = model.encode(sentences)
    title_embedding = model.encode(title)

    similarity_scores = []
    for embedding in sentence_embeddings:
        cosine_similarity_score = util.cos_sim(embedding, title_embedding)
        similarity_scores.append(cosine_similarity_score)

    zip_scores = zip(similarity_scores, sentences)
    list_zip_scores = list(zip_scores)
    sort_scores = sorted(list_zip_scores, key=lambda x: x[0], reverse=True)
    topN_sentences = [t[1] for t in sort_scores]
    return topN_sentences[:n]