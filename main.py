import pyterrier as pt
import nltk
import pandas as pd
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-22"

if not pt.started():
    pt.init()

categories = ['business', 'entertainment', 'food',
              'graphics', 'historical', 'space', 'sport', 'technologie']


def read_data(categories):
    collection = []
    for category in categories:
        for i in range(1, 101):
            filename = f'./archive_2/{category}/{category}_{i}.txt'
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        collection.append({'category': category, 'text': line})
    return collection


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def remove_stopwords(text):
        tokens = word_tokenize(text)
        filtered_tokens = [word.lower()
                           for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)

    def stem_text(text):
        tokens = word_tokenize(text)
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed_tokens)

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"RT ", " ", text)
    text = re.sub(r"@[\w]*", " ", text)
    text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = remove_stopwords(text)
    text = stem_text(text)
    return text


def build_inverted_index(df):
    inverted_index = {}
    for index, row in df.iterrows():
        doc_id = index + 1
        words = row['text'].split()
        for term in words:
            if term not in inverted_index:
                inverted_index[term] = {}
            if doc_id not in inverted_index[term]:
                inverted_index[term][doc_id] = 0
            inverted_index[term][doc_id] += 1
    return inverted_index


def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"RT ", " ", text)
    text = re.sub(r"@[\w]*", " ", text)
    text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def index_documents(df):
    df['processed_text'] = df['text'].apply(preprocess_text)
    df['processed_text'] = df['processed_text'].apply(clean_text)
    df['docno'] = range(1, len(df)+1)
    df['docno'] = df['docno'].apply(str)
    return df


def search_index(index, query):
    stemmer = PorterStemmer()
    terms = query.split()
    stemmed_terms = [stemmer.stem(term) for term in terms]
    doc_ids = []  # Initialize an empty list for doc_ids
    for term in stemmed_terms:
        try:
            pointer = index.getLexicon()[term]
            posting_list = index.getInvertedIndex().getPostings(pointer)
            if posting_list is not None:
                # Use list comprehension
                term_doc_ids = [posting.getDocId() for posting in posting_list]
                if not doc_ids:  # If doc_ids is empty, assign term_doc_ids directly
                    doc_ids = term_doc_ids
                else:
                    # Update doc_ids by taking intersection
                    doc_ids = list(set(doc_ids).intersection(term_doc_ids))
            else:
                print("Posting list for term '%s' is empty" % term)
        except KeyError:
            print("Term '%s' not found in the index" % term)
    return doc_ids


def rank_documents(index, doc_ids, query):
    scores = {}
    stemmer = PorterStemmer()
    terms = query.split()
    stemmed_terms = [stemmer.stem(term) for term in terms]
    for doc_id in doc_ids:
        score = 0
        for term in stemmed_terms:
            try:
                pointer = index.getLexicon()[term]
                posting_list = index.getInvertedIndex().getPostings(pointer)
                if posting_list is not None:
                    for posting in posting_list:
                        if posting.getDocId() == doc_id:
                            score += posting.getFrequency()
                else:
                    print("Posting list for term '%s' is empty" % term)
            except KeyError:
                print("Term '%s' not found in the index" % term)
        scores[doc_id] = score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


def display_document_index(index, docid):
    di = index.getDirectIndex()
    doi = index.getDocumentIndex()
    lex = index.getLexicon()
    doc_entry = doi.getDocumentEntry(docid)
    posting_list = di.getPostings(doc_entry)
    for posting in posting_list:
        termid = posting.getId()
        lex_entry = lex.getLexiconEntry(termid)
        print(lex_entry.getKey() + " -> " + str(posting) +
              " doclen=%d" % posting.getDocumentLength())


def main():
    collection = read_data(categories)
    df = pd.DataFrame(collection)
    df.to_csv('text.csv', index=False)
    df = pd.read_csv('./text.csv')

    df = index_documents(df)

    inverted_index = build_inverted_index(df)

    print(df)

    print(inverted_index)

    indexer = pt.DFIndexer(
        'C:\\Users\\midoh\\Desktop\\UST-CSAI\\Y2S2\\DSAI 201 (Mining & IR)\\Project\\myFirstIndex', overwrite=True)
    index_ref = indexer.index(df["processed_text"], df["docno"])
    index = pt.IndexFactory.of(index_ref)

    query = input("Enter your query: ")

    doc_ids = search_index(index, query)

    ranked_documents = rank_documents(index, doc_ids, query)

    print("Ranked Documents:")
    for doc_id, score in ranked_documents:
        print(f"Document ID: {doc_id}, Score: {score}")

    docid = 10
    display_document_index(index, docid)


if __name__ == "__main__":
    main()
