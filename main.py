import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import pyterrier as pt
import nltk
import pandas as pd
import os
import torch
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from transformers import AutoTokenizer, AutoModel

if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

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


def build_inverted_index(df):
    inverted_index = {}
    df2 = index_documents(df)
    for index, row in df2.iterrows():
        doc_id = index + 1
        words = row['text'].split()

        for term in words:
            if term not in inverted_index:
                inverted_index[term] = {}
            if doc_id not in inverted_index[term]:
                inverted_index[term][doc_id] = 0
            inverted_index[term][doc_id] += 1
    return inverted_index


def search_index(index, query):
    lexicon = index.getLexicon()
    metadata = index.getMetaIndex()
    inverted = index.getInvertedIndex()
    lex = lexicon.getLexiconEntry(query)

    if lex is None:
        return []

    postings = inverted.getPostings(lex)

    if postings is None:
        return []

    ids = [metadata.getItem("docno", posting.getId())
           for posting in postings]
    return ids


def rank_tfidf(index, query):
    tfidf_retr = pt.BatchRetrieve(
        index, controls={"wmodel": "TF_IDF"})
    return tfidf_retr.search(query)


def display_document_index(index, docid):
    di = index.getDirectIndex()
    doi = index.getDocumentIndex()
    lex = index.getLexicon()
    for i in range(len(docid)):
        doc_entry = doi.getDocumentEntry(docid[i])
        posting_list = di.getPostings(doc_entry)
        for posting in posting_list:
            termid = posting.getId()
            lex_entry = lex.getLexiconEntry(termid)
            print(lex_entry.getKey() + " -> " + str(posting) +
                  " doclen=%d" % posting.getDocumentLength())


def get_embeddings(text):
    tokens = encode(text)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    output = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    return output


def encode(text, max_length=32):
    return bert_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )


def search():
    query = query_entry.get()
    if not query:
        messagebox.showerror("Error", "Please enter a query.")
        return
    for term in query.split():
        inv_indx = inverted_index[f'{term}']
        if inv_indx is None:
            result_text.insert(
                tk.END, f'Term "{term}" not found in the index\n')
        else:
            result_text.insert(tk.END, f'Term "{term}" found in the index\n')
    inv_indx = list(inv_indx.keys())
    docid = inv_indx
    display_document_index(index, docid)
    ranked_documents = rank_tfidf(index, query)
    docid = list(ranked_documents['docid'])
    score = list(ranked_documents['score'])
    df3 = pd.DataFrame(list(zip(docid, score)),
                       columns=['Document ID', 'Score'])
    result_text.insert(tk.END, "Ranked Documents:\n")
    result_text.insert(tk.END, df3.to_string(index=False) + "\n\n")
    query_processed = preprocess_text(query)
    query_processed = clean_text(query_processed)
    bm25 = pt.BatchRetrieve(index, controls={"wmodel": "BM25"}, num_results=10)
    bm25_res = bm25.search(query_processed)
    rm3_expander2 = pt.rewrite.RM3(index, fb_terms=10, fb_docs=100)
    rm3_qe = bm25 >> rm3_expander2
    expanded_query = rm3_qe.search(query_processed).iloc[0]["query"]
    expanded_query_formatted = ' '.join(expanded_query.split()[1:])
    results_wqe = bm25.search(expanded_query_formatted)
    result_text.insert(tk.END, "   Before Expansion    After Expansion\n")
    result_text.insert(tk.END, pd.concat([bm25_res[['docid', 'score']][0:5].add_suffix('_1'),
                                          results_wqe[['docid', 'score']][0:5].add_suffix('_2')], axis=1).fillna(''))
    result_text.insert(tk.END, "\n")
    result_text.insert(tk.END, df['text'][df['docno'].isin(
        results_wqe['docno'].loc[0:5].tolist())])
    result_text.insert(tk.END, "\n")
    for term in query.split():
        text = f"{term}"
        embeddings = get_embeddings(text)
        result_text.insert(tk.END, f"Term : {term}\n")
        result_text.insert(tk.END, f"Embedding shape : {
                           embeddings[0].shape}\n")
        result_text.insert(tk.END, f"Embedding : {embeddings[0]}\n\n")


collection = read_data(categories)
df = pd.DataFrame(collection)
df.to_csv('text.csv', index=False)
df = pd.read_csv('./text.csv')
df2 = index_documents(df)
inverted_index = build_inverted_index(df)
indexer = pt.DFIndexer(
    'C:\\Users\\midoh\\Desktop\\UST-CSAI\\Y2S2\\DSAI 201 (Mining & IR)\\Project\\mySecondIndex', overwrite=True)
index_ref = indexer.index(df["processed_text"], df["docno"])
index = pt.IndexFactory.of(index_ref)

model_name = "bert-base-uncased"
device = torch.device("cpu")
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

root = tk.Tk()
root.title("IR System")

query_label = tk.Label(root, text="Enter your query:")
query_label.pack()
query_entry = tk.Entry(root, width=50)
query_entry.pack()

search_button = tk.Button(root, text="Search", command=search)
search_button.pack()

result_text = scrolledtext.ScrolledText(root, width=100, height=30)
result_text.pack()

root.mainloop()
