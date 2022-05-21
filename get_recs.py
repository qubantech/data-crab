from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import json
from urllib.request import urlopen
import requests, json, os, sys, time, re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import warnings
import sklearn

warnings.filterwarnings("ignore")


def run_jsonifier(df):
    # convert index values to string (when they're something else)
    df.index = df.index.map(str)
    # convert column names to string (when they're something else)
    df.columns = df.columns.map(str)

    # convert DataFrame to dict and dict to string
    js = str(df.to_dict())
    # store indices of double quote marks in string for later update
    idx = [i for i, _ in enumerate(js) if _ == '"']
    # jsonify quotes from single to double quotes
    js = js.replace("'", '"')
    # add \ to original double quotes to make it json-like escape sequence
    for add, i in enumerate(idx):
        js = js[:i + add] + '\\' + js[i + add:]
    return js


def main():
    goods_df = pd.read_csv("goods.csv")
    goods_name_list = goods_df["name"]

    # russian_stopwords = stopwords.words("russian")

    # tfidf_vectorizer = TfidfVectorizer(
    #     strip_accents="unicode", stop_words=russian_stopwords
    # )

    tfidf_vectorizer = TfidfVectorizer(strip_accents="unicode")

    tfidf = tfidf_vectorizer.fit_transform(list(goods_df["description"]))
    tfidf_vectorizer.get_feature_names()

    dic_recommended_1 = {}
    for index in range(goods_df["description"].shape[0]):
        similarities_1 = linear_kernel(tfidf[index], tfidf).flatten()
        related_docs_indices_1 = (-similarities_1).argsort()[:10]

        dic_recommended_1.update(
            {
                goods_name_list[index]: [
                    goods_name_list[i] for i in related_docs_indices_1
                ]
            }
        )
    df_content_based_results_1 = pd.DataFrame(dic_recommended_1)
    df_content_based_results_1.reset_index(inplace=True)

    json_recs = run_jsonifier(df_content_based_results_1[1:4])
    # print(df_content_based_results_1[1:4].T.head())
    # print('\n')
    # print(json_recs)
    return json_recs

print(main())
