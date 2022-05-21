import io

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
cyrillic_translit = {'\u0410': 'A', '\u0430': 'a',
                     '\u0411': 'B', '\u0431': 'b',
                     '\u0412': 'V', '\u0432': 'v',
                     '\u0413': 'G', '\u0433': 'g',
                     '\u0414': 'D', '\u0434': 'd',
                     '\u0415': 'E', '\u0435': 'e',
                     '\u0416': 'Zh', '\u0436': 'zh',
                     '\u0417': 'Z', '\u0437': 'z',
                     '\u0418': 'I', '\u0438': 'i',
                     '\u0419': 'I', '\u0439': 'i',
                     '\u041a': 'K', '\u043a': 'k',
                     '\u041b': 'L', '\u043b': 'l',
                     '\u041c': 'M', '\u043c': 'm',
                     '\u041d': 'N', '\u043d': 'n',
                     '\u041e': 'O', '\u043e': 'o',
                     '\u041f': 'P', '\u043f': 'p',
                     '\u0420': 'R', '\u0440': 'r',
                     '\u0421': 'S', '\u0441': 's',
                     '\u0422': 'T', '\u0442': 't',
                     '\u0423': 'U', '\u0443': 'u',
                     '\u0424': 'F', '\u0444': 'f',
                     '\u0425': 'Kh', '\u0445': 'kh',
                     '\u0426': 'Ts', '\u0446': 'ts',
                     '\u0427': 'Ch', '\u0447': 'ch',
                     '\u0428': 'Sh', '\u0448': 'sh',
                     '\u0429': 'Shch', '\u0449': 'shch',
                     '\u042a': '"', '\u044a': '"',
                     '\u042b': 'Y', '\u044b': 'y',
                     '\u042c': "'", '\u044c': "'",
                     '\u042d': 'E', '\u044d': 'e',
                     '\u042e': 'Iu', '\u044e': 'iu',
                     '\u042f': 'Ia', '\u044f': 'ia'
                     }


def transliterate(word, translit_table):
    converted_word = ''
    for char in word:
        transchar = ''
        if char in translit_table:
            transchar = translit_table[char]
        else:
            transchar = char
        converted_word += transchar
    return converted_word


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

    tfidf_vectorizer = TfidfVectorizer(strip_accents="unicode")

    tfidf = tfidf_vectorizer.fit_transform(list(goods_df["description"]))
    tfidf_vectorizer.get_feature_names()

    recommended_dict = {}
    for index in range(goods_df["description"].shape[0]):
        similarities_1 = linear_kernel(tfidf[index], tfidf).flatten()
        related_docs_indices_1 = (-similarities_1).argsort()[:10]

        recommended_dict.update(
            {
                goods_name_list[index]: [
                    goods_name_list[i] for i in related_docs_indices_1
                ]
            }
        )
    df_results = pd.DataFrame(recommended_dict)
    df_results.reset_index(inplace=True)

    df_results = df_results[1:4].T[1:]
    df_results.columns = ["rec1", "rec2", "rec3"]

    result_df = pd.DataFrame(columns=["item", "rec1", "rec2", "rec3"])

    rec1_names = df_results['rec1'].tolist()
    rec2_names = df_results['rec2'].tolist()
    rec3_names = df_results['rec3'].tolist()
    json_good = []
    json_rec1 = []
    json_rec2 = []
    json_rec3 = []

    # print(rec1_names)
    # print(rec2_names)
    # print(rec3_names)
    big_string = "["
    for i in range(0, len(goods_name_list) - 1):
        good = goods_name_list[i]
        rec1 = rec1_names[i]
        rec2 = rec2_names[i]
        rec3 = rec3_names[i]
        good_row = goods_df.loc[goods_df['name'] == good]
        rec1_row = goods_df.loc[goods_df['name'] == rec1]
        rec2_row = goods_df.loc[goods_df['name'] == rec2]
        rec3_row = goods_df.loc[goods_df['name'] == rec3]
        # print(good_row)
        # print(rec1_row)
        # print(rec2_row)
        # print(rec3_row)

        good_name = good_row.iloc[0]['name']
        good_name = json.dumps(good_name, ensure_ascii=False).encode('utf-8')
        good_name = json.loads(good_name)

        ins_dict = {
            "item": {
                "description": good_row.iloc[0]['description'],
                "imageUrl": good_row.iloc[0]['imageUrl'],
                "name": good_row.iloc[0]['name'],
                "price": str(good_row.iloc[0]["price"]),
                "uuid": good_row.iloc[0]["uuid"]
            },
            "rec1": {
                "description": rec1_row.iloc[0]['description'],
                "imageUrl": rec1_row.iloc[0]['imageUrl'],
                "name": rec1_row.iloc[0]['name'],
                "price": str(rec1_row.iloc[0]["price"]),
                "uuid": rec1_row.iloc[0]["uuid"]
            },
            "rec2": {
                "description": rec2_row.iloc[0]['description'],
                "imageUrl": rec2_row.iloc[0]['imageUrl'],
                "name": rec2_row.iloc[0]['name'],
                "price": str(rec2_row.iloc[0]["price"]),
                "uuid": rec2_row.iloc[0]["uuid"]
            },
            "rec3": {
                "description": rec3_row.iloc[0]['description'],
                "imageUrl": rec3_row.iloc[0]['imageUrl'],
                "name": rec3_row.iloc[0]['name'],
                "price": str(rec3_row.iloc[0]["price"]),
                "uuid": rec3_row.iloc[0]["uuid"]
            }
        }
        # print(ins_dict)
        json_object = json.dumps(ins_dict, indent=3).encode('utf-8')
        json_object = json.loads(json_object)
        print(json_object)
        big_string += str(json_object)
        big_string += ','
    big_string += ']'

    # print(big_string)
    # with open('boba.json', 'w') as f:
    #     f.write(big_string)

    return big_string
    # }
    # current_good_obj = run_jsonifier(goods_df.loc[goods_df['name'] == good_name])
    # current_good_list.append(current_good_obj)
    # for good in goods_name_list:
    #     row = goods_df.loc[goods_df['name'] == good]
    #     print(row)
    #     json_good.append({"description": row["description"],
    #                       "imageUrl": row["imageUrl"],
    #                       "name": row["name"],
    #                       "price": row["price"],
    #                       "uuid": row["uuid"]})
    #
    # for rec1 in rec1_names:
    #     row = goods_df.loc[goods_df['name'] == rec1]
    #     json_rec1.append({"description": row["description"],
    #                       "imageUrl": row["imageUrl"],
    #                       "name": row["name"],
    #                       "price": row["price"],
    #                       "uuid": row["uuid"]})
    #
    # for rec2 in rec2_names:
    #     row = goods_df.loc[goods_df['name'] == rec2]
    #     json_rec2.append({"description": row["description"],
    #                       "imageUrl": row["imageUrl"],
    #                       "name": row["name"],
    #                       "price": row["price"],
    #                       "uuid": row["uuid"]})
    # for rec3 in rec3_names:
    #     row = goods_df.loc[goods_df['name'] == rec3]
    #     json_rec3.append({"description": row["description"],
    #                       "imageUrl": row["imageUrl"],
    #                       "name": row["name"],
    #                       "price": row["price"],
    #                       "uuid": row["uuid"]})
    #
    # for i in range(0, len(goods_name_list)):
    #     ins_dict = {"item": json_good[i], "rec1": json_rec1[i], "rec2": json_rec2[i], "rec3": json_rec3[i]}
    #     if i == 1:
    #         print(ins_dict)
    #     result_df.append(ins_dict, ignore_index=True)
    # result_df.to_csv('a.csv')
    # return result_df

    # three_recs_dict = df_content_based_results_1[1:4].T
    # json_recs = three_recs_dict

    # json_recs = run_jsonifier()
    # print(df_content_based_results_1[1:4].T.head())
    # print('\n')
    # print(json_recs)
    # return json_recs


main()
