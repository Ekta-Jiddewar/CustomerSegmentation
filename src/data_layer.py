import pandas

import random

import utilities

MAX_ROWS = 1000000


def load():
    df_raw = pandas.read_csv("../data/raw/train_ver2.csv", nrows=MAX_ROWS)
    columns = ['sexo', 'antiguedad', 'age', 'renta'] + list(df_raw.columns[24:])
    # TODO remove n
    df_to_segment = df_raw.sample(n=10000, random_state=0)[columns]

    productwise_fav = round(100 * df_to_segment.iloc[:, 4:].sum() / df_to_segment.shape[0], 2)
    uncheced_interested_products = list(productwise_fav[productwise_fav > 8].keys())
    df_to_segment = df_to_segment[['sexo', 'antiguedad', 'age'] + uncheced_interested_products]
    interested_products = []

    # drop products with missing > 20%
    for p in uncheced_interested_products:
        if 100*df_to_segment[p].isna().sum()/df_to_segment.shape[0] < 5:
            interested_products.append(p)
            df_to_segment[p] = df_to_segment[p].fillna(random.choice([0, 1]))

    return df_to_segment


def clean(df_to_segment):
    df_to_segment['sexo'] = df_to_segment['sexo'].fillna(random.choice(['V', 'H']))
    # Cleaning antiguedad
    df_to_segment = utilities.impute_median(df_to_segment, 'antiguedad')
    # Cleaning age
    df_to_segment = utilities.impute_median(df_to_segment, 'age')
    return df_to_segment


def process(df_to_segment):
    # making sex column into numerics
    df_to_segment['sexo'] = pandas.get_dummies(df_to_segment['sexo'], drop_first=True).astype(int)
    # return normalized dataframe
    return (df_to_segment - df_to_segment.mean()) / df_to_segment.std()
