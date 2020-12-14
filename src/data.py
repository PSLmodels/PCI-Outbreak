import pandas as pd
import numpy as np
import sqlite3
import os, re

from src.aws_s3 import *


def cut_sent(para):
    para = re.sub('([。！？\? ])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\? ])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.strip()
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sent_list = para.split("\n")
    return [sent.strip() for sent in sent_list]


def cut_semi_sent(para):
    """
    Semi sentences are separated using the separators in cut_sent as well as semi-colons and comas.
    """
    para = re.sub('([；;：:，,。！!？\? ])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！!？\?][”’])([^；;，,。！!？\? ])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.strip()
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sent_list = para.split("\n")
    sent_list = [re.sub('；$|;$|：$|:$|，$|,$|。$|！$|!$|？$|\?$|\.{6}$|…{2}$', '', sent) for sent in sent_list]
    return [sent.strip() for sent in sent_list]

    
def read_data_from_sql(input_path, keywords, from_date, to_date, min_ratio, min_count, foreign_countries, foreign_min_count):
    conn = sqlite3.connect(input_path)
    df = pd.read_sql_query("select title, body, date, id from main where date >= Datetime('"+str(from_date)+"') and date <= Datetime('"+str(to_date)+"')", conn)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    conn.close()

    df['text'] = df['title'] + " " + df['body']

    df['text_count'] = df['text'].str.count('|'.join(keywords))
    df['text_len'] = df['text'].str.len()
    df['text_ratio'] = df['text_count']/df['text_len']
    df['relevance'] = ((df['text_ratio']>=min_ratio) & (df['text_count']>=min_count)) + 0

    df['foreign_count'] = df['text'].str.count('|'.join(foreign_countries))
    df['foreign'] = (df['foreign_count']>=foreign_min_count) + 0

    df_relevant = df.loc[df['relevance']==1]
    # df_ambiguous = df.loc[(df['relevance']==0) & (df['text_count']>0)]
    df_irrelevant = df.loc[df['text_count']==0]

    return df_relevant, df_irrelevant


def split_to_sentences(df, use_titles=False):
    """
    use_titles: whether to include titles in the sentence-level data
    """
    df_title = df
    df_title['sentence'] = df_title['title'].apply(lambda x: cut_sent(x))
    lens = [len(item) for item in df_title['sentence']]

    df_out_title = pd.DataFrame({"date" : np.repeat(df_title['date'].values,lens),
                                 "relevance" : np.repeat(df_title['relevance'].values,lens),
                                 "foreign" : np.repeat(df_title['foreign'].values,lens),
                                 "article_id" : np.repeat(df_title['id'].values,lens),
                                 "sentence" : np.hstack(df_title['sentence'])
                                })
    df_out_title['title_indicator'] = 1

    df_body = df
    df_body['sentence'] = df_body['body'].apply(lambda x: cut_sent(x))
    lens = [len(item) for item in df_body['sentence']]

    df_out_body = pd.DataFrame({"date" : np.repeat(df_body['date'].values,lens),
                                "relevance" : np.repeat(df_body['relevance'].values,lens),
                                "foreign" : np.repeat(df_body['foreign'].values,lens),
                                "article_id" : np.repeat(df_body['id'].values,lens),
                                "sentence" : np.hstack(df_body['sentence'])
                               })
    df_out_body['title_indicator'] = 0

    if use_titles:
        df_out = pd.concat([df_out_title, df_out_body])
    else:
        df_out = df_out_body

    df_out["sentence_id"] = df_out.groupby("article_id")["relevance"].transform(lambda x: np.arange(1, len(x)+1))
    
    df_out["unique_id"] = np.arange(1, len(df_out)+1)

    df_out = df_out.sort_values(by=['date', 'article_id', 'sentence_id'])

    return df_out


def split_to_semi_sentences(df, use_titles=False):
    """
    use_titles: whether to include titles in the sentence-level data
    """
    df_title = df
    df_title['sentence'] = df_title['title'].apply(lambda x: cut_semi_sent(x))
    lens = [len(item) for item in df_title['sentence']]

    df_out_title = pd.DataFrame({"date" : np.repeat(df_title['date'].values,lens),
                                 "relevance" : np.repeat(df_title['relevance'].values,lens),
                                 "foreign" : np.repeat(df_title['foreign'].values,lens),
                                 "article_id" : np.repeat(df_title['id'].values,lens),
                                 "sentence" : np.hstack(df_title['sentence'])
                                })
    df_out_title['title_indicator'] = 1

    df_body = df
    df_body['sentence'] = df_body['body'].apply(lambda x: cut_semi_sent(x))
    lens = [len(item) for item in df_body['sentence']]

    df_out_body = pd.DataFrame({"date" : np.repeat(df_body['date'].values,lens),
                                "relevance" : np.repeat(df_body['relevance'].values,lens),
                                "foreign" : np.repeat(df_body['foreign'].values,lens),
                                "article_id" : np.repeat(df_body['id'].values,lens),
                                "sentence" : np.hstack(df_body['sentence'])
                               })
    df_out_body['title_indicator'] = 0

    if use_titles:
        df_out = pd.concat([df_out_title, df_out_body])
    else:
        df_out = df_out_body

    df_out["sentence_id"] = df_out.groupby("article_id")["relevance"].transform(lambda x: np.arange(1, len(x)+1))
    
    df_out["unique_id"] = np.arange(1, len(df_out)+1)

    df_out = df_out.sort_values(by=['date', 'article_id', 'sentence_id'])

    return df_out


def process_data(model_settings):

    old_list = []
    old_irrel_list = []
    new_list = []
    new_irrel_list = []

    for file in gen_raw_data_path(model_settings):
        df_old_i, df_old_irrel_i = read_data_from_sql(input_path=file, keywords=model_settings['keywords_old'],
            from_date=model_settings['from_date_old'], to_date=model_settings['to_date_old'],
            min_ratio=model_settings['min_ratio_old'], min_count=model_settings['min_count_old'],
            foreign_countries=model_settings['foreign_countries'], foreign_min_count=model_settings['foreign_min_count'])
        old_list.append(df_old_i)
        old_irrel_list.append(df_old_irrel_i)

        df_new_i, df_new_irrel_i = read_data_from_sql(input_path=file, keywords=model_settings['keywords_new'],
            from_date=model_settings['from_date_new'], to_date=model_settings['to_date_new'],
            min_ratio=model_settings['min_ratio_new'], min_count=model_settings['min_count_new'],
            foreign_countries=model_settings['foreign_countries'], foreign_min_count=model_settings['foreign_min_count'])
        new_list.append(df_new_i)
        new_irrel_list.append(df_new_irrel_i)

    df_old = pd.concat(old_list)
    df_old_sentences = split_to_sentences(df_old, use_titles=True)
    # df_old_semi_sentences = split_to_semi_sentences(df_old, use_titles=True)
    
    df_old_irrel = pd.concat(old_irrel_list)
    df_old_irrel_sentences = split_to_sentences(df_old_irrel, use_titles=True)
    # df_old_irrel_semi_sentences = split_to_semi_sentences(df_old_irrel, use_titles=True)
    
    df_new = pd.concat(new_list)
    df_new_sentences = split_to_sentences(df_new, use_titles=True)
    # df_new_semi_sentences = split_to_semi_sentences(df_new, use_titles=True)
    
    df_new_irrel = pd.concat(new_irrel_list)
    df_new_irrel_sentences = split_to_sentences(df_new_irrel, use_titles=True)
    # df_new_irrel_semi_sentences = split_to_semi_sentences(df_new_irrel, use_titles=True)

    df_old.to_pickle(gen_data_path(model_settings, 'SARS_articles.pickle'))
    df_old_sentences.to_pickle(gen_data_path(model_settings, 'SARS_sentences.pickle'))
    df_old_irrel.to_pickle(gen_data_path(model_settings, 'SARS_irrel_articles.pickle'))
    df_old_irrel_sentences.to_pickle(gen_data_path(model_settings, 'SARS_irrel_sentences.pickle'))

    df_new.to_pickle(gen_data_path(model_settings, 'COVID_articles.pickle'))
    df_new_sentences.to_pickle(gen_data_path(model_settings, 'COVID_sentences.pickle'))
    df_new_irrel.to_pickle(gen_data_path(model_settings, 'COVID_irrel_articles.pickle'))
    df_new_irrel_sentences.to_pickle(gen_data_path(model_settings, 'COVID_irrel_sentences.pickle'))

    # df_old_semi_sentences.to_pickle(gen_data_path(model_settings, 'SARS_semi_sentences.pickle'))
    # df_old_irrel_semi_sentences.to_pickle(gen_data_path(model_settings, 'SARS_irrel_semi_sentences.pickle')
    # df_new_semi_sentences.to_pickle(gen_data_path(model_settings, 'COVID_semi_sentences.pickle')
    # df_new_irrel_semi_sentences.to_pickle(gen_data_path(model_settings, 'COVID_irrel_semi_sentences.pickle')

    print("Finished processing data.")


def gen_sum_stats(model_settings):

    stats_sars_relevant = pd.read_pickle(gen_data_path(model_settings, 'SARS_sentences.pickle'))        
    stats_sars_irrelevant = pd.read_pickle(gen_data_path(model_settings, 'SARS_irrel_sentences.pickle'))
    stats_covid_relevant = pd.read_pickle(gen_data_path(model_settings, 'COVID_sentences.pickle'))

    stats_sars_relevant['text_len'] = stats_sars_relevant['sentence'].str.len()
    stats_sars_irrelevant['text_len'] = stats_sars_irrelevant['sentence'].str.len()
    stats_covid_relevant['text_len'] = stats_covid_relevant['sentence'].str.len()

    sum_stats_data = {'group' : ['sars_relevant', 'sars_irrelevant', 'covid_relevant'],
                      'num_articles': [stats_sars_relevant['article_id'].nunique(),
                                       stats_sars_irrelevant['article_id'].nunique(),
                                       stats_covid_relevant['article_id'].nunique()],
                      'num_words_per_art': [stats_sars_relevant.groupby("article_id")["text_len"].sum().reset_index()['text_len'].mean().round(),
                                            stats_sars_irrelevant.groupby("article_id")["text_len"].sum().reset_index()['text_len'].mean().round(),
                                            stats_covid_relevant.groupby("article_id")["text_len"].sum().reset_index()['text_len'].mean().round()],
                      'num_sent_per_art': [stats_sars_relevant.groupby("article_id")["sentence_id"].max().reset_index()['sentence_id'].mean().round(),
                                           stats_sars_irrelevant.groupby("article_id")["sentence_id"].max().reset_index()['sentence_id'].mean().round(),
                                           stats_covid_relevant.groupby("article_id")["sentence_id"].max().reset_index()['sentence_id'].mean().round()]}

    sum_stats = pd.DataFrame(data=sum_stats_data)

    sum_stats.to_excel(gen_data_path(model_settings, 'sum_stats.xlsx'))

    stats_sars_relevant.groupby('date')['article_id'].nunique().reset_index().to_excel(gen_data_path(model_settings, 'sars_time_series.xlsx'))
    stats_covid_relevant.groupby('date')['article_id'].nunique().reset_index().to_excel(gen_data_path(model_settings, 'covid_time_series.xlsx'))