# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from pandas import Series
import re
from gensim import corpora
from gensim import similarities
import collections
import random
import time
import logging
import redis
import MySQLdb as mdb
import datetime
import os


def parse_log(address):
    result = {}
    for line in open(address):
        temp = line.strip().split(',')
        try:
            userId = int(temp[0])
            if int(temp[-1]) > 0:
                bookId = int(temp[-1])
            else:
                continue
        except:
            continue
        if result.has_key(userId):
            result[userId].add(bookId)
        else:
            result[userId] = {bookId}
    return result


def analyse_log():
    logger = loggers()
    now = datetime.datetime.now()
    delta = datetime.timedelta(days=1)
    n_days = now - delta
    address = r'/home/docker/logs/' + n_days.strftime('%Y/%m/%d') + '/book.log'
    if os.path.isfile(address):
        df = pd.read_table(address, sep=',', header=None, usecols=[0, 5], names=['userId', 'bookId'], na_values=[0])
        df.dropna(inplace=True)
        try:
            temp = df.astype(int)
            result = temp.groupby('userId')['bookId'].apply(lambda x: set(x))
        except:
            # 打印日志
            print 'error in log,use parse_log to analyze'
            address = location
            result = parse_log(address)
        result = Series(result).apply(lambda x: list(x) if len(x) < 20 else random.sample(list(x), 20))
    else:
        logger.error('the book.log file dose not exit!please check %s' % address)
        print '-' * 10 + 'error! the file dose not exit!please check!' + '-' * 10
        result = 0
    return result



def convert_doc_to_wordlist(str_doc, cut_all=True):
    if str_doc is np.nan:
        return u''
    sent_list = [word for word in jieba.cut(str_doc, cut_all) if len(word) >= 2 and word.isdigit() == False]
    return u' '.join(sent_list)


def prepro_author(x):
    if x is np.nan:
        return u''
    return (u' '.join(x.lower().replace(u' ', u'').split(',')))


def apply_argv(user_collect, documents):
    result = u''
    for book_id in user_collect:
        if book_id in documents.index:
            result = result + u' ' + documents[book_id]
        result = result + u''
    return result


def processing():
    try:
        db = mdb.connect('mysql.communion.net.cn', 'root', 'yuhai', 'hainan_bigdata', charset='utf8')
    except:
        print 'Mysql connect Error!please check'
        return 0
    # 筛选用户收藏夹图书列表
    base_user = pd.read_sql(
        'SELECT userId,GROUP_CONCAT(bookId) as book_col  from tbl_book_collect GROUP BY userId HAVING COUNT(bookId)>15',
        index_col='userId', con=db)
    base_user = base_user.book_col.apply(lambda x: list(eval(x)))
    print '-' * 10 + 'base_user_tbl finish!' + '-' * 10
    # 筛选图书标签列表
    base_tags_mysql = pd.read_sql(
        'select a.bookId,b.tag,a.num from tbl_book_tag a,tbl_tag b WHERE a.tagId=b.id  ORDER BY a.bookId', con=db)
    temp = base_tags_mysql[['bookId', 'num']].groupby('bookId', sort=False)['num'].apply(lambda x: np.array(x)).apply(
        lambda b: np.ceil((b + 0.1 - b.min()) * 4 / (b.max() + 0.1 - b.min())).astype(int))
    temp2 = base_tags_mysql[['bookId', 'tag']].groupby('bookId', sort=False)['tag'].apply(lambda x: np.array(x))
    temp3 = (temp2 + ' ') * temp
    base_tags = temp3.apply(lambda x: ' '.join(x.tolist()))
    base_tags.dropna(inplace=True)
    print '-' * 10 + 'base_tags finish!' + '-' * 10
    # 筛选图书详情列表
    base_books = pd.read_sql('select id,title,author,rating from tbl_book', con=db, index_col='id')
    base_books['tags'] = base_tags_mysql.groupby('bookId', sort=False)['tag'].apply(lambda x: ' '.join(x))
    print '-' * 10 + 'base_books finish!' + '-' * 10
    # 筛选图书文档
    author = base_books.author.apply(prepro_author)
    documents = base_tags + u' ' + author + u' ' + author
    documents.dropna(inplace=True)
    print '-' * 10 + 'documents finish!' + '-' * 10
    # 筛选收藏夹图书向量
    UsersCollect = base_user.apply(lambda x: apply_argv(x, documents).strip().replace(u'  ', u' '))
    print '-' * 10 + 'UsersCollect finish!' + '-' * 10
    return base_books, documents, base_user, UsersCollect


def k_means(UsersCollect, clusters_sum=10):
    vectorizer = CountVectorizer(min_df=10)
    count = vectorizer.fit_transform(UsersCollect)
    print '-' * 10 + 'vectorizer finish!' + '-' * 10
    print '-' * 10 + 'start k_means,the clusters_sum is:%s' % clusters_sum + '-' * 10
    km = KMeans(n_clusters=clusters_sum, n_init=1, verbose=1, init='k-means++')
    km.fit(count)
    return km, vectorizer


def all_cluster_simility(clusters_number, UsersCollect, labels):
    cluster_dict = {}
    cluster_dict_vector = {}
    for clus_num in clusters_number:
        userid_in_clusters = UsersCollect.index[labels == clus_num].values
        temp = UsersCollect.loc[userid_in_clusters].apply(lambda x: re.split(r'\s+', x))
        dictionary_vector = corpora.Dictionary(temp)
        vector = [dictionary_vector.doc2bow(text) for text in temp]
        sim_index = similarities.SparseMatrixSimilarity(vector, num_features=len(dictionary_vector))
        cluster_dict[clus_num] = sim_index
        cluster_dict_vector[clus_num] = dictionary_vector
    return cluster_dict, cluster_dict_vector


def hot_books(base_user, limit_num=30):
    # 用户收藏次数最多的图书
    all_bookids_count = collections.defaultdict(int)
    for books in base_user:
        for bookid in books:
            all_bookids_count[bookid] += 1
    all_bookids_count_list = sorted(all_bookids_count.items(), key=lambda x: x[1], reverse=True)
    hot_bookid = map(lambda x: x[0], all_bookids_count_list[:limit_num])
    print '-' * 10 + 'hot_books finish and the limit_num is: %s' % limit_num + '-' * 10
    return hot_bookid


def recommend_User_Book(index_bookid, base_user, vectorizer, documents, km, UsersCollect, labels, cluster_dict,
                        cluster_dict_vector, hot_bookid):
    doc_index = documents.index
    index_bookid = [i for i in index_bookid if i in doc_index]
    index_user_collect = vectorizer.transform(Series(u' '.join(documents.loc[index_bookid])))
    index_predict = km.predict(index_user_collect.astype('float'))[0]
    userid_in_clusters = UsersCollect.index[labels == index_predict].values
    vec_index = cluster_dict_vector[index_predict].doc2bow(re.split(r'\s+', u' '.join(documents.loc[index_bookid])))
    recommend_userid = userid_in_clusters[
        cluster_dict[index_predict][vec_index].argsort()[::-1][:int(0.1 * len(userid_in_clusters))]]
    recommend_values = cluster_dict[index_predict][vec_index][
        cluster_dict[index_predict][vec_index].argsort()[::-1][:int(0.1 * len(userid_in_clusters))]]
    recommend_userid_values = zip(recommend_userid, recommend_values)
    recommend_bookids = collections.defaultdict(float)
    recommend_bookids_count = collections.defaultdict(int)
    for userid, score in recommend_userid_values:
        for book_id in base_user.loc[userid]:
            recommend_bookids_count[book_id] += 1
            recommend_bookids[book_id] += score * (1.0 / (np.log(recommend_bookids_count[book_id] + 1)))
    result_book = sorted(recommend_bookids.items(), key=lambda x: x[1], reverse=True)
    temp = [i for i, v in result_book[:100] if i not in hot_bookid] + random.sample(hot_bookid, 5)
    Rec_Book_temp = temp
    Rec_User = u','.join(map(lambda x: str(x), recommend_userid[:40]))
    return Rec_Book_temp, Rec_User

def loggers():
    # creat logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('log_analyze')
    # cret handler and add to log.txt
    handler = logging.FileHandler('LogAnalyze_log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def redis_con(logger):
    try:
        r = redis.Redis(host='redis.communion.net.cn', port=6379, password='redistest')
    except:
        logger.error('error occur when connect to redis.sleep 60s and try again!')
        time.sleep(60)
        r = redis.Redis(host='redis.communion.net.cn', port=6379, password='redistest')
    return r


def main():
    logger = loggers()
    base_books, documents, base_user, UsersCollect = processing()
    logger.info('data processing finish!')
    km, vectorizer = k_means(UsersCollect)
    logger.info('kmeans finish!')
    labels = km.labels_
    clusters_number = Series(labels).value_counts()[:10].index  # 前10个最大的簇计算簇内相似度
    cluster_dict, cluster_dict_vector = all_cluster_simility(clusters_number, UsersCollect, labels)
    logger.info('cluster_dict,cluster_dict_vector finish!')
    hot_bookid = hot_books(base_user)

    default_hot_bookId = u','.join(map(lambda x: str(x), hot_bookid))
    default_userId = u'4,5,7,9,10,14,16,17,19,20'

    r = redis_con(logger)
    print '-' * 10 + 'ready for compute and insert to redis' + '-' * 10

    result = analyse_log()
    for uid in result.index:
        index_bookid = result[uid]
        try:
            temp2 = base_user, vectorizer, documents, km, UsersCollect, labels, cluster_dict, cluster_dict_vector, hot_bookid
            Rec_Book_temp, Rec_User = recommend_User_Book(index_bookid, *temp2)
            Rec_Book = u','.join(map(lambda x: str(x), [i for i in Rec_Book_temp]))
        except:
            Rec_Book, Rec_User = (default_hot_bookId, default_userId)
            logger.error('Error to compute Rec_Book,the index_bookId is:%s,userId is: %s' % (index_bookid, uid))
        try:
            r.set('b_like:%s' % uid, Rec_Book)
            r.set('u_similar:%s' % uid, Rec_User)
            print '-' * 10 + 'insert into redis--b_like,the userId is: %s' % uid + '-' * 10
        except:
            logger.error(
                'enable to set data to redis,please check.sleep 30s and try again! uid is: %s,Rec_book is: %s' % (
                uid, Rec_Book))
            time.sleep(30)
            try:
                r = redis_con(logger)
                r.set('b_like:%s' % uid, Rec_Book)
                r.set('u_similar:%s' % uid, Rec_User)
                logger.info('redis connect problem fixed.')
                print '-' * 10 + 'insert into redis--b_like,the userId is: %s' % uid + '-' * 10
            except:
                logger.error('the redis connect problem is still not fixed.please check' % uid)
                print '-' * 10 + 'error occur when insert into redis--b_like,the userId is: %s' % uid + '-' * 10
    print '-' * 10 + 'yeah! all user log analyze finished!' + '-' * 10

if __name__ == '__main__':
    main()