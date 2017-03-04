# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import jieba
from gensim import corpora
from gensim import similarities
import re
import redis
import MySQLdb as mdb

def convert_doc_to_wordlist(str_doc,cut_all=True):
    if str_doc is np.nan:
        return u''
    sent_list=[word for word in jieba.cut(str_doc,cut_all) if len(word)>=2 and word.isdigit()==False]
    return u' '.join(sent_list)

def prepro_author(x):
    if x is np.nan:
        return u''
    return (u' '.join(x.lower().replace(u' ',u'').split(',')))

def apply_argv(user_collect,documents):
    result = u''
    for book_id in user_collect:
        if book_id in documents.index:
            result = result + u' ' + documents[book_id]
        result = result + u''
    return result

def processing():
    try:
        db = mdb.connect('mysql.communion.net.cn','root','yuhai','hainan_bigdata',charset='utf8')
    except:
        print 'Mysql connect Error!please check'
        return 0
    #筛选图书标签列表
    base_tags_mysql = pd.read_sql('select a.bookId,b.tag,a.num from tbl_book_tag a,tbl_tag b WHERE a.tagId=b.id  ORDER BY a.bookId',con=db)
    temp = base_tags_mysql[['bookId','num']].groupby('bookId',sort=False)['num'].apply(lambda x:np.array(x)).apply(lambda b:np.ceil((b+0.1 - b.min())*4/(b.max()+0.1- b.min())).astype(int))
    temp2 = base_tags_mysql[['bookId','tag']].groupby('bookId',sort=False)['tag'].apply(lambda x:np.array(x))
    temp3 = (temp2 + ' ') * temp
    base_tags = temp3.apply(lambda x:' '.join(x.tolist()))
    base_tags.dropna(inplace=True)
    print '-'*10 + 'base_tags finish!'+'-'*10
    #筛选图书详情列表
    base_books = pd.read_sql('select id,title,author,rating from tbl_book',con=db,index_col='id')
    base_books['tags']=base_tags_mysql.groupby('bookId',sort=False)['tag'].apply(lambda x:' '.join(x))
    base_books.dropna(inplace=True)
    print '-'*10 + 'base_books finish!'+'-'*10
    #筛选图书文档
    author = base_books.author.apply(prepro_author)
    temp_doc = base_tags +u' '+author+u' '+author
    temp_doc.dropna(inplace=True)
    documents = temp_doc.apply(lambda x:re.split(r'\s+',x))
    print '-'*10 + 'documents finish!'+'-'*10
    return base_books,documents

def main():
    base_books, documents = processing()
    print '-' * 10 + 'begin to compute vector Cos_simility' + '-' * 10
    dictionary_vector = corpora.Dictionary(documents)
    vector = [dictionary_vector.doc2bow(text) for text in documents]
    index_vector = similarities.Similarity('douban_sim_books', vector, num_features=len(dictionary_vector),
                                           num_best=100)
    r = redis.Redis(host='redis.communion.net.cn', port=6379, password='redistest')
    print '-' * 10 + 'begin to compute and insert into redis' + '-' * 10
    g = index_vector.__iter__()
    title = base_books.title
    rating = base_books.rating.values
    index = base_books.index
    count = 0
    result = {}
    for values in g:
        try:
            base_title = [re.sub(u'[.·・《》-]', '', re.sub(u'[（）【】/ () 0-9，第:精].*', '', title.iloc[count].lower()))]
        except:
            base_title = [title.iloc[count]]
        recommend_id = []
        rating_list = []
        for recommend in values[1:]:
            try:
                title_replaced = re.sub(u'[.·・《》-]', '',
                                        re.sub(u'[（）【】/ () 0-9，第:精].*', '', title.iloc[recommend[0]].lower()))
            except:
                title_replaced = title.iloc[recommend[0]]
            finally:
                if title_replaced not in base_title:
                    base_title.append(title_replaced)
                    recommend_id.append(index[recommend[0]])
                    rating_list.append(rating[recommend[0]])
        count += 1
        temp = np.array(recommend_id)[np.array(rating_list).argsort()[::-1]].astype(int).tolist()
        result[count] = temp[:15]
        to_redis = ','.join(map(lambda x: str(x), temp[:20]))
        r.set('b_similar:%s' % count, to_redis)  # 插入redis数据库
        if count % 5000 == 0:
            print 'now the total counts is: %s'% count

if __name__ == '__main__':
    main()