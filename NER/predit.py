# -*- coding: utf-8 -*-
import pickle
import nltk
import numpy as np
from nltk.corpus import wordnet

from utils import CONSTANTS
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk import word_tokenize


def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatization(sentence):#词性还原
    tokens = word_tokenize(sentence)  # 分词
    tagged_sent = nltk.pos_tag(tokens)  # 获取单词词性

    wnl = nltk.WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原

    str = ""

    for i in range(len(lemmas_sent)):
        if i + 1 < len(lemmas_sent) and lemmas_sent[i + 1] != ',' and lemmas_sent[i + 1] != '.' and lemmas_sent[i + 1] != '!' and lemmas_sent[i + 1] != '?':
            str = str + lemmas_sent[i] + " "
        else:
            str = str + lemmas_sent[i]

    return str


with open(CONSTANTS[1], 'rb') as f:
    word_dictionary = pickle.load(f)
with open(CONSTANTS[4], 'rb') as f:
    output_dictionary = pickle.load(f)

lstm_model = load_model(CONSTANTS[0])
input_shape = 60
# 需要识别的语句s
s = "In Los Angeles, thousands of protesters took to the streets of Hollywood on Tuesday in the area's largest demonstration of the day, and many remained on the streets even after the curfew took effect in the evening."
gea = "false"  # 句子中有不认识的词
lemmatization_mode = "false"  # 是否需要词性还原，默认不需要
print(len(s))
res = []
while gea == "false":

    try:
        if lemmatization_mode == "true":
            s = lemmatization(s)

        ss = word_tokenize(s)
        new_x = [[word_dictionary[word] for word in ss]]
        x = pad_sequences(maxlen=input_shape, sequences=new_x, padding='post', value=0)

        words = splitSentence(s)

        for i in range(len(words)):
            print(words[i])
            ss = word_tokenize(words[i])
            new_x = [[word_dictionary[word] for word in ss]]
            x = pad_sequences(maxlen=input_shape, sequences=new_x, padding='post', value=0)
            y_predict = lstm_model.predict(x)

            ner_tag = []
            for i in range(0, len(ss)):
                ner_tag.append(np.argmax(y_predict[0][i]))

            ner = [output_dictionary[i] for i in ner_tag]
            print(ss)
            print(ner)

            ner_reg_list = []
            for word, tag in zip(ss, ner):
                if tag != 'O':
                    ner_reg_list.append((word, tag))

            # 输出结果
            print("识别结果：")
            if ner_reg_list:
                for i, item in enumerate(ner_reg_list):
                    if item[1].startswith('B'):
                        end = i + 1
                        while end <= len(ner_reg_list) - 1 and ner_reg_list[end][1].startswith('I'):
                            end += 1

                        ner_type = item[1].split('-')[1]
                        ner_type_dict = {'PER': 'PERSON: ',
                                         'LOC': 'LOCATION: ',
                                         'ORG': 'ORGANIZATION: ',
                                         'MISC': 'MISC: '
                                         }
                        print(ner_type_dict[ner_type], \
                              ' '.join([item[0] for item in ner_reg_list[i:end]]))

                        str = ner_type_dict[ner_type] + ' '.join([item[0] for item in ner_reg_list[i:end]])
                        # str=ner_type_dict[ner_type]+([item[0] for item in ner_reg_list[i:end]])
                        res.append(str)

            else:
                print("模型并未识别任何有效命名实体。")

        gea = "true"
    except KeyError as err:
        temp = str(err)
        temp = temp.replace("'", "")
        # print(temp)
        print("不在词汇表中的单词为：%s." % err)
        s = s.replace(temp, "")
        gea = "false"

print("完成预测，结果为：")
set = set(res)
res = list(set)
print(res)
