#训练集添加程序
import nltk

document = "President Xi Jinping said Wednesday that China stands ready to work with Germany and the European Union (" \
           "EU) to strengthen strategic cooperation, uphold multilateralism, tackle global challenges, and jointly " \
           "add certainty to the current world of uncertainty. "
sentences = nltk.sent_tokenize(document)
# for sent in sentences:
#     print(nltk.pos_tag(nltk.word_tokenize(sent)))

data = []
temp=""
for sent in sentences:
    temp=nltk.pos_tag(nltk.word_tokenize(sent))

s1=""
s2=""
s3=""
for i in temp:
    s1=s1+i[0]+"\t"
    s2=s2+i[1]+"\t"
    s3=s3+"O"+"\t"

# print(s1)
# print(s2)
s1=s1+"\n"
s2=s2+"\n"

path = ""     # 创建的txt文件的存放路径
name="识别结果"
full_path = path + name + '.txt'
file = open(full_path, 'w')
file.write(s1+s2+s3)
file.close()
