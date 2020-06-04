add_train为向训练集train.txt中添加训练文本的程序，在add.py中生成完毕后，需要手动识别ORG,LOC与PER才能放入train.txt中.

在NER中，首次使用请训练模型(data中有训练好的文件，所以可跳过此步骤直接识别)
```
python train.py
```

识别文本
```
python predit.py
```

添加或修改需要识别的文本在predit.py **61**行
```
#predit.py-61
# 需要识别的语句s
s = "放入需要识别的文本"
```