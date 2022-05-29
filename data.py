'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2021-01-03 21:33:04
LastEditTime: 2021-01-03 21:37:58
FilePath: /JD_NLP1-text_classfication/data.py
Desciption: Process data.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import config

id2label = {}
with open(config.label_ids_file, 'r', encoding='UTF-8-sig') as txt: # revised with encoding
    for line in txt:
        ID, label = line.strip().split('\t')
        id2label[ID] = label

print(id2label)
for filepath in [config.train_raw_file,config.eval_raw_file,config.test_raw_file]:
    samples = []
    with open(filepath, 'r', encoding='UTF-8-sig') as txt:  # revised with encoding
        for line in txt:
            ID, text = line.strip().split('\t')
            label = id2label[ID]
            sample = label+'\t'+text
            samples.append(sample)

    outfile = config.train_data_file
    if 'eval' in filepath:
        outfile = config.eval_data_file
    if 'test' in filepath:
        outfile = config.test_data_file

    with open(outfile, 'w', encoding='UTF-8-sig') as csv:  # revised with encoding
        csv.write('label\ttext\n')
        for sample in samples:
            csv.write(sample)
            csv.write('\n')
