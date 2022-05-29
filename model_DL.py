import json
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# 数据集读取
class JD_Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

class Classifier:
    def __init__(self, train_mode=False) -> None:
        self.stopWords = [x.strip() for x in open(r'./data/stopwords.txt', 'r', encoding='UTF-8-sig').readlines()]
        self.labelToIndex = json.load(open('./data/label2id.json', encoding='utf-8-sig'))
        self.ix2label = {v: k for k, v in self.labelToIndex.items()}
        self.train = pd.read_csv('./data/train.csv',sep='\t').dropna().reset_index(drop=True)[:]  # revised
        self.dev = pd.read_csv('./data/eval.csv',sep='\t').dropna().reset_index(drop=True)[:]  # revised
        self.test = pd.read_csv('./data/test.csv',sep='\t').dropna().reset_index(drop=True)[:]  # revised

    def process(self):
        print("start of trainer")  # marker

        train = self.train[:]
        y_train = train['label'].map(self.labelToIndex)
        X_train = train['text'].apply(lambda x: " ".join([w for w in x.split() if w not in self.stopWords and w != '']))
        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.3, random_state=42, shuffle=False)

        y_test = self.dev['label'].map(self.labelToIndex)
        X_test = self.dev['text'].apply(lambda x: " ".join([w for w in x.split() if w not in self.stopWords and w != '']))

        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        train_encoding = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=64)
        eval_encoding = tokenizer(X_eval.tolist(), truncation=True, padding=True, max_length=64)
        test_encoding = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=64)

        train_dataset = JD_Dataset(train_encoding, y_train)
        eval_dataset = JD_Dataset(eval_encoding, y_eval)
        test_dataset = JD_Dataset(test_encoding, y_test)

        self.model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=11)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # 单个读取到批量读取
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

        for epoch in range(2):
            print("------------Epoch: %d ----------------" % epoch)
            self.train_model(train_loader, device, epoch)
            self.validate_model(test_loader, device)

    # 训练函数
    def train_model(self, train_loader, device, epoch):
        self.model.train()
        total_train_accuracy = 0
        total_train_loss = 0
        iter_num = 0
        total_iter = len(train_loader)

        # 优化方法
        optim = AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * 1
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps = 0, # Default value in run_glue.py
            num_training_steps = total_steps
            )

        for batch in train_loader:
            # 正向传播
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_train_loss += loss.item()
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_train_accuracy += self.flat_accuracy(logits, label_ids)

            # 反向梯度信息
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # 参数更新
            optim.step()
            scheduler.step()

            iter_num += 1
            if (iter_num % 100 == 0):
                print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                    epoch, iter_num, loss.item(), iter_num / total_iter * 100))

        print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))
        avg_train_accuracy = total_train_accuracy / len(train_loader)
        print("Training accuracy: %.4f" % (avg_train_accuracy))

    def validate_model(self, test_loader, device):
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        for batch in test_loader:
            with torch.no_grad():
                # 正常传播
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            logits = outputs[1]

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(test_loader)
        print("Test accuracy: %.4f" % (avg_val_accuracy))
        print("Average testing loss: %.4f" % (total_eval_loss / len(test_loader)))
        print("-------------------------------")

    def save(self):
        torch.save(self.model.state_dict(), './model/clf_dl')

    # 精度计算
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


if __name__ == "__main__":
    bc = Classifier()
    bc.process()
    bc.save()

