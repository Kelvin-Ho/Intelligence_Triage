# -*- coding: UTF-8 -*-
'''
Author: xiaoyao jiang
LastEditors: Peixin Lin
Date: 2020-08-31 14:19:30
LastEditTime: 2021-01-03 21:36:09
FilePath: /JD_NLP1-text_classfication/model.py
Desciption:
'''
import json
import jieba
import joblib
import lightgbm as lgb
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTEENN

from embedding import Embedding
from features import (get_basic_feature, get_embedding_feature,
                      get_lda_features, get_tfidf, label2idx)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from skopt import BayesSearchCV # scikit-optimize 版本需要0.8.0以上，否则与scikit-learn不兼容
from skopt.space import Real, Integer # 定义搜索空间

from simpletransformers.classification import ClassificationModel, ClassificationArgs

class Classifier:
    def __init__(self, train_mode=False) -> None:
        self.stopWords = [x.strip() for x in open(r'./data/stopwords.txt', 'r', encoding='UTF-8-sig').readlines()]
        self.embedding = Embedding()
        self.embedding.load()
        self.labelToIndex = json.load(
            open('./data/label2id.json', encoding='utf-8-sig'))
        self.ix2label = {v: k for k, v in self.labelToIndex.items()}
        if train_mode:
            self.train = pd.read_csv('./data/train.csv',
                                     sep='\t').dropna().reset_index(drop=True)[:]  # revised
            self.dev = pd.read_csv('./data/eval.csv',
                                   sep='\t').dropna().reset_index(drop=True)[:]  # revised
            self.test = pd.read_csv('./data/test.csv',
                                    sep='\t').dropna().reset_index(drop=True)[:]  # revised
        self.exclusive_col = ['text', 'lda', 'bow', 'label']

        # label2idx
        # self.labelToIndex = ix2label(self.train)
        self.train['label'] = self.train['label'].map(self.labelToIndex)
        self.dev['label'] = self.dev['label'].map(self.labelToIndex)
        self.test['label'] = self.test['label'].map(self.labelToIndex)

    def feature_engineer(self, data):
        data = get_tfidf(self.embedding.tfidf, data)
        print("get tfidf feature")
        data = get_embedding_feature(data, self.embedding.w2v)
        print("get w2v feature")
        data = get_lda_features(data, self.embedding.lda)
        print("get lda feature")
        data = get_basic_feature(data)
        print("get basic feature")
        return data

    def trainer(self, isBaseline=False, isGridSearchCV=False, isBayesSearchCV=False, isImbalanced_oversampling=False,
                isImbalanced_undersampling=False, isImbalanced_SMOTEENN=False, isTransformer=False):
        print("start of trainer")  # marker

        if isTransformer == True:
            self.train = self.train[:]
            # self.train['label'] = self.train['label'].map(self.labelToIndex)
            self.train['text'] = self.train['text'].apply(
                lambda x: " ".join([w for w in x.split() if w not in self.stopWords and w != '']))
            train_df, eval_df = train_test_split(self.train[['text', 'label']], test_size=0.3, random_state=42,
                                                 shuffle=True)
            y_train = train_df['label']

            # self.dev['label'] = self.dev['label'].map(self.labelToIndex)
            self.dev['text'] = self.dev['text'].apply(
                lambda x: " ".join([w for w in x.split() if w not in self.stopWords and w != '']))
            test_df = self.dev[['text', 'label']]
            y_test = test_df['label']

        else:
            # 如果采用机器学习模型，需要做特征工程
            self.train = self.feature_engineer(self.train[:])
            dev = self.feature_engineer(self.dev[:])

            cols = [x for x in self.train.columns if x not in self.exclusive_col]
            print("finish feature engineering")

            X_train = self.train[cols]
            y_train = self.train['label']

            X_test = dev[cols]
            y_test = dev['label']

            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

        num_labels = len(y_train.unique())

        lgb_params = {
            'n_jobs': -1,  # 多线程并行运算的核的数量
            'num_leaves': 30,  # 数的最大叶子数，控制模型复杂程度的关键参数
            'max_depth': 5,  # 决策树的最大深度
            'learning_rate': 0.1,  # 学习率
            'n_estimators': 500,  # 弱学习器的数量，可理解为训练的轮数
            'reg_alpha': 0,  # L1正则化系数
            'reg_lambda': 1,  # L2正则化系数
            'objective': 'multiclass',  # 任务目标
            'metric': 'multi_logloss',  # 模型度量标准
            'num_class': num_labels,  # 类别数
            'device': 'gpu',  # 模型运算设备
            'early_stopping_round': 500,  # 提前终止训练的轮数
            'verbosity': -1,  # 打印训练信息
            'random_state': 42,  # 随机种子数
            # 'colsample_bytree':0.9,
            # 'subsample':0.6,
        }

        self.clf = lgb.LGBMClassifier(**lgb_params)

        if isGridSearchCV == True:
            print("grid search optimization")
            gsearch = self.Grid_Train_model(self.clf, X_train[:3500], y_train[:3500], X_eval[:1500], y_eval[:1500])
            train_prediction = gsearch.predict(X_train)
            test_prediction = gsearch.predict(X_test)
            self.print_score(y_train, train_prediction, y_test, test_prediction)

        elif isBayesSearchCV == True:
            print("Bayes optimization")
            # 贝叶斯优化采用一部分样本训练
            clf = lgb.LGBMClassifier(
                n_jobs=-1,
                device='gpu',
                verbosity=-1,
                objective='multiclass',
                metric='multi_logloss',
                num_class=num_labels,
                random_state=42,
            )
            bayes_cv_tuner = self.BayesOptimize_Train_model(clf, X_train[:3500], y_train[:3500], X_eval[:1500],
                                                            y_eval[:1500])
            best_params = pd.Series(bayes_cv_tuner.best_params_)
            print("best parameters are {}".format(best_params))
            param_dict = pd.Series.to_dict(best_params)

            # 使用全部样本重新训练模型
            bayes_opt_model = lgb.LGBMClassifier(
                learning_rate=param_dict['learning_rate'],
                max_depth=int(param_dict['max_depth']),
                num_leaves=int(param_dict['num_leaves']),
                max_bin=int(param_dict['max_bin']),
                min_child_samples=int(param_dict['min_child_samples']),
                subsample=param_dict['subsample'],
                subsample_freq=int(param_dict['subsample_freq']),
                colsample_bytree=param_dict['colsample_bytree'],
                min_split_gain=param_dict['min_split_gain'],
                subsample_for_bin=int(param_dict['subsample_for_bin']),
                reg_lambda=param_dict['reg_lambda'],
                reg_alpha=param_dict['reg_alpha'],
                n_estimators=int(param_dict['n_estimators']),
                n_jobs=-1,
                objective='multiclass',
                num_class=num_labels,
                device='gpu',
                early_stopping=500,
                verbosity=-1,
                random_state=42,
            )

            bayes_opt_model.fit(X_train[:], y_train[:], eval_set=[(X_eval, y_eval)],
                                eval_metric='logloss')

            train_prediction = bayes_opt_model.predict(X_train)
            test_prediction = bayes_opt_model.predict(X_test)
            self.print_score(y_train, train_prediction, y_test, test_prediction)

        elif isImbalanced_oversampling == True:
            print('Initial label distribution: \n{}'.format(y_train.value_counts()))
            print("Start over-sampling")
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
            print('Label distribution after over-sampling: \n{}'.format(y_train.value_counts()))
            print('X_train: ', X_train.shape, 'y_train: ', y_train.shape)
            print("start of lgb training")
            self.clf.fit(X=X_train, y=y_train, eval_set=[(X_eval, y_eval)],
                         eval_metric='logloss')
            train_prediction = self.clf.predict(X_train)
            test_prediction = self.clf.predict(X_test)
            self.print_score(y_train, train_prediction, y_test, test_prediction)

        elif isImbalanced_undersampling == True:
            print('Initial label distribution: \n{}'.format(y_train.value_counts()))
            print("Start under-sampling")
            cc = ClusterCentroids()
            X_train, y_train = cc.fit_resample(X_train, y_train)
            print('Label distribution after under-sampling: \n{}'.format(y_train.value_counts()))
            print('X_train: ', X_train.shape, 'y_train: ', y_train.shape)
            print("start of lgb training")
            self.clf.fit(X=X_train, y=y_train, eval_set=[(X_eval, y_eval)],
                         eval_metric='logloss')
            train_prediction = self.clf.predict(X_train)
            test_prediction = self.clf.predict(X_test)
            self.print_score(y_train, train_prediction, y_test, test_prediction)

        elif isImbalanced_SMOTEENN == True:
            print('Initial label distribution: \n{}'.format(y_train.value_counts()))
            print("Start over-sampling and under-sampling")
            sme = SMOTEENN()
            X_train, y_train = sme.fit_resample(X_train, y_train)
            print('Label distribution after SMOTEENN: \n{}'.format(y_train.value_counts()))
            print('X_train: ', X_train.shape, 'y_train: ', y_train.shape)
            print("start of lgb training")
            self.clf.fit(X=X_train, y=y_train, eval_set=[(X_eval, y_eval)],
                         eval_metric='logloss')
            train_prediction = self.clf.predict(X_train)
            test_prediction = self.clf.predict(X_test)
            self.print_score(y_train, train_prediction, y_test, test_prediction)

        elif isTransformer == True:
            model_args = ClassificationArgs(
                max_seq_length=512,
                train_batch_size=16,
                num_train_epochs=5,
                fp16=False,
                evaluate_during_training=True,
                evaluate_during_training_verbose=False,
                overwrite_output_dir=True,
            )

            model_type = 'bert'
            model_name = 'bert-base-chinese'
            print("train {}......".format(model_name))
            model_args.cache_dir = './caches' + '/' + model_name.split('/')[-1]
            model_args.output_dir = './outputs' + '/' + model_name.split('/')[-1]

            model = ClassificationModel(
                model_type,
                model_name,
                num_labels=11,
                use_cuda=False,
                args=model_args,
            )

            model.train_model(
                train_df=train_df,
                eval_df=eval_df,
            )
            train_prediction, _ = model.predict(train_df)
            test_prediction, _ = model.predict(test_df)
            self.print_score(y_train, train_prediction, y_test, test_prediction)

        else:
            print("start of lgb training")
            self.clf.fit(X=X_train, y=y_train, eval_set=[(X_eval, y_eval)],
                         eval_metric='logloss')
            train_prediction = self.clf.predict(X_train)
            test_prediction = self.clf.predict(X_test)
            self.print_score(y_train, train_prediction, y_test, test_prediction)

    def print_score(self, y_train, train_prediction, y_test, test_prediction):
        print(
            "------------------------------------------------------------------------------------------------------------")
        print("Score of training set:")
        print("accuracy score is {}".format(metrics.accuracy_score(y_train, train_prediction)))
        print("precision score is {}".format(metrics.precision_score(y_train, train_prediction, average='weighted')))
        print("recall score is {}".format(metrics.recall_score(y_train, train_prediction, average='weighted')))
        print("f1 score is {}".format(metrics.f1_score(y_train, train_prediction, average='weighted')))
        # print("auc score is {}".format(metrics.roc_auc_score(y_train, train_prediction, average='weighted')))

        print(
            "------------------------------------------------------------------------------------------------------------")
        print("Score of test set:")
        print("accuracy score is {}".format(metrics.accuracy_score(y_test, test_prediction)))
        print("precision score is {}".format(metrics.precision_score(y_test, test_prediction, average='weighted')))
        print("recall score is {}".format(metrics.recall_score(y_test, test_prediction, average='weighted')))
        print("f1 score is {}".format(metrics.f1_score(y_test, test_prediction, average='weighted')))
        # print("auc score is {}".format(metrics.roc_auc_score(y_test, test_prediction, average='weighted')))

    def Grid_Train_model(self, model, X_train, y_train, X_eval, y_eval):
        parameters = {'max_depth': [4, 5],
                      'learning_rate': [0.03, 0.05],
                      'n_estimators': [500, 700],
                      'subsample': [0.6, 0.9],
                      'colsample_bytree': [0.6, 0.9],
                      'reg_alpha': [5, 10],
                      # 'reg_lambda': [10, 50],
                      }

        fit_params = {"eval_set": [[X_eval, y_eval]]}
        gsearch = GridSearchCV(model,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=2,
                               verbose=4)
        gsearch.fit(X_train, y_train, **fit_params)
        print("best parameters are {}".format(gsearch.best_params_))
        return gsearch

    def BayesOptimize_Train_model(self, model, X_train, y_train, X_eval, y_eval):
        search_space = {
            'learning_rate': Real(0.01, 1.0, 'log-uniform'),
            'max_depth': Integer(3, 10),
            'num_leaves': Integer(2, 100),
            'max_bin': Integer(5, 200),
            'min_child_samples': Integer(20, 100),
            'subsample': Real(0.1, 1.0, 'uniform'),
            'subsample_freq': Integer(1, 50, 'uniform'),
            'colsample_bytree': Real(0.1, 1.0, 'uniform'),
            'min_split_gain': Real(0, 1.0, 'uniform'),
            'subsample_for_bin': Integer(10000, 20000),
            'reg_lambda': Real(0, 10, 'uniform'),
            'reg_alpha': Real(0, 10, 'uniform'),
            'n_estimators': Integer(10, 1000),
        }

        fit_params = {"eval_set": [[X_eval, y_eval]]}

        bayes_cv_tuner = BayesSearchCV(estimator=model,
                                       search_spaces=search_space,
                                       # scoring='accuracy',
                                       cv=KFold(n_splits=2),
                                       n_iter=20,
                                       verbose=4,
                                       refit=True,
                                       return_train_score=True,
                                       random_state=42,
                                       )

        bayes_cv_tuner.fit(X_train, y_train, **fit_params)
        return bayes_cv_tuner

    def isImbalanced(self, model, train_data, train_label):
        print("ok")

    def data_transform_mlb(self, train_label, test_label):
        # 初始化多标签训练
        mlb = MultiLabelBinarizer(sparse_output=False)

        y_train_new = []
        y_test_new = []
        for i in train_label:
            y_train_new.append([i])
        for i in test_label:
            y_test_new.append([i])

        y_train = mlb.fit_transform(y_train_new)
        y_test = mlb.transform(y_test_new)
        print('number of labels: ', mlb.classes_)
        return y_train, y_test

    def save(self):
        joblib.dump(self.clf, './model/clf')

    def load(self):
        self.model = joblib.load('./model/clf')

    def predict(self, text, desc):
        df = pd.DataFrame([[text]], columns=['text'])
        df['text'] = df['text'].apply(lambda x: " ".join(
            [w for w in jieba.cut(x) if w not in self.stopWords and w != '']))
        df = get_tfidf(self.embedding.tfidf, df)

        print("marker")
        print(df)  # check

        df = get_embedding_feature(df, self.embedding.w2v)
        df = get_lda_features(df, self.embedding.lda)
        df = get_basic_feature(df)
        cols = [x for x in df.columns if x not in self.exclusive_col]
        #######################################################################
        #          TODO:  lgb模型预测 #
        #######################################################################
        # 利用模型获得预测结果
        pred = self.model.predict(df[cols]).toarray()[0]

        return [self.ix2label[i] for i in range(len(pred)) if pred[i] > 0]

if __name__ == "__main__":
    bc = Classifier(train_mode=True)
    bc.trainer(isBaseline=True)
    bc.save()
