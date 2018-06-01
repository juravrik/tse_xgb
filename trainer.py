import pickle
import sys
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import pandas as pd
import numpy as np
import xgboost as xgb

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, accuracy_score

import loader
import maker

RESULT_DIR = 'output/'

logger = getLogger(__name__)

log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(RESULT_DIR + 'train.py.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)




def train():
    """
    学習工程本体
    """
    logger.info('start')

    #トレインセットをロード
    df = loader.load_traindata()

    #トレインセットを入力と解答ラベルに分割
    x_train = df.drop('target', axis=1)
    y_train = df['target'].values


    logger.info('data preparation end {}'.format(x_train.shape))

    #バリデーション用の分割
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    #パラメータチューン用のディクショナリ
    #とりあえず一回チューンしたのでその時のだけにしてある
    all_params = {'max_depth': [3],
                  'learning_rate': [0.1],#[0.01]
                  'min_child_weight': [1],
                  'n_estimators': [10000],
                  'colsample_bytree': [0.8],
                  'colsample_bylevel': [0.8],
                  'reg_alpha': [0.1],
                  'max_delta_step': [0.1],
                  'seed': [0],
                  'gamma': [0.3]
                  }
    #バリデーション用の記録用変数
    #ダミーの値をとりあえず
    min_score = 100
    min_params = None



    #パラメータチューン用のループ
    #現状意味をなしてないけど書きなおすの面倒なので残ってる
    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))

        #記録用のダミーリスト
        list_logloss_score = []
        list_best_iterations = []
        #交差検証用のループ
        for train_idx, valid_idx in cv.split(x_train, y_train):
            #選ばれたバリデーションセットを分離
            trn_x = x_train.iloc[train_idx, :]
            val_x = x_train.iloc[valid_idx, :]

            #選ばれたトレインセットをまとめる
            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            #インスタンスの作成
            clf = xgb.sklearn.XGBClassifier(**params)
            #データから学習
            clf.fit(trn_x,
                    trn_y,
                    eval_set=[(val_x, val_y)],
                    early_stopping_rounds=100,
                    eval_metric='logloss'
                    )

            #確率値で予測の出力
            pred = clf.predict_proba(val_x, ntree_limit=clf.best_ntree_limit)[:, 1]
            logger.debug('val_y: {}, pred: {}'.format(len(val_y), len(pred)))
            #バリデーションセットに対するスコア(logloss)の計算
            sc_logloss = log_loss(val_y, pred)
            #スコアを記録
            list_logloss_score.append(sc_logloss)
            #
            list_best_iterations.append(clf.best_iteration)
            logger.debug('   logloss: {}'.format(sc_logloss))
            #ここn_estimatorsちゃんとすればbreakしちゃってよさそう
            #break

        #現在のパラメータによるスコアの記録
        params['n_estimators'] = int(np.mean(list_best_iterations))
        sc_logloss = np.mean(list_logloss_score)
        #ベストを更新していたら塗り替え
        if min_score > sc_logloss:
            min_score = sc_logloss
            min_params = params

        logger.info('logloss: {}'.format(sc_logloss))
        logger.info('current min score: {}, params: {}'.format(min_score, min_params))

    logger.info('minimum params: {}'.format(min_params))
    logger.info('minimum logloss: {}'.format(min_score))

    #採用するパラメータによるモデルの作成
    clf = xgb.sklearn.XGBClassifier(**min_params)
    logger.info('prep end')
    #データから学習
    clf.fit(x_train, y_train)
    logger.info('train end')

    #完成したモデルの書き出し
    with open(RESULT_DIR + 'model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)

    return clf



def pred():
    with open(RESULT_DIR + 'model.pkl', 'rb') as f:
        clf = pickle.load(f)

    df = loader.load_testdata()

    x_test = df.drop('target', axis=1)
    pred_test = clf.predict_proba(x_test)[:, 1]
    #pred_test = clf.predict(x_test)
    pd.DataFrame(pred_test, columns=['predict']).to_csv(RESULT_DIR+'pred.csv', index=False)




def check():
    result = pd.read_csv(RESULT_DIR+'pred.csv')['predict']
    ans = pd.read_csv(maker.INPUT_DIR+'test_data.csv')['target']

    return accuracy_score(ans, result)



if __name__ == '__main__':
    if '--train' in sys.argv:
        train()

    if '--pred' in sys.argv:
        pred()

    if '--check' in sys.argv:
        print(check())
