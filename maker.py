import pandas as pd
import numpy as np

import quandl
import pickle
import requests
import sys
import json

from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

#quandlのAPIキーのロード
#ユーザー登録するだけで無料で利用できるので自前のに書き換えてください
with open('conf.json') as f:
    quandl.ApiConfig.api_key = json.load(f)['api']

#データを保存するディレクトリ
INPUT_DIR = 'dataset/'

#log用
RESULT_DIR = 'output/'

logger = getLogger(__name__)

log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(RESULT_DIR + 'make.py.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)





def download_tse():
    '''
    http://kabusapo.com/dl-file/dl-stocklist.php　から東証に上場中の銘柄のリストを確保
    リストを使いquandlから値動きを取得
    データ数一定以下とマザーズの銘柄は除く
    '''
    logger.debug('enter')

    #東証に現在上場中の銘柄の一覧をGETで入手
    #文字コード周りめんどうだったので引用元変更
    #result = requests.get('http://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls')
    result = requests.get('http://kabusapo.com/dl-file/dl-stocklist.php')
    
    #エラーはいたら止める
    #例外処理をするかどうかは検討中
    result.raise_for_status()

    #一旦ファイルにcsvとして書き出し
    with open(INPUT_DIR+'TSE_List.csv', 'wb') as f:
        for chunk in result.iter_content(100000):
                f.write(chunk)

    #csvを読み込み直す
    #TODO:read_csvはURLからも読めるらしいのでここで読んだほうが良さそう
    tselist = pd.read_csv(INPUT_DIR + 'TSE_List.csv')
    #マザーズ以外の銘柄コードをリストとして確保
    codelist = list(tselist[tselist['市場名'] != 'マザーズ']['銘柄コード'])

    logger.debug('current codelist: {}'.format(codelist))

    #入手したリストとQuandlにあるデータを照らし合わせながダウンロード
    #使用銘柄の一覧の記録も同時に行うのでリストのコピーをとってイテレーターに
    for c in tqdm(codelist[:]):
        #Quandlのコードに変換
        name = 'TSE/' + str(c)
        #データがあったら
        try:
            #データを入手
            df = quandl.get(name)
            #データ数が閾値に達しているかチェック
            if len(df.index.values) > 50:
                #使用可能なのでファイルに記録
                df.to_csv(INPUT_DIR + name + '.csv')
                logger.debug('TSE {}: success'.format(c))
            else:
                #使用不可能なので一覧から削除
                codelist.remove(c)
                logger.debug('TSE {}: failed by datanum'.format(c))
        #データがなかったら
        except Exception as e:
            #使用銘柄から削除
            codelist.remove(c)
            logger.debug('TSE {}: failed by not exist'.format(c))

    logger.debug('remein codelist: {}'.format(codelist))

    #pickleとして使用銘柄コード一覧を記録
    with open(INPUT_DIR + 'TSE_List.pkl', 'wb') as f:
        pickle.dump(codelist, f)

    logger.debug('exit')




def dataloader(code):
    '''
    生データから銘柄ごとの各種株価の平均をとった状態で読み込み
    '''
    logger.debug('enter')

    #銘柄コードからファイル名に変換
    name = INPUT_DIR + 'TSE/' + str(code) + '.csv'
    #csvから取引量以外の値(始値終値高値低値)を読みだし
    df = pd.read_csv(name).drop('Volume', axis=1)
    #日付と各値の平均値で新しいフレーム作成
    df = pd.concat([df['Date'], np.mean(df, axis=1)], axis=1)
    #行名を設定
    df.columns = ['Date', 'TSE_' + str(code)]

    logger.debug('exit')
    return df




def sheetmaker():
    '''
    生データをもとにdataloaderで加工し、全銘柄ひとまとめにしてpickleで保存
    '''
    logger.debug('enter')

    #使用銘柄コード一覧を読み込み
    with open(INPUT_DIR + 'TSE_List.pkl', 'rb') as f:
        codelist = pickle.load(f)

    df = pd.DataFrame(columns=['Date'])

    #csvから読み込んで各値の平均値をひとつのデータフレームとしてまとめる
    for code in tqdm(codelist):
        #コードごとに順番にロードして日付基準でouter結合
        df = pd.merge(df, dataloader(code), on='Date', how='outer')

    #nanの除去
    df = df.fillna(0)

    #できたデータフレームをpickleとして書き出し
    with open(INPUT_DIR + 'TSE_Sheet.pkl', 'wb') as f:
        pickle.dump(df, f)

    logger.debug('exit')
    return df

if __name__ == '__main__':
    if '--all' in sys.argv:
        download_tse()

    if '--sheet' in sys.argv:
        sheetmaker()
