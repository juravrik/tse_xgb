import pandas as pd
import scipy.stats

import pickle
import sys

from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import maker

RESULT_DIR = 'output/'

#log用
logger = getLogger(__name__)

log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(RESULT_DIR + 'load.py.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)



def conv(code):
    '''
    数値から証券コードの文字列への変換器
    '''
    return 'TSE_' + str(code)

def sheetslider(code, sheet):
    '''
    対象データを一行スライド
    '''
    #スライド処理
    sheet[conv(code)] = sheet[conv(code)].shift(1)
    #かけたデータを切り落とす
    sheet = sheet.drop(0).reset_index(drop=True)
    return sheet


def calc_pearsonr(code, sheet, originsheet, codelist, pear_line = 0.7):
    '''
    相関係数を計算し閾値を超えた列を抽出
    '''

    #記録用のデータフレームを用意
    df = pd.DataFrame()
    #一日ずつデータをずらしながら相関係数を計算
    for c in tqdm(codelist[:]):
        #相関係数を計算
        pv = scipy.stats.pearsonr(sheet[conv(code)], sheet[conv(c)])[0]
        #相関係数が閾値を超えていたら
        if abs(pv) > pear_line:
            #記録用のフレームに合成
            df = pd.concat([df, originsheet[conv(c)]], axis=1)
            #一度抽出した銘柄はもういらないので削除
            #この処理いらない気がしている
            codelist.remove(c)

    return df


def expander(sheet):
    '''
    過去七日間ずつの値動きをひとまとめに
    mapを使ってみたかった
    '''

    #ひとつのシートから、過去七日間のみにまとめられたシートを生成
    debris = map(lambda i: sheet[i-6:i+1], range(6, len(sheet)))

    
    def deform(df):
        """
        何日前の株価かをサフィックスとして列名に示した状態で
        過去七日間の値動きを一行にまとめたシートに変形
        """
        
        #サフィックスつけて列を抜き取っていく無名関数
        frag = lambda x:df.add_suffix('_' + str(x)).iloc[x]
        #過去七日間の値動きを一行にまとめたシートに変形
        return pd.concat(list(map(frag, reversed(range(7)))), axis=0)

    #debrisで生成した複数のシートに対してdefromを適用
    data_sheet = pd.concat(list(map(deform, debris)), axis=1).T
    return data_sheet



def load_dataset(obj_code, test_num, use_num):
    """
    目標とする銘柄と学習に使用する全データ数およびそのうち
    いくつをテストに使用するかを受けて、xgboostに突っ込めるように
    """

    #分析対象銘柄の一覧をロード
    with open(maker.INPUT_DIR+'TSE_List.pkl', 'rb') as f:
        codelist = pickle.load(f)

    #値動きの一覧をロード
    with open(maker.INPUT_DIR+'TSE_Sheet.pkl', 'rb') as f:
        origin_sheet = pickle.load(f)

    #使用データ数が手持ちのデータ数を超えていたら全部を使用する
    if(use_num > len(origin_sheet)):
        use_num = len(origin_sheet)

    #使用するデータ数分をスライス
    sheet = origin_sheet.drop('Date', axis=1)[-use_num:].reset_index().copy()

    #実際の上昇下降に合わせてターゲットラベルを作成
    target = pd.DataFrame(columns={'target'})
    for i in range(len(sheet) - 1):
        if sheet[conv(obj_code)][i] <= sheet[conv(obj_code)][i + 1]:
            #上昇が1
            d = 1
        else:
            #下降が0
            d = 0
        target.loc[i] = d

    #最新データは実際の上下がわからないのでカット
    sheet = sheet.drop(len(sheet)-1)
    
    #相関計算用のシートを用意
    #テストに使用する場所については相関を見ないのでカット
    calc_sheet = sheet[:-test_num].reset_index()

    #相関ある銘柄を抜き出す
    #記録用シートを用意
    #こっちには相関を見なかった部分の値動きも含まれている
    base_sheet = pd.DataFrame()
    for i in range(7):
        #相関あるのを抽出
        df = calc_pearsonr(obj_code, calc_sheet, sheet, codelist)
        #これまでのやつと結合
        base_sheet = pd.concat([base_sheet, df], axis=1)
        #日付をずらす
        calc_sheet = sheetslider(obj_code, calc_sheet)
        

    #行ごとに過去一週間の値動きをのせる
    base_sheet = expander(base_sheet)
    
    #予め作ってあったターゲットラベルと結合
    #最新6日間に関しては一週間分の値動きが用意できないのでカット
    base_sheet = pd.concat([target[6:].reset_index(drop=True), base_sheet], axis=1)

    #トレインセットを用意
    train_sheet = base_sheet[:-test_num]
    #テストセットを用意
    test_sheet = base_sheet[-test_num:]
    #途中から切り出したのでインデックスの付け直し
    test_sheet = test_sheet.reset_index(drop=True)

    return train_sheet, test_sheet

def load_traindata():
    """
    学習データの読み出し用
    """
    df = pd.read_csv(maker.INPUT_DIR + 'train_data.csv')
    return df

def load_testdata():
    '''
    テストデータの読み出し用
    '''
    df = pd.read_csv(maker.INPUT_DIR + 'test_data.csv')
    return df


def write_data(obj_code=1885, test_num=7, use_num=5000):
    '''
    データセットのcsv書き出し関数
    手動でやる場合はここのデフォルト値を書き換える
    obj_code -> 予測対象銘柄
    test_num -> テストセットのデータ数
    use_num -> 使用するデータ数(手持ちデータ数を超えていたら全部使用)
    '''
    train, test = load_dataset(obj_code, test_num, use_num)

    train.to_csv(maker.INPUT_DIR + 'train_data.csv', index=False)
    test.to_csv(maker.INPUT_DIR + 'test_data.csv', index=False)


if __name__ == '__main__':
    if '--write' in sys.argv:
        write_data()
