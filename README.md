# 勾配ブースティングによる株価予測 #
# 概要
東京証券取引所の株価データの上昇下降の予測をXGBoostによって行います。  
上昇を1、下降を0に対応させて予測を行いました。  
特徴量として時間相関のある株価の値動きを利用しました。  
# APIについて
実行にはquandlのAPIキーが必要です。  
取得したキーをconf.jsonというファイル名で以下のような形式で保存してください。  
`{"api":"取得したキー"}`
# 実行方法
maker.py --all --sheet -> loader.py --write -> trainer.py --train --pred の順に実行していくと推定結果がoutput直下に出力されます。対象銘柄や使用データ数の変更や,確率値とラベルでの出力の切り替えはコードを書き換えてください。またtrainer.py --checkを実行することで現状の正解率を標準出力に表示可能です。
# 補足
ipynbだったものを簡易的にまとめたものなので随時さらに整理していきます。
