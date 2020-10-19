# My-HobbyProject-HorseRacingAI
院に入って, 研究を行うにあたって, 機械学習の勉強を始めました.<br> 
さらに,最近コロナの影響でおうち時間が増えたことを利用して,教授に内緒で,競馬の順位や買い目を予測するAIの自作に挑戦しています.

使っている時間が趣味:研究=8:2ぐらいになっていて困っています.

# プログラムの説明
ネットから, 機械学習に必要なデータをスクレイピングし, 特徴量を抽出, ハイパーパラメータ自動最適化ツールなどを用いて機械学習モデルを自作, 学習・テストをしています.<br>

* 1-2-2sclape_traindata.ipynb:<br>
ネット競馬(https://www.netkeiba.com/) からレース結果(教師データ)をスクレピングするプログラム
* 1-4sclape_horse_results.ipynb:<br>
ネット競馬(https://www.netkeiba.com/) から馬ごとのデータ(特徴量)をスクレピングするプログラム
* 1-8sclape_jocky_results.ipynb:<br>
ネット競馬(https://www.netkeiba.com/) から騎手ごとのデータ(特徴量)をスクレピングするプログラム
* 1-11sclape_Pedigree.ipynb:<br>
ネット競馬(https://www.netkeiba.com/) から馬の血統データ(特徴量)をスクレピングするプログラム
* 1-12sclape_tables.ipynb:<br>
ネット競馬(https://www.netkeiba.com/) から払い戻し金額データをスクレピングするプログラム

* 1-15model_hold.ipynb:<br>
それらデータを加工, モデル作成・検証を行うプログラム

# 学んだこと
* Pythonの基礎文法知識(スクレイピング, データ加工)
* 機械学習の基礎知識
* プログラム全体の設計
* 仮説・検証,PDCAの大切さ

# 課題
* プログラムの整理
* モデルの改良
* 保守や運用
* 自動化

# Author
* 浮田 凌佑
* 立命館大院 情報理工学研究科 情報理工学専攻
* is0343sf@ed.ritsumei.ac.jp
