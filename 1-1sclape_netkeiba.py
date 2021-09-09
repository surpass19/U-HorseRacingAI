# 実際の当日の未知のデータをスクレイピングする → 予測に使える形にする
import pandas as pd
import time
from tqdm import tqdm as tqdm
from bs4 import BeautifulSoup
import requests
# 正規表現のモジュール
import re
from tqdm import tqdm as tqdm
import lightgbm as lgb
import numpy as np

race_id = '202004010501'
race_id = '202008030911'
url = 'https://race.netkeiba.com/race/shutuba.html?race_id=' + race_id + '&rf=race_list'

#########################馬情報, read_htmlで, Tableタグは取ってこれる##############################
df = {}
url = 'https://race.netkeiba.com/race/shutuba.html?race_id=202008030911&rf=shutuba_submenu'
df[race_id] = pd.read_html(url, header=0)[0]
df[race_id] = pd.read_html(url)
df
df[race_id] = pd.read_html(url, header=0)[0].drop(
    [0, 1, 2], axis=0).drop(['お気に入り馬', 'Unnamed: 12'], axis=1)
for key in df.keys():
    df[key].index = [key] * len(df[key])
pd.concat([df[key] for key in df.keys()], sort=False)
df[race_id]
df
df = df.drop(['お気に入り馬', 'Unnamed: 12'], axis=1)
for key in df:
    df[key].index = race_id
df.index
results = pd.concat([test3[key] for key in test3.keys()], sort=False)


#########################レース情報, Tableタグ以外は, BeautifulSoupで ##############################
html = requests.get(url)
html.encoding = 'EUC-JP'
soup = BeautifulSoup(html.text, 'html.parser')

# text = soup.find("div", attrs={"class": 'RaceData01'}).text
text = soup.find("div", attrs={"class": 'RaceList_Item02'}).text
text
info = re.findall(r'\w+', text)
info

race_infos = {}
# 見づらいからdataframeに
info_dict = {}
info_dict['race_name'] = info[0]
for text in info:
    if '芝' in text:
        info_dict['race_type'] = '芝'
    if 'ダ' in text:
        info_dict['race_type'] = 'ダート'
    if '障' in text:
        info_dict['race_type'] = '障害'
    if 'm' in text:
        info_dict['course_len'] = int(re.findall(r'\d+', text)[0])
    if text in ['良', '稍重', '重', '不良']:
        info_dict['grand_state'] = text
    if text in ['曇', '晴', '雨', '小雨', '小雪', '雪']:
        info_dict['weather'] = text
    if '年' in text:
        info_dict['date'] = text
    if text in ['右', '左']:
        info_dict['race_type_direction'] = text

race_infos[race_id] = info_dict
pd.DataFrame(race_infos).T


########################horseとjockerも入れられるようにする###################################
race_id = '202008030911'
url = 'https://race.netkeiba.com/race/shutuba.html?race_id=' + race_id + '&rf=race_list'
html = requests.get(url)
html.encoding = 'EUC-JP'
soup = BeautifulSoup(html.text, 'html.parser')

soup.find("div", attrs={"class": 'RaceTableArea'}).find_all('a')

soup.find("div", attrs={"class": 'RaceTableArea'}).find_all(
    'a', attrs={'href': re.compile("^https://db.netkeiba.com/horse")})

soup.find("div", attrs={"class": 'RaceTableArea'}).find_all(
    'a', attrs={'href': re.compile("^https://db.netkeiba.com/jockey")})
test = soup.find("div", attrs={"class": 'RaceTableArea'}).find_all(
    'a', attrs={'href': re.compile("^https://db.netkeiba.com/horse")})
test[0]['href']
# この中の数字だけ
re.findall(r'\d+', test[0]['href'])
# リスト化
horse_id_list = []
for a in test:
    horse_id = re.findall(r'\d+', a['href'])
    horse_id_list.append(horse_id[0])

horse_id_list
######################### 前日までだとオッズと人気が取れない ##############################
race_id = '202008030911'
race_id = '202008031004'
url = 'https://race.netkeiba.com/race/shutuba.html?race_id=' + race_id + '&rf=race_list'
url2 = 'https://race.netkeiba.com/odds/index.html?race_id=' + \
    race_id + '&rf=race_submenu'
url3 = 'https://race.netkeiba.com/yoso/mark_list.html?race_id=' + \
    race_id + '&rf=race_submenu'
html = requests.get(url3)
html.encoding = 'EUC-JP'
soup = BeautifulSoup(html.text, 'html.parser')

pd.read_html(url3, header=0)
df[race_id] = pd.read_html(url2, header=0)[0]
df
# 人気
test = soup.find("div", attrs={"class": 'YosoTableWrap'}).find_all(
    "li", attrs={'class': 'Txt_C'})
test
test[-18]
popular_list = []
int((len(test) / 3))
lenght.astype(int)
for t in range(int(len(test) / 3)):
    print(t - int(len(test) / 3))
    popular_list.append(test[t - int(len(test) / 3)].text)
popular_list
# soup.find("div", attrs={"class": 'RaceOdds_HorseList Tanfuku'})
# soup.find("div", attrs={"class": 'RaceOdds_HorseList Tanfuku'}).find_all(
#     'td', attrs={'class': 'Ninki'})

# オッズ
soup.find("div", attrs={"class": 'YosoTableWrap'}).find_all(
    'li', attrs={'class': 'Popular'})
soup.find("div", attrs={"class": 'YosoTableWrap'}).find_all("li")
AAA = soup.find("div", attrs={"class": 'YosoTableWrap'}).find_all(
    'li', attrs={'class': 'Popular'})
AAA
AAA[8].text
odds_list = []
len(AAA)
for t in range(len(AAA)):
    odds_list.append(AAA[t].text)

odds_list
df = []
df["単勝"] = odds_list

######################### 日付 ##############################
url2 = 'https://race.netkeiba.com/race/shutuba.html?race_id=202008030911&rf=shutuba_submenu'
html = requests.get(url2)
html.encoding = 'EUC-JP'
soup = BeautifulSoup(html.text, 'html.parser')

# text = soup.find("div", attrs={"class": 'RaceData01'}).text
soup.find("div", attrs={"class": 'RaceList_Date clearfix'}).find_all('a')
test = soup.find("div", attrs={"class": 'RaceList_Date clearfix'}).find_all(
    'a', attrs={'class': 'Active'})
test[0]
test[0]['href']
re.findall(r'\d+', test[0]['href'])

#######################################################################


#############
### 関数化 ###
#############


def sclape_race_info_today(race_id):
    ####################### 当日の馬柱 ###############################
    url = 'https://race.netkeiba.com/race/shutuba.html?race_id=' + race_id + '&rf=race_list'
    df = {}
    df[race_id] = pd.read_html(url, header=0)[0].drop(
        [0, 1, 2], axis=0).drop(['お気に入り馬', 'Unnamed: 12', '人気'], axis=1)
    for key in df.keys():
        df[key].index = [key] * len(df[key])
    results_df = pd.concat([df[key] for key in df.keys()], sort=False)
    ##################当日の日付, horse_IDとjocker_IDも入れられるようにする#################
    html = requests.get(url)
    html.encoding = 'EUC-JP'
    soup = BeautifulSoup(html.text, 'html.parser')
    # date_list作成(同じ日が入る)
    date_list = []
    date_Active = soup.find("div", attrs={"class": 'RaceList_Date clearfix'}).find_all(
        'a', attrs={'class': 'Active'})
    date = re.findall(r'\d+', date_Active[0]['href'])
    # horse_id_list作成
    horse_id_list = []
    horse_a_list = soup.find("div", attrs={"class": 'RaceTableArea'}).find_all(
        'a', attrs={'href': re.compile("^https://db.netkeiba.com/horse")})
    # この中の数字だけ
    # 日付とhorse_idリスト化
    for a in horse_a_list:
        horse_id = re.findall(r'\d+', a['href'])
        horse_id_list.append(horse_id[0])
        # 同じ長さのdate_listも作成
        date_list.append(date[0])

    # jockey_id_list作成
    jockey_id_list = []
    jockey_a_list = soup.find("div", attrs={"class": 'RaceTableArea'}).find_all(
        'a', attrs={'href': re.compile("^https://db.netkeiba.com/jockey")})
    # この中の数字だけ
    # jockey_idリスト化
    for a in jockey_a_list:
        jockey_id = re.findall(r'\d+', a['href'])
        jockey_id_list.append(jockey_id[0])
    # results_dfに追加
    results_df["date"] = date_list
    results_df["horse_id"] = horse_id_list
    results_df["jockey_id"] = jockey_id_list
    #################################レースの情報#####################
    race_infos = {}
    text = soup.find("div", attrs={"class": 'RaceList_Item02'}).text
    info = re.findall(r'\w+', text)
    # 見づらいからdataframeに
    info_dict = {}
    info_dict['race_name'] = info[0]
    for text in info:
        if '芝' in text:
            info_dict['race_type'] = '芝'
        if 'ダ' in text:
            info_dict['race_type'] = 'ダート'
        if '障' in text:
            info_dict['race_type'] = '障害'
        if 'm' in text:
            info_dict['course_len'] = int(re.findall(r'\d+', text)[0])
        if text in ['良', '稍重', '重', '不良']:
            info_dict['grand_state'] = text
        if '稍' in text:
            info_dict['grand_state'] = '稍重'
        if text in ['曇', '晴', '雨', '小雨', '小雪', '雪']:
            info_dict['weather'] = text
        if '年' in text:
            info_dict['date'] = text
        if text in ['右', '左']:
            info_dict['race_type_direction'] = text

    race_infos[race_id] = info_dict
    race_infos_pd = pd.DataFrame(race_infos).T
    # つなげる
    race_today = results_df.merge(
        race_infos_pd, left_index=True, right_index=True, how='inner')
    ################## 当日オッズと人気 予想urlからとる#################
    url = 'https://race.netkeiba.com/yoso/mark_list.html?race_id=' + \
        race_id + '&rf=race_list'
    html = requests.get(url)
    html.encoding = 'EUC-JP'
    soup = BeautifulSoup(html.text, 'html.parser')

    # オッズ
    odds_list = []
    # 人気
    popular_list = []
    odds_li_list = soup.find("div", attrs={"class": 'YosoTableWrap'}).find_all(
        'li', attrs={'class': 'Popular'})
    popular_li_list = soup.find("div", attrs={"class": 'YosoTableWrap'}).find_all(
        "li", attrs={'class': 'Txt_C'})

    for t in range(len(odds_li_list)):
        odds_list.append(odds_li_list[t].text)
        popular_list.append(
            popular_li_list[t - int(len(popular_li_list) / 3)].text)

    race_today["単勝"] = odds_list
    race_today["人気"] = popular_list
    ########################### 前処理 ############################################
    # 性齢を性と年齢にわける
    race_today['性'] = race_today['性齢'].map(lambda x: str(x)[0])
    race_today['年齢'] = race_today['性齢'].map(lambda x: str(x)[1:]).astype(int)

    # 馬体重を体重と増減にわける
    race_today['体重'] = race_today['馬体重(増減)'].str.split(
        "(", expand=True)[0].astype(int)
    race_today['体重変化'] = race_today['馬体重(増減)'].str.split(
        "(", expand=True)[1].str[:-1].astype(int)
    # race_today['体重変化'] = race_today['馬体重(増減)'].str.split(
    #    "(", expand=True)[1].str[:-1]
    race_today.drop(['馬体重(増減)'], axis=1, inplace=True)

    race_today['枠'] = race_today['枠'].astype(int)
    race_today['馬番'] = race_today['馬番'].astype(int)
    race_today['斤量'] = race_today['斤量'].astype(float)

    # race_today['date'] = pd.to_datetime(race_today['date'], format='%Y年%m月%d日')
    race_today['date'] = pd.to_datetime(race_today['date'])
    race_today['course_len'] = race_today['course_len'].astype(int)
    race_today['単勝'] = race_today['単勝'].astype(float)
    race_today['人気'] = race_today['人気'].astype(float)

    # 不要な列を削除
    race_today.drop(['性齢'], axis=1, inplace=True)
    race_today.drop(['印'], axis=1, inplace=True)
    race_today.drop(['厩舎'], axis=1, inplace=True)
    race_today.drop(['Unnamed: 9'], axis=1, inplace=True)
    race_today.rename(columns={'枠': '枠番'}, inplace=True)
    # df.drop(['タイム', '着差', '調教師', '性齢', '馬体重', 'ﾀｲﾑ指数', '通過','上り', '調教ﾀｲﾑ', '厩舎ｺﾒﾝﾄ', '備考', '馬主', '賞金(万円)'], axis = 1, inplace = True)

    return race_today


#################
### 関数終わり ###
################
#######################################################################
Today_result = sclape_race_info_today('202005021011')
Today_result
Today_result.info()
Today_result.columns
# 当日の1レースのhorse_id,jockey_idが取り出せた
horse_id_list = Today_result['horse_id'].unique()
horse_id_list
len(horse_id_list)
# 当日の1レースのhorse_id,jockey_idから過去レースを取り出したい
url = 'https://db.netkeiba.com/horse/2016105547/'
pd.read_html(url)[3]


def sclape_Today_horse_pastresults(Today_result):
    # 当日の1レースのhorse_id, jockey_dから過去レースをスクレイピング(辞書型) → データフレーム化まで
    horse_results = {}
    horse_id_list = Today_result['horse_id'].unique()
    for horse_id in tqdm(horse_id_list):
        try:
            url = 'https://db.netkeiba.com/horse/' + horse_id
            df = pd.read_html(url)[3]
            if df.columns[0] == '受賞歴':
                df = pd.read_html(url)[4]
            horse_results[horse_id] = df
            time.sleep(0.1)
        except IndexError:
            print("IndexError")
            continue
        except:
            # for文から抜け出す
            print("bleak")
            break
    # horseIDをindexにして, DataFrame化
    for key in horse_results.keys():
        horse_results[key].index = [key] * len(horse_results[key])

    horse_results = pd.concat([horse_results[key]
                               for key in horse_results.keys()], sort=False)
    return horse_results


# 実行
Today_horse_past_results = sclape_Today_horse_pastresults(Today_result)

Today_result.info()
Today_result
Today_horse_past_results
Today_horse_past_results.info()
Today_horse_past_results[Today_horse_past_results.index == '2017102759']


class HorseResults:
    # 馬の過去データを加工する
    def __init__(self, horse_results):
        # このクラスを作った時に実行される関数
        # self. → このクラスの という意味
        self.horse_results = horse_results[['日付', '着順', '賞金']]
        self.preprosessing()
        # self.horse_results.rename(
        #     columns={'着順': '着順_ave', '賞金': '賞金_ave'}, inplace=True)

    def preprosessing(self):
        df = self.horse_results.copy()
        # 着順が数字以外のものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df['着順'] = df['着順'].astype(int)

        df['date'] = pd.to_datetime(df['日付'])
        df.drop(['日付'], axis=1, inplace=True)

        # 賞金のNaNを0でうめる
        df['賞金'].fillna(0, inplace=True)

        self.horse_results = df

    def average(self, horse_id_list, date, n_samples='all'):
        target_df = self.horse_results.loc[horse_id_list]
        # 過去何レース取り出すか指定
        if n_samples == 'all':
            filtered_df = target_df[target_df['date'] < date]
        elif n_samples > 0:
            filtered_df = target_df[target_df['date'] < date].sort_values(
                'date', ascending=False).groupby(level=0).head(n_samples)
        else:
            raise Exception('n_samples must be >0')

        average = filtered_df.groupby(level=0)[['着順', '賞金']].mean()

        return average.rename(columns={'着順': '着順_{}R'.format(n_samples), '賞金': '賞金_{}R'.format(n_samples)})

    def merge(self, results, date, n_samples='all'):
        # くっつけたいデータを日付で絞り込む
        df = results[results['date'] == date]
        horse_id_list = df['horse_id']
        merged_df = df.merge(self.average(
            horse_id_list, date, n_samples), left_on='horse_id', right_index=True, how='left')
        return merged_df

    def merge_all(self, results, n_samples='all'):
        date_list = results['date'].unique()
        merged_df = pd.concat([self.merge(results, date, n_samples)
                               for date in tqdm(date_list)])
        merged_df['着順_{}R'.format(n_samples)].fillna(0, inplace=True)
        merged_df['賞金_{}R'.format(n_samples)].fillna(0, inplace=True)
        return merged_df


Today_horse_past_results.head()
# これを実行 (Today_horse_past_resultsでインスタンス化)
hr = HorseResults(Today_horse_past_results)
hr.horse_results['賞金'].isnull().sum()

sample_date = Today_result['date'].iloc[0]
sample_date
horse_id_list = Today_result[Today_result['date'] == sample_date]['horse_id']
horse_id_list

target_df = hr.horse_results.loc[horse_id_list]
target_df[target_df['date'] < sample_date]
target_df[target_df['date'] < sample_date].groupby(
    level=0)[['着順_ave', '賞金_ave']].mean()

hr.average(horse_id_list, sample_date)
hr.merge_all(Today_result)

# これをToday_resultsにくっつけたい
results_m = hr.merge_all(Today_result)
results_m
# くっついた!
hr.horse_results['2013106119']
hr.horse_results.index == '2013106119'
sample.sort_values('date', ascending=False).groupby(level=0).head(3)

# これを実行
Today_results_3R = hr.merge_all(Today_result, n_samples=3)
Today_results_3R.drop(
    ['date', '騎手', 'horse_id', '馬名', 'race_type_direction', 'race_name', 'horse_id'], axis=1, inplace=True)
#Today_results_3R.drop(['date', '騎手', 'horse_id', '馬名', 'race_name', 'horse_id'], axis=1, inplace=True)

Today_results_3R.columns
Today_results_3R.info()
# 最終的にmodelに入れるDataFrame
Today_results_3R
Today_results_3R.columns
Today_results_3R
len(Today_results_3R)
Today_results_3R.info()
###############################
#predict_d = pd.get_dummies(Today_results_3R)
# predict_d.head()
# ダミー変数の数が合わない
results_3R.info()
results_3R_d = results_3R.drop(['date', 'rank'], axis=1)
results_3R_d.info()
Today_results_3R.info()
dummie = pd.concat([Today_results_3R, results_3R_d])
dummie
dummie.info()
dummie.head(len(Today_results_3R))
get_dummies = pd.get_dummies(dummie)
get_dummies.head(len(Today_results_3R))
get_dummies.info()
predict_d = get_dummies.head(len(Today_results_3R))
predict_d.info()
predict_d
len(predict_d)

predict_d.columns
# 結果表示(引数をTodayのものにする)
y_pred = lgb_clf.predict_proba(predict_d)[:, 1]
pd.Series(y_pred).sort_values(ascending=False)
pd.Series([0 if p < 0.73 else 1 for p in y_pred])
############################# モデル学習第9回 #####################################
horse_results = pd.read_pickle('horse7.pickle')
hr_all = HorseResults(horse_results)
results_p = pd.read_pickle('results_p8.pickle')
results_3R = hr_all.merge_all(results_p, n_samples=3)
results_3R.head()
# 着順が3着以内に入るかどうか学習
results_3R['rank'] = results_3R['着順'].map(lambda x: 1 if x < 4 else 0)
results_3R.drop(['着順', '騎手', 'horse_id', '馬名'], axis=1, inplace=True)
results_3R.columns
results_3R.info()
results_3R.head()

results_3R['rank'].value_counts()
results_d = pd.get_dummies(results_3R)
results_d.columns
results_d.info()
results_d.head()


def split_data(df, test_size=0.3):
    sorted_id_list = df.sort_values('date').index.unique()
    train_id_list = sorted_id_list[:round(
        len(sorted_id_list) * (1 - test_size))]
    test_id_list = sorted_id_list[round(
        len(sorted_id_list) * (1 - test_size)):]

    train = df.loc[train_id_list].drop(['date'], axis=1)
    test = df.loc[test_id_list].drop(['date'], axis=1)

    return train, test


train, test = split_data(results_d)
X_train = train.drop(['rank'], axis=1)
y_train = train['rank']

X_test = test.drop(['rank'], axis=1)
y_test = test['rank']

X_train.head()
X_train.info()
X_train.columns
X_train.head()

#############################勾配ブースティング木##################################
params = {
    'num_leaves': 5,
    'n_estimators': 70,
    # 'min_date_in_leaf': 500,
    'class_weight': "balanced",
    'random_state': 100
}
# モデル決定
lgb_clf = lgb.LGBMClassifier(**params)
# 学習
lgb_clf.fit(X_train.values, y_train.values)
# 結果表示(引数をTodayのものにする)
y_pred = lgb_clf.predict_proba(X_test)[:, 1]
[0 if p < 0.5 else 1 for p in y_pred]
y_pred_train
y_pred

print(roc_auc_score(y_train, y_pred_train))
print(roc_auc_score(y_test, y_pred))

importances = pd.DataFrame(
    {"features": X_train.columns, "importance": lgb_clf.feature_importances_})

importances.sort_values("importance", ascending=False)[:20]
'''

終わり
'''
