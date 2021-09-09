import pandas as pd
import time
from tqdm import tqdm as tqdm

url = "https://db.netkeiba.com/race/201901010101"
pd.read_html(url, header=0)[0]
# 検証のTableタグは全て取れる

# 01:札幌, 02:函館.......


def sclape_race_results(race_id_list, pre_race_results={}):
    race_results = {}
    race_results = pre_race_results
    for race_id in tqdm(race_id_list):
        # if race_id in race_results.keys():
        #     continue
        try:
            url = 'https://db.netkeiba.com/race/' + race_id
            race_results[race_id] = pd.read_html(url, header=0)[0]
            time.sleep(1)
        except IndexError:
            continue
        except:
            # for文から抜け出す
            break
    return race_results


race_id_list = ['20190201010101', '20190201010102', '20190201010103']
test2 = sclape_race_results(race_id_list)
test2

race_id_list = []
for place in range(1, 11, 1):
    for kai in range(1, 6, 1):
        for day in range(1, 9, 1):
            for r in range(1, 13, 1):
                race_id = '2019' + \
                    str(place).zfill(2) + str(kai).zfill(2) + \
                    str(day).zfill(2) + str(r).zfill(2)
                race_id_list.append(race_id)

# len(race_id_list)

test3 = sclape_race_results(race_id_list)
test3['201901010102']

for key in test3.keys():
    test3[key].index = [key] * len(test3[key])
test3['201901010304']

results = pd.concat([test3[key] for key in test3.keys()], sort=False)
results

results.to_pickle('results.pickle')

pd.read_pickle('results.pickle')
