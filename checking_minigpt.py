import os
import json 

data_root = "/raid/t1/scratch/vikrant.dewangan/datas/"
json_list = os.listdir(data_root)

cnt = 0
gg = []

for filename in json_list:

    flg=0
    data = []

    if os.path.exists(data_root+f'/{filename}/answer_pred_both.json'):
        f = open(data_root+f'/{filename}/answer_pred_both.json')
        data = json.load(f)

        for obj in data:
            kks = list(obj.keys())
            for kk in kks:
                if 'mini' in kk.lower():
                    flg=1

        if flg:
            print(filename)

        f.close()
