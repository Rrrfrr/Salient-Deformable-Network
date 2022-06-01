import glob
import numpy as np
import csv
import os
def eval_reult_csv():
    lambda1_path = 'G:\data_fubu/fubu_processed\abdominal_ct_160\img'
    lambda1_path = 'Z:\Desktop\python_code\join_reg_and_seg\logs\Fubu/baseline_fubu_0818\epoch53'
    path_list = glob.glob(os.path.join(lambda1_path, '*dic.csv'))
    print(len(path_list))
    num = 0
    sum_dice = np.zeros((90,13))
    for path in path_list:

        csvFile = open(path, "r")
        reader = csv.reader(csvFile)
        rows = [row for row in reader]
        data = np.array(rows)

        dice = data[1]
        print(dice)

        sum_dice[num] = dice
        num += 1
    mean_dice = np.mean(sum_dice, axis=0)
    print('mean',mean_dice)
eval_reult_csv()