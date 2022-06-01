import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
# root_data ='Y:\Desktop\python_code/reg3d_and_seg_V1\logs\LPBA/baseline_LPBA_mutilevel_sim_smooth/train_loss.txt'
root_data ='C:/Users/lab503/Desktop/100.txt'
#root_data ='./train_log.txt'
def all_loss():
    with open(root_data) as f:
        lines = f.readlines()
        num= len(lines)
        new_NUm = np.arange(0,100,1)
        all = []
        print(len(lines))
        for line in lines:
            batch =  line.split(',',4)[1].split('|',2)[0].split(' ',3)[2]

            if int(batch) == 0:
                print(batch)
                loss = line.split(',',4)[2].split('||',8)[0].split(':',2)[1]
                all.append(float(loss))
                print('meiyihang',loss)
        print(len(all))
        plt.plot(new_NUm, all[0:100], marker='o', markerfacecolor='blue', markersize=5)
        plt.savefig('./demo.png')

def aff():
    with open(root_data) as f:
        lines = f.readlines()
        num = len(lines)
        new_NUm = np.arange(0, 100, 1)
        all = []
        print(len(lines))
        for line in lines:
            batch = line.split(',', 4)[1].split('|', 2)[0].split(' ', 3)[2]
            if int(batch) == 0:
                print(batch)
                loss = line.split(',', 4)[2].split('||', 8)[5].split(':', 2)[1].split('\'',2)[0]
                all.append(float(loss))
                print('meiyihang', loss)
        print(len(all))
        plt.plot(new_NUm, all[0:100], marker='o', markerfacecolor='blue', markersize=5)
        plt.savefig('./demo.png')
def val():
    with open(root_data) as f:
        lines = f.readlines()
        num = len(lines)
        new_NUm = np.arange(0, 100, 1)
        all = []
        print(len(lines))
        for line in lines:

            loss = line
            print(loss)
            all.append(float(loss))
            print('meiyihang', loss)
        print(len(all))
        plt.plot(new_NUm, all[0:100], marker='o', markerfacecolor='blue', markersize=5)
        plt.savefig('./demo.png')
def eval_reult():
    with open(root_data) as f:
        lines = f.readlines()
        num = len(lines)
        print(num)
        all = []
        print(len(all))
        iou=[]
        label=[]
        i=4
        for line in lines:
            #Dice =  line.split('Dice',2)[1].split('IOU')[0]
            #all.append(float(Dice))

            iou_dice =  line.split('IOU_dice',2)[1].split('Time',2)[0].split(',',5)[i]
            iou.append(float(iou_dice))
            label_dice = line.split('Dice', 2)[1].split('IOU_dice', 2)[0].split(',', 6)[i]

            label.append(float(label_dice))
        print(np.mean(iou))
        print(np.mean(label))
def eval_reult_csv():
    lambda1_path = 'U:\python_code/UGATIT-pytorch/11\logs\prwnet_muti_lever_2\epoch57'
    path_list = glob.glob(os.path.join(lambda1_path, '*dic.csv'))
    print(len(path_list))
    num = 0
    sum_dice = np.zeros((380,5))
    for path in path_list:
        csvFile = open(path, "r")
        reader = csv.reader(csvFile)
        rows = [row for row in reader]
        data = np.array(rows)
        print(data[1])
        dice = data[1]
        print(dice)

        sum_dice[num] = dice
        num += 1
    mean_dice = np.mean(sum_dice, axis=0)
    print(mean_dice)

    #return mean_dice
def evalwnet_reult_csv():
    lambda1_path = 'Z:\python_code\join_reg_and_seg\logs\MR/baseline_mr_ncc\epoch106'
    path_list = glob.glob(os.path.join(lambda1_path, '*sub_dic.csv'))
    print(len(path_list))
    num = 0
    sum_dice = np.zeros((25,3))
    for path in path_list:
        csvFile = open(path, "r")
        reader = csv.reader(csvFile)
        rows = [row for row in reader]
        data = np.array(rows)
        print(data[0])
        dice = data[1]
        print(dice)

        sum_dice[num] = dice
        num += 1
    mean_dice = np.mean(sum_dice, axis=0)
    print(np.std(sum_dice, axis=0, ddof=1))
    print(mean_dice)

if __name__ == '__main__':
    # eval_reult()
    evalwnet_reult_csv()
    #eval_reult_csv()
