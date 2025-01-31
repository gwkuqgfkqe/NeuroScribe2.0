import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


syn=sio.loadmat('path')
data1=np.random.randn(5, 1000)
count1=0
data2=np.random.randn(5, 1000)
count2=0
data3=np.random.randn(5, 1000)
count3=0
for i in range(450):
    print(i)
    if syn['BCI']['TrialData'][0,0]['tasknumber'][0,i][0,0]==1 and syn['BCI']['TrialData'][0,0]['result'][0,i][0,0]==1 and syn['BCI']['TrialData'][0,0]['artifact'][0,i][0,0]==0:
        count1+=1

        with open('path', 'a') as f:

            for j in range(syn['BCI']['positionx'][0,0][0,i].shape[1]):
                syn['BCI']['positionx'][0,0][0,i]=np.nan_to_num(syn['BCI']['positionx'][0, 0][0, i], nan=-1)
                if syn['BCI']['positionx'][0,0][0,i][0,j]!=-1:
                    f.write(str(syn['BCI']['positionx'][0,0][0,i][0,j])+' ')
            f.write('\n')
            f.close()
        with open('path', 'a') as f:

            for j in range(syn['BCI']['positiony'][0, 0][0, i].shape[1]):
                syn['BCI']['positiony'][0, 0][0, i] = np.nan_to_num(syn['BCI']['positiony'][0, 0][0, i], nan=-1)
                if syn['BCI']['positiony'][0, 0][0, i][0, j] != -1:
                    f.write(str(syn['BCI']['positiony'][0, 0][0, i][0, j]) + ' ')
            f.write('\n')
            f.close()
    elif syn['BCI']['TrialData'][0,0]['tasknumber'][0,i][0,0]==2 and syn['BCI']['TrialData'][0,0]['result'][0,i][0,0]==1 and syn['BCI']['TrialData'][0,0]['artifact'][0,i][0,0]==0:
        count2+=1

        with open('path', 'a') as f:

            for j in range(syn['BCI']['positionx'][0, 0][0, i].shape[1]):
                syn['BCI']['positionx'][0, 0][0, i] = np.nan_to_num(syn['BCI']['positionx'][0, 0][0, i], nan=-1)
                if syn['BCI']['positionx'][0, 0][0, i][0, j] != -1:
                    f.write(str(syn['BCI']['positionx'][0, 0][0, i][0, j]) + ' ')
            f.write('\n')
            f.close()
        with open('path', 'a') as f:

            for j in range(syn['BCI']['positiony'][0, 0][0, i].shape[1]):
                syn['BCI']['positiony'][0, 0][0, i] = np.nan_to_num(syn['BCI']['positiony'][0, 0][0, i], nan=-1)
                if syn['BCI']['positiony'][0, 0][0, i][0, j] != -1:
                    f.write(str(syn['BCI']['positiony'][0, 0][0, i][0, j]) + ' ')
            f.write('\n')
            f.close()
    elif syn['BCI']['TrialData'][0,0]['tasknumber'][0,i][0,0]==3 and syn['BCI']['TrialData'][0,0]['result'][0,i][0,0]==1 and syn['BCI']['TrialData'][0,0]['artifact'][0,i][0,0]==0:
        count3+=1

        with open('path', 'a') as f:

            for j in range(syn['BCI']['positionx'][0, 0][0, i].shape[1]):
                syn['BCI']['positionx'][0, 0][0, i] = np.nan_to_num(syn['BCI']['positionx'][0, 0][0, i], nan=-1)
                if syn['BCI']['positionx'][0, 0][0, i][0, j] != -1:
                    f.write(str(syn['BCI']['positionx'][0, 0][0, i][0, j]) + ' ')
            f.write('\n')
            f.close()
        with open('path', 'a') as f:

            for j in range(syn['BCI']['positiony'][0, 0][0, i].shape[1]):
                syn['BCI']['positiony'][0, 0][0, i] = np.nan_to_num(syn['BCI']['positiony'][0, 0][0, i], nan=-1)
                if syn['BCI']['positiony'][0, 0][0, i][0, j] != -1:
                    f.write(str(syn['BCI']['positiony'][0, 0][0, i][0, j]) + ' ')
            f.write('\n')
            f.close()

