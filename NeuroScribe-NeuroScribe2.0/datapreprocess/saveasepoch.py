import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
from braindecode.models import EEGNetv4
from skorch.callbacks import LRScheduler
from skorch.dataset import ValidSplit
from braindecode import EEGClassifier

from braindecode.datasets import (
    create_from_mne_raw, create_from_mne_epochs)

from braindecode.augmentation import FrequencyShift
from braindecode.augmentation import AugmentedDataLoader, SignFlip


syn=sio.loadmat('path')
data1=np.random.randn(5, 1000)
count1=0
data2=np.random.randn(5, 1000)
count2=0
data3=np.random.randn(5, 1000)
count3=0
my_list = []




with open('path', 'r') as f:

    lines = f.readlines()


    my_list = lines[0].split()


for i in range(450):
    print(i)
    if syn['BCI']['TrialData'][0,0]['tasknumber'][0,i][0,0]==1 and syn['BCI']['TrialData'][0,0]['result'][0,i][0,0]==1 and syn['BCI']['TrialData'][0,0]['artifact'][0,i][0,0]==0:
        count1 += 1
        middle=syn['BCI']['data'][0, 0][0, i][:,4000:syn['BCI']['TrialData'][0,0]['resultind'][0,i][0,0]]
        if count1 == 1:
            while middle.shape[1]<4000:
                middle=np.concatenate((middle, syn['BCI']['data'][0, 0][0, i][:,4000:syn['BCI']['TrialData'][0,0]['resultind'][0,i][0,0]]), axis=1)
            middle=middle[:,:4000]
            data1 = middle
        else:
            while middle.shape[1] < 4000:
                middle = np.concatenate((middle, syn['BCI']['data'][0, 0][0, i][:,
                                                 4000:syn['BCI']['TrialData'][0, 0]['resultind'][0, i][0, 0]]), axis=1)
            middle = middle[:, :4000]
            data1 = np.concatenate((data1, middle), axis=1)
    # elif syn['BCI']['TrialData'][0,0]['tasknumber'][0,i][0,0]==2 and syn['BCI']['TrialData'][0,0]['result'][0,i][0,0]==1 and syn['BCI']['TrialData'][0,0]['artifact'][0,i][0,0]==0:
    #     count2+=1
    #     if count2==1:
    #         data2=syn['BCI']['data'][0,0][0,i]
    #     else:
    #         data2 = np.concatenate((data2, syn['BCI']['data'][0,0][0,i]), axis=1)
    #
    # elif syn['BCI']['TrialData'][0,0]['tasknumber'][0,i][0,0]==3 and syn['BCI']['TrialData'][0,0]['result'][0,i][0,0]==1 and syn['BCI']['TrialData'][0,0]['artifact'][0,i][0,0]==0:
    #     count3+=1
    #     if count3==1:
    #         data3=syn['BCI']['data'][0,0][0,i]
    #     else:
    #         data3 = np.concatenate((data3, syn['BCI']['data'][0,0][0,i]), axis=1)



info = mne.create_info(
    ch_names=my_list,
    ch_types=['eeg'] * 62,
    sfreq=1000
)

raw = mne.io.RawArray(data1, info)


raw_data=raw.get_data()

mean_values = np.mean(raw_data, axis=1,keepdims=True)


std_values = np.std(raw_data, axis=1,keepdims=True)


std_values[std_values == 0] = 1


eeg_data_standardized = (raw_data - mean_values) / std_values

raw=mne.io.RawArray(eeg_data_standardized, info)



dim1=[]
endtime=[]
with open('your path', 'r') as f:
    lines = f.readlines()


    endtime = lines[0].split()
index=0
for i in range(len(endtime)):
    if i==0:
        dim1.append(4000)
        index=int(endtime[i])+1000#syn['BCI']['TrialData'][0,0]['resultind'][0,i][0,0]
    else:
        dim1.append(index+4000)
        index+=(int(endtime[i])+1000)#syn['BCI']['TrialData'][0,0]['resultind'][0,i][0,0]
dim1=np.arange(0, 70 * 4000, 4000)

dim2=[0]*len(endtime)

dim3 = []

for i in range(450):
    if syn['BCI']['TrialData'][0, 0]['result'][0, i][0, 0] == 1 and syn['BCI']['TrialData'][0,0]['tasknumber'][0,i][0,0]==1:
        dim3.append(syn['BCI']['TrialData'][0,0]['targetnumber'][0,i][0,0]-1)

mergelist=[]
mergelist.append(dim1)
mergelist.append(dim2)
mergelist.append(dim3)

mergearray=np.array(mergelist)

events=mergearray.T

event_id = dict(right=0, left=1)

epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=3.9991,baseline=(0,0),preload=True)


windows_dataset = create_from_mne_epochs(
    [epochs],
    window_size_samples=4000,
    window_stride_samples=4000,
    drop_last_window=False
)



model = EEGNetv4(
    n_chans=62,
    n_times=4000,
    n_outputs=2,
    final_conv_length='auto',
)

freq_shift = FrequencyShift(
    probability=1.,  # defines the probability of actually modifying the input
    sfreq=1000,
    max_delta_freq=2.  # the frequency shifts are sampled now between -2 and 2 Hz
)
sign_flip = SignFlip(probability=0.5)

transforms = [
    freq_shift,
    sign_flip
]

net=EEGClassifier(module=model,
# iterator_train=AugmentedDataLoader,  # This tells EEGClassifier to use a custom DataLoader
#     iterator_train__transforms=transforms,
             classes=[0,1],
             criterion=torch.nn.CrossEntropyLoss,
             optimizer=torch.optim.Adam,
             optimizer__lr=0.0005,
             train_split=ValidSplit(0.2),
             optimizer__weight_decay=0,
             batch_size=2,
             callbacks=[
                "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=599)),
            ],
             #verbose=self.verbose
             max_epochs=600
             )

net.fit(windows_dataset)