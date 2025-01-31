import matplotlib
from torch.optim import lr_scheduler, Adam

from braindecode.models import EEGNetv4,ATCNet
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2
import signatory
from sklearn.preprocessing import StandardScaler
import torch
from functools import partial
from datetime import datetime
import os
import numpy as np
from braindecode.datasets import (
    create_from_mne_raw, create_from_mne_epochs)
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)
import argparse
import json
import sys
sys.path.append("..") 
from ndp_nets.cnn_ndp_main import NeuroScribe
from imednet.imednet.data.data_loader import fif_txt_Loader

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='cnn-ndp-il')
parser.add_argument('--tasks', type=str, default='signature')
args = parser.parse_args()

# training on different synthetic dataset
dataset_name = 'smnist' # smnist, smnist-awgn, smnist-mb, smnist-rc-awgn
neuros = 500   # 200 or 500
data_path1 = '../data/LaoYang/ecog_epochs_25s.fif'#'../data/S1_Session1_3.fif'
data_path2 = '../data/S1_S.fif'
x_path='../data/LaoYang/x_25s.txt'
y_path='../data/LaoYang/y_25s.txt'
pre_trained = '../mnist_cnn/cnn_trained/mnist_cnn_net_' + str(neuros) +'(mnist).pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if_sep=0
if_psd=0
images1,  or_tr = fif_txt_Loader.load_data([data_path1],x_path,y_path,if_sep)
if if_psd:
    images1=images1.compute_psd(fmax=200)
for i in range(1404):
    or_tr[i]=or_tr[i]/6

# windows_dataset = create_from_mne_epochs(
#     [images1],
#     window_size_samples=4000,
#     window_stride_samples=4000,
#     drop_last_window=False
# )
#
#
#
# factor_new = 1e-3
# init_block_size = 1000
# sfreq = 1000
# # transforms = [
# #     Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
# #     Preprocessor(
# #         exponential_moving_standardize,  # Exponential moving standardization
# #         factor_new=factor_new,
# #         init_block_size=init_block_size,
# #     ),
# # ]
# #
# # # Transform the data
# # preprocess(windows_dataset, transforms)
#
# preprocessors = [Preprocessor('pick', picks=['eeg'])]
#
#
#
# preprocessors.append(Preprocessor(exponential_moving_standardize, factor_new=1e-3))
# preprocess(windows_dataset, preprocessors)





images = images1.get_data()

ecog_data_standardized = np.zeros_like(images)

# 遍历每个通道，对每个通道进行标准化
for channel in range(images.shape[1]):
    # 提取该通道的所有epoch数据（形状: epoch × samples）
    channel_data = images[:, channel, :]
    
    # 创建一个 StandardScaler 对象
    scaler = StandardScaler()
    
    # 对该通道的数据进行标准化
    # 注意：标准化是基于每个通道的所有epoch上的数据
    channel_data_standardized = scaler.fit_transform(channel_data)
    
    # 将标准化后的数据放回到标准化结果中
    ecog_data_standardized[:, channel, :] = channel_data_standardized

images= ecog_data_standardized
input_size = images.shape[1] * images.shape[2]

inds = np.arange(1404)
#np.random.shuffle(inds)
test_inds = inds[1200:]
train_inds = inds[:1200]
X = torch.Tensor(images[:1404]).float()  # [70, 62, 4000]
Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[:1404]
X, Y = X.to(device), Y.to(device)
time = str(datetime.now())
time = time.replace(' ', '_')
time = time.replace(':', '_')
time = time.replace('-', '_')
time = time.replace('.', '_')
model_save_path = '../EEG_ndp_models/' + args.name

# hyper parameters
k = 1
T = 199 / k
N = 20
depth=2
learning_rate = 0.001
num_epochs = 1000
batch_size = 60
signature=0
param_str = "T" + str(T) + "_K" + str(k) + "_N" + str(N) + "_L" + str(learning_rate) + "_E" + str(num_epochs) + "_B" + str(batch_size)
model_save_path = model_save_path+'_'+args.tasks + '_' + '(' + dataset_name + ')_(' + param_str + ')_(' + time + ')'
os.mkdir(model_save_path)

image_save_path = model_save_path + '/images'
os.mkdir(image_save_path)

# data sets
Y = Y[:, ::k, :]
X_train = X[train_inds]
Y_train = Y[train_inds]
X_test = X[test_inds]
Y_test = Y[test_inds]

# the ndp_cnn model
ndpn = NeuroScribe(T=T, l=1, N=N, state_index=np.arange(2),psd_f_size=images.shape[2])
#getgnet=EEGNetv4(n_chans=62,n_outputs=2,n_times=4000)
print(ndpn)
ndpn.to(device)
#getgnet.to(device)criterion_reg = nn.MSELoss() 
criterion_reg = nn.MSELoss() 
optimizer = torch.optim.Adam(ndpn.parameters(), lr=learning_rate)
#scheduler=lr_scheduler.ExponentialLR(optimizer, gamma=0.999) 
loss_values = []
# training process
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=19999, eta_min=0.0000001)
# optimizer2 = torch.optim.Adam(getgnet.parameters(), lr=learning_rate)
# scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer2, T_max=27999, eta_min=0.000001)

def euclidean_distance(p1, p2):
    """计算两点之间的欧氏距离"""
    return np.sqrt(np.sum((p1 - p2) ** 2))

def dtw(traj1, traj2, dist=euclidean_distance):
    """
    计算两个轨迹之间的DTW距离。
    
    参数:
    traj1, traj2 -- 形状为(n, 2)的numpy数组，表示两个轨迹，其中n是点的数量，2是xy坐标。
    dist -- 一个函数，用于计算两个点之间的距离，默认为欧氏距离。
    
    返回:
    dtw_distance -- 两个轨迹之间的DTW距离。
    """
    n, m = len(traj1), len(traj2)
    dtw_matrix = np.zeros((n+1, m+1))
    
    # 初始化DTW矩阵的边界
    for i in range(n+1):
        dtw_matrix[i, 0] = float('inf')
    for j in range(m+1):
        dtw_matrix[0, j] = float('inf')
    dtw_matrix[0, 0] = 0
    
    # 填充DTW矩阵
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist(traj1[i-1], traj2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # 插入
                                         dtw_matrix[i, j-1],    # 删除
                                         dtw_matrix[i-1, j-1])  # 匹配
    
    # DTW距离是矩阵右下角的值
    return dtw_matrix[n, m]

for epoch in range(num_epochs):
    inds = np.arange(X_train.shape[0])
    #np.random.shuffle(inds)
    ndpn.train()
    #getgnet.train()
    count=0
    for ind in np.split(inds, len(inds) // batch_size):
        # goal=getgnet(X_train[ind])
        # g = goal.cpu().detach().numpy()
        # Goal=torch.Tensor(g).to(device)
        # loss2=torch.mean((goal - Y_train[ind, -1, :]) ** 2)
        # loss2.backward()
        # optimizer2.step()
        # scheduler2.step()
        # optimizer2.zero_grad()

        
        # output y

        y_h = ndpn(X_train[ind], Y_train[ind, 0, :])  # [100, 301, 2]
        #y_h=torch.tensor(y_h).to(device)
        loss_display = (y_h - Y_train[ind]) ** 2
        #print(y_h.shape,Y_train[ind].shape,Y_train.shape)
        
        abs_diff = torch.abs(y_h - Y_train[ind])
        #abs_diff[:,:,1]=0
        abs_diff=torch.mean(abs_diff)
        loss = torch.where(abs_diff < 1, 0.5 * abs_diff ** 2, abs_diff - 0.5)
        sig_h=torch.zeros(batch_size,60)
        sig_tra=torch.zeros(batch_size,60)
        if signature==1:
            for k in range(10):
                #if k==0:
                sig_h[:,k*6:(k+1)*6]=signatory.signature(y_h[:,k*20:(k+1)*20,:], depth)
                sig_tra[:,k*6:(k+1)*6]= signatory.signature(Y_train[ind][:,k*20:(k+1)*20,:], depth)
#                 else:
#                     sig_h+=signatory.signature(y_h[:,k*20:(k+1)*20,:], depth)
#                     sig_tra+= signatory.signature(Y_train[ind][:,k*20:(k+1)*20,:], depth)
#             sig_h = signatory.signature(y_h, depth)
#             #print(sig_h)
#             sig_tra= signatory.signature(Y_train[ind], depth)
#             sig_h=sig_h/10
#             sig_tra=sig_tra/10
#             loss_display=torch.mean((sig_h - sig_tra)**2)
#             loss=loss_display
            loss=criterion_reg(sig_h, sig_tra)
        #loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        optimizer.zero_grad()
        
        
        count+=1
        print('Epoch: '+str(epoch)+'    '+'Data: '+str(count)+'    '+'Loss: '+str(float(loss))+'    '+'Lr: '+str(scheduler.get_lr()))

#str(float(loss_display.mean(1).mean(1).mean().item()))
    # exampes for 0-9
    if (epoch+1)%1000==0:
        ndpn.eval()
        
        #getgnet.eval()
        #for j in range(48):
        test_sample_indices = test_inds#[:50]
        train_sample_indices=train_inds[:10]#[100:150]
        # goal_t=getgnet(X[test_sample_indices])
        y_ht = ndpn(X[test_sample_indices], Y[test_sample_indices, 0, :])
        #print(y_ht.shape)
        #y_ht=torch.tensor(y_ht).to(device)
        y_rt = Y[test_sample_indices]
        #goal_e=getgnet(X[train_sample_indices])
        y_he = ndpn(X[train_sample_indices], Y[train_sample_indices, 0, :])
        #y_he=torch.tensor(y_he).to(device)
        y_re = Y[train_sample_indices]
        for i in range(0, len(test_sample_indices)):
            plt.figure(figsize=(8, 8))
            plt.tight_layout()
            #image = images1[test_sample_indices[i]]
            #H, W = image.shape
            #plt.imshow(image, cmap='gray', extent=[0, H + 1, W + 1, 0])
            plt.plot(y_rt[i, :, 0].detach().cpu().numpy(), y_rt[i, :, 1].detach().cpu().numpy(), c='#DB70DB',alpha=0.7, linewidth=5,label='Original Trajectory')
            plt.plot(y_ht[i, :, 0].detach().cpu().numpy(), y_ht[i, :, 1].detach().cpu().numpy(), c='blue',alpha=0.7, linewidth=5,label='Generated Trajectory')
            # plt.xticks([ (i - 1) * 0.05 for i in range(1, 21)])
            # plt.yticks([ (i - 1) * 0.05 for i in range(1, 21)])
            # plt.xticks(range(1,20,1))
            # plt.yticks(range(1,20,1))
            plt.legend()
            plt.xlabel('X')
            plt.ylabel('Y')
    
            #plt.show()
            plt.savefig(image_save_path + '/valid_img_' + str(epoch) + '_' + str(i) + '.png')
            plt.close('all')
        for i in range(0, len(train_sample_indices)):
            plt.figure(figsize=(8, 8))
            plt.tight_layout()
            #image = images1[test_sample_indices[i]]
            #H, W = image.shape
            #plt.imshow(image, cmap='gray', extent=[0, H + 1, W + 1, 0])
            plt.plot(y_re[i, :, 0].detach().cpu().numpy(), y_re[i, :, 1].detach().cpu().numpy(), c='#DB70DB',alpha=0.5, linewidth=5,label='Original Trajectory')
            if (epoch+1)%1000==0:
                with open(model_save_path + '/X.txt', 'a') as f:
                    # 写入一些内容到文件
                    for j in list(y_he[i, :, 0].detach().cpu().numpy()):
            
                        f.write(str(j) + ' ')
                    f.write('\n')
                    f.close()
                with open(model_save_path + '/Y.txt', 'a') as f:
                    # 写入一些内容到文件
                    for j in list(y_he[i, :, 1].detach().cpu().numpy()):
            
                        f.write(str(j) + ' ')
                    f.write('\n')
                    f.close()
                with open(model_save_path + '/DWT.txt', 'a') as f:
    
            
                    f.write(str(dtw(y_he[i, :,:].detach().cpu().numpy(), y_re[i, :, :].detach().cpu().numpy())) )
                    f.write('\n')
                    f.close()
            plt.plot(y_he[i, :, 0].detach().cpu().numpy(), y_he[i, :, 1].detach().cpu().numpy(), c='blue',alpha=0.5, linewidth=5,label='Generated Trajectory')
            # plt.xticks(range(1,20,1))
            # plt.yticks(range(1,20,1))
            plt.legend()
            plt.xlabel('X')
            plt.ylabel('Y')
    
            #plt.show()
            plt.savefig(image_save_path + '/train_img_' + str(epoch) + '_' + str(i) + '.png')
            plt.close('all')

    # for i in range(0, len(test_sample_indices)):
    #     plt.figure(figsize=(8, 8))
    #     plt.tight_layout()
    #     # image = images1[test_sample_indices[i]]
    #     # H, W = image.shape
    #     # plt.imshow(image, cmap='gray', extent=[0, H + 1, W + 1, 0])
    #     plt.plot(y_h[i, :, 0].detach().cpu().numpy(), y_h[i, :, 1].detach().cpu().numpy(), c='r', linewidth=3)
    #     plt.axis('off')
    #     plt.savefig(image_save_path + '/valid_img_r_' + str(epoch) + '_' + str(i) + '.png')

    # # if epoch % 2 == 0:
    x_test = X_train[np.arange(10)]
    y_test = Y_train[np.arange(10)]
    #goal_fe=getgnet(x_test)
    y_htest = ndpn(x_test, y_test[:, 0, :])
    #y_htest=torch.tensor(y_htest).to(device)
    # for j in range(10):
    #     plt.figure(figsize=(8, 8))
    #     plt.plot(0.667 * y_h[j, :, 0].detach().cpu().numpy(), -0.667 * y_h[j, :, 1].detach().cpu().numpy(), c='r', linewidth=5)
    #     plt.axis('off')
    #     plt.savefig(image_save_path + '/train_img_' + str(epoch) + '_' + str(j) + '.png')
    #
    #     plt.figure(figsize=(8, 8))
    #     img = X_train[ind][j].cpu().numpy() * 255
    #     img = np.asarray(img * 255, dtype=np.uint8)
    #     plt.imshow(img, cmap='gray')
    #     plt.axis('off')
    #     plt.savefig(image_save_path + '/ground_train_img_' + str(epoch) + '_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
    #
    #     plt.figure(figsize=(8, 8))
    #     plt.plot(0.667 * y_htest[j, :, 0].detach().cpu().numpy(), -0.667 * y_htest[j, :, 1].detach().cpu().numpy(), c='r', linewidth=5)
    #     plt.axis('off')
    #     plt.savefig(image_save_path + '/test_img_' + str(epoch) + '_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
    #
    #     plt.figure(figsize=(8, 8))
    #     img = X_test[j].cpu().numpy() * 255
    #     img = np.asarray(img * 255, dtype=np.uint8)
    #     plt.imshow(img, cmap='gray')
    #     plt.axis('off')
    #     plt.savefig(image_save_path + '/ground_test_img_' + str(epoch) + '_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

    test = ((y_htest - y_test) ** 2).mean(1).mean(1)
    # loss = torch.mean((y_h - Y_train[ind]) ** 2)  # loss value
    print('Epoch: ' + str(epoch) + ', Test Error: ' + str(test.mean().item()))
    loss_values.append(str(test.mean().item()))
    #torch.save(ndpn, model_save_path + '/cnn-model.pt')

# write value to file
with open(model_save_path + '/test_loss.txt', 'w') as f:
    f.write(json.dumps(loss_values))
