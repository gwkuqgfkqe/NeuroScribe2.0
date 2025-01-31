
import numpy as np
import mne




def sample_evenly(arr, sep,num_samples=200):
    if sep==1:
        return arr[:150]
    N = len(arr)
    if N==200:
        return arr
    if N < num_samples:
        arr1=[]
        x=200/N
        y=200%N
        count=0
        if int(x)>1:
            i=0
            while i<200:
                if i-count<y:
                    for j in range(int(x)+1):
                        arr1.append(arr[i-count])
                        i += 1
                        count+=1
                    count-=1

                else:
                    for j in range(int(x)):
                        arr1.append(arr[i-count])
                        i += 1
                        count += 1
                    count -= 1
        else:
            i=0
            while i<200:
                if i-count<y:
                    for j in range(int(x)+1):
                        arr1.append(arr[i-count])
                        i += 1
                        count += 1
                    count -= 1
                else:

                    arr1.append(arr[i-count])
                    i += 1


        return arr1

    step = int(N / num_samples)

    samples = []

    for i in range(0, N, step):
        samples.append(arr[i])
    samples=np.array(samples)
    if len(samples)>200:
        samples=samples[:200]


    return samples


class fif_txt_Loader:
    def load_data(fileepoch,filex,filey,sep):

        epochs=mne.read_epochs(fileepoch[0])
        
        for i in range(len(fileepoch)):
            if i>0:
                epochs1 = mne.read_epochs(fileepoch[i])
                epochs=mne.concatenate_epochs([epochs,epochs1])

        length=epochs.events.shape[0]
        

        epochs.filter(l_freq=8,h_freq=30)


        trix=[]
        with open(filex, 'r') as f:
            count1 = 0
            lines = f.readlines()
            for line in lines:

                l1=line.split()
                count2=0
                for i in l1:
                    l1[count2]=float(i)
                    count2+=1
                lines[count1]= np.array(l1)
                count1+=1
            trix=lines
        count=0
        for i in trix:
            trix[count]=sample_evenly(i,sep=sep)
            count+=1

        triy = []
        with open(filey, 'r') as f:
            count1 = 0
            lines = f.readlines()
            for line in lines:

                l1 = line.split()
                count2 = 0
                for i in l1:
                    l1[count2] = float(i)
                    count2 += 1
                lines[count1] = np.array(l1)
                count1 += 1
            triy = lines
        count = 0
        for i in triy:
            triy[count] = sample_evenly(i,sep=sep)
            count += 1
        trajectory=[]
        for i in range(length):
            x=trix[i]
            y=triy[i]
            xy=np.concatenate((np.array(x).reshape(-1,1), np.array(y).reshape(-1,1)), axis=1)
            trajectory.append(xy)



        return epochs,  trajectory
