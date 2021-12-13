import pickle as pkl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def smooth(data, sm=1):
    smooth_data = data 
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")

            smooth_data.append(d)

    return smooth_data


if __name__ == "__main__":
    aid =[0,1]
    num=[10,1000000]
    file = 'DDPG_H//DDPG_H_10_0.001_{}_{}_10.pkl'
    record = []
    N = 300
    name = ['REDQ','REDQ+H']
    linestyle = ['-', '--', ':', '-.']
    color = ['r', 'g', 'b', 'k']
    fig = plt.figure()
    for n in num:
        dd = []
        for a in aid:
            with open(file.format(a, n),'rb') as f:
                pp = np.array(pkl.load(f))[:N,0]
                # print(pp.shape)
            if n == 10:
                print(pp)
            dd.append(pp)
        dd = np.array(dd)
        print(dd.shape)
        record.append(dd)
    record = np.array(record)
    print(record.shape)
    _, ax = plt.subplots()
    xrange = [i*5 for i in range(N)]
    for i in range(len(name)):
        # print(record[i].shape)
        sns.tsplot(time=xrange, data=smooth(record[i], sm=1),condition=name[i],color=color[i], linestyle=linestyle[i])
    plt.legend()
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Evaluation')
    ax.grid(axis='y')
    plt.savefig('result.png')

