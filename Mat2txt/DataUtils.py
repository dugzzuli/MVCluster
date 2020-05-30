
import scipy.io as sio
import numpy as np
def load_BBC():
    dataMat=sio.loadmat('../Database/BBC/BBC.mat')
    data=dataMat['data']
    target=dataMat['truelabel']
    dataR=[]
    for i in range(data.size):
        dataR.append(np.transpose(data[0, i]))
    return dataR,target[0,0]

if __name__=='__main__':
    dataR,target=load_BBC()
    # for i in range(len(dataR)):
    #     dd=dataR[i]
    #     m,n=np.shape(dd)
    #     with open("./dataset/View"+str(i+1)+".txt",mode="w") as f:
    #         for k in range(m):
    #             f.write(str(k))
    #             for j in range(n):
    #                 f.write(" "+str(dd[k][j]))
    #             f.write("\n")
    #         f.flush()

    m,n=np.shape(target)
    with open("../Database/BBC/group.txt", mode="w") as f:
        for j in range(n):
            f.write(str(j)+" "+str(target[0][j]))
            f.write("\n")
        f.flush()

