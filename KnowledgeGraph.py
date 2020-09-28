import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import networkx
import csv
from numpy import genfromtxt
import math
import time
def get_adj_matrix(graphA,graphR,threshold=0.6):
  A = torch.FloatTensor(graphA)
  R = torch.FloatTensor(graphR)
  nA = F.normalize(A)
  nR = F.normalize(R)
  addAR = nA + nR
  graphOUT = addAR > threshold
  graphOUT= graphOUT.type(torch.FloatTensor)
  return Variable(graphOUT,requires_grad=False)

def get_base_graph(graph, novel_list):
    for i in range(len(novel_list)):
        graph=np.delete(graph,novel_list[i],axis=0)
        graph=np.delete(graph,novel_list[i],axis=1)
    baseGraph=torch.from_numpy(graph)
    #baseGraph=Variable(baseGraph, requires_grad=False)
    return baseGraph

def get_total_graph(in_graph, novel_list):
  #in_graph=cls_r_prob_np
  reversed_list = list(reversed(novel_list))  # 2,5,9,13,17
  arrow = list()
  for i in range(20):
    arrow.append(i)
  for i in range(len(reversed_list)):
    arrow.remove(reversed_list[i])
    arrow.append(reversed_list[i])
  out_graph = in_graph[arrow][:, arrow]
  out_graph=torch.from_numpy(out_graph)
  #out_graph=Variable(out_graph,requires_grad=False)
  return out_graph

def iterRowColumnNormalize(in_graph, numIter):
    for i in range(numIter):
        in_graph=F.normalize(in_graph,p=1)
        in_graph=F.normalize(in_graph.permute(1,0),p=1)
    return in_graph



cls_a_prob_np=np.load('vocGraphA.npy')
cls_r_prob_np=np.load('vocGraphR.npy')
cls_r_prob=Variable(torch.from_numpy(cls_r_prob_np), requires_grad=False)



novel_id = [17, 13, 9, 5, 2]
total_a_prob = get_total_graph(cls_r_prob_np, novel_id)
base_a_prob = get_base_graph(cls_r_prob_np, novel_id)
total_r_prob = get_total_graph(cls_r_prob_np, novel_id)
base_r_prob = get_base_graph(cls_r_prob_np, novel_id)

adj=(torch.randn(10,10))
t0=time.time()
test=iterRowColumnNormalize(adj,999)
t1=time.time()
test2=iterRowColumnNormalize(adj,1)
t2=time.time()
x1=t1-t0
x2=t2-t1
print(x1)
print(x2)
print(x1/x2)
#for i in range(test.size(0)):
#    print(torch.sum(test2[i]))
test=test.numpy()
test2=test2.numpy()

def buildKGfromGLOVE():
    words=genfromtxt('word_glo.csv',delimiter=',')
    wordsC=torch.from_numpy(words)
    wordsC=wordsC.permute(1,0)
    wordsV=torch.zeros(20,300)

    wordsV[0]=wordsC[65]
    wordsV[1]=wordsC[1]
    wordsV[2]=wordsC[11]
    wordsV[3]=wordsC[6]
    wordsV[4]=wordsC[31]
    wordsV[5]=wordsC[4]
    wordsV[6]=wordsC[2]
    wordsV[7]=wordsC[68]
    wordsV[8]=wordsC[45]
    wordsV[9]=wordsC[15]
    wordsV[10]=wordsC[49]
    wordsV[11]=wordsC[12]
    wordsV[12]=wordsC[13]
    wordsV[13]=wordsC[3]
    wordsV[14]=wordsC[0]
    wordsV[15]=wordsC[47]
    wordsV[16]=wordsC[14]
    wordsV[17]=wordsC[46]
    wordsV[18]=wordsC[66]
    wordsV[19]=wordsC[50]

    voc=wordsV
    Vp= voc.numpy()
    np.save('vocWV_raw.npy', Vp)

    coco=wordsC

    #for i in range(20):
    #    for j in range(20):
    #        Vout[i][j]= torch.dist(voc[i],voc[j])

    #for k in range(20):
    #    Vout2[k]=(0.4)**(Vout[k]-torch.min(Vout[k]+100*torch.eye(20,20)))
    nV=F.normalize(voc)
    nC=F.normalize(coco)
    Cout=torch.mm(nC,nC.permute(1,0))
    Vout=torch.mm(nV,nV.permute(1,0))
    buf=Vout.numpy()
    #Vout-=torch.eye(20,20)
    Vp = torch.exp(2*(Vout-1))
    Cp = torch.exp(5*(Cout - 1))

    #for k in range(20):
    #   #Vout[k]=(4)**(Vout[k]-torch.min(Vout[k]+100*torch.eye(20,20)))
    #    Vout[k]=torch.exp(Vout[k])
    #Vp=torch.exp(Vout)
    #Vp=Vp/math.exp(1)
    #Cp=torch.exp(Cout)
    #Cp=Cp/math.exp(1)

    for i in range(20):
        for j in range(20):
            if Vp[i][j]<0.2:
                Vp[i][j]= 0
#
    #for i in range(80):
    #    for j in range(80):
    #        if Cp[i][j]<0.4:
    #            Cp[i][j]= 0
    vp=Vp.numpy()
    cp=Cp.numpy()
    np.save('vocWV2.npy',Vp)
    #np.save('cocoWV.npy', Cp)

    cls_r_prob_np = np.load('vocGraphR.npy')
    cls_r_prob = (torch.from_numpy(cls_r_prob_np))
    #Vp=torch.from_numpy(Vp)

    #for i in range(20):
    #    print(torch.sum(Vp[i]))
    #print('wv')
    #for i in range(20):
    #    print(torch.sum(cls_r_prob[i]))
    #print('genome')


buildKGfromGLOVE()
#print('done')

