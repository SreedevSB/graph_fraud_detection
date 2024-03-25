import pandas as pd
import numpy as nup
import seaborn as sns
import matplotlib.pyplot as plt

!pip install networkx

import networkx as nx
from scipy.io import loadmat
import scipy.sparse as sp
from collections import defaultdict
import pickle
!wget https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip

!unzip YelpChi.zip
  
def sparse_to_adjlist(sp_matrix, filename):

  """
  Transfer sparse matrix to adjacency list
  :param sp_matrix: the sparse matrix
  :param filename: the filename of adjlist
  """

  # add self loop
  homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])

  # create adj_list
  adj_lists = defaultdict(set)
  edges = homo_adj.nonzero()

  for index, node in enumerate(edges[0]):
    adj_lists[node].add(edges[1][index])
    adj_lists[edges[1][index]].add(node)

  with open(filename, 'wb') as file:
    pickle.dump(adj_lists, file)
  file.close()

  

yelp = loadmat("/content/YelpChi.mat")

prefix=''

net_rur = yelp['net_rur']
net_rtr = yelp['net_rtr']
net_rsr = yelp['net_rsr']
yelp_homo = yelp['homo']

sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')


yelp['label'].shape
labels = yelp['label'].flatten()
feat_data = yelp['features'].todense().A

# load the preprocessed adj_lists
with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
  homo = pickle.load(file)

file.close()

with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
  relation1 = pickle.load(file)

file.close()

with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
  relation2 = pickle.load(file)

file.close()

with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
  relation3 = pickle.load(file)

file.close()

print ([homo, relation1, relation2, relation3], feat_data, labels)



type(homo),len(homo)
type(relation1),len(relation1)
type(feat_data),len(feat_data)
type(labels),len(labels)
