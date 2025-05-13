import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.nn.functional as F
import time
from torch_geometric.nn import SAGEConv, GraphConv
import pickle

# Aqui está a definição das redes
# Nosso problema exige a utilização apenas de redes "especiais", as convolucionais.
# Em geral, rede neural "clássica" tem saída de tamanho fixo.
# As camadas convolucionais, não. A saída tem tamanho proporcional as entradas. São elas que devemos usar.
# para isso, segui os exemplos que citei lá em cima.
# Primeiro criei uma rede convolucional com camadas do tipo SAGEConv, e defini como que os dados passam pela rede.
# Porém, a SAGEConv não leva em consideração os pesos das arestas, então troquei por redes do tipo GraphConv
# (https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.nn.conv.GraphConv.html)
# que levam elas em consideração. Vou manter o nome "SAGE" por enquanto, mas depois seria bom alterar
# Ou ainda integrar tudo diretente na classe Model. Mas deixa isso pro futuro.

class SAGE(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = GraphConv(
            in_channels=in_feats, out_channels=hid_feats, aggr='mean')
        self.conv2 = GraphConv(
            in_channels=hid_feats, out_channels=out_feats, aggr='mean')

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        # h = F.relu(h)
        h = F.sigmoid(h)
        h = self.conv2(h, edge_index, edge_attr)
        return h

# Depois, criei a classe DotProductPredictor.
# Salvo pelo https://github.com/pyg-team/pytorch_geometric/discussions/3554
# Note que é ela que define que a saída da rede neural tem tamanho igual a (1,N_edges)
class DotProductPredictor(torch.nn.Module):
    def forward(self, x, edge_index):
        src, dst = edge_index
        score = (x[src] * x[dst]).sum(dim=-1)
        return score

# E finalmente o modelo da rede neural, que usa as classes SAGE e DotProductPredictor
class Model(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()
    def forward(self, graph):
        h = self.sage(graph.x, graph.edge_index, graph.edge_attr)
        return self.pred(h, graph.edge_index)
    