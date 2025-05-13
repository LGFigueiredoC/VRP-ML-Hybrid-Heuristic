import aco_cvrp_cpp
import time
import vrplib
import model
import torch
import csv
from torch_geometric.data import Data
from conversor import Conversor
import os
import numpy as np
import math

def getProbMatrix (matrix, edge_index):
    # tamanho do edge index é n*n - n, que é menor que n*n e maior que (n-1)*(n-1),
    # portanto tira-se a raiz do tamanho do edge index, pega o valor inteiro, que será n-1,
    # e soma 1
    n = int(math.sqrt(len(edge_index[0]))//1) + 1

    probMatrix = np.zeros(n*n)
    probMatrix = probMatrix.reshape(n, n)

    #atribuição da matriz
    for i in range (probMatrix.size-n):
        probMatrix[edge_index[0][i]][edge_index[1][i]] = abs(matrix[i])    

    #normalização
    for i in range(probMatrix.shape[0]):
        soma = probMatrix.sum(axis=1)[i]
        for j in range(probMatrix.shape[1]):
            probMatrix[i][j] /= soma;

    #transformação do tipo np.array para list do python
    return probMatrix



nAnts = 10  # Nùmero formigas
nCities = 16 #Número Cidades
capac = 35 # Capacidade de cada veículo
alpha = 0.5 # Parâmetro do ACO
beta = 0.8 # Parâmetro do ACO
q = 1 # Parâmetro do ACO (esse é o único que não tinha na outra implementação)
rho = 0.05 # Parâmetro do ACO - (1 - taxa de decaimento)
initCity = 0 # Cidade inicial
probNew = 0.1 # Probabilidade dele buscar uma rota nova
seed = 0 # Semente aleatória


with open('BestTestLoss.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(["set_name"] +["ACO_time"] + ["ACO_iterations"] + ["ACO_obj_func"] +
                    ["GCN_AS_conv_time"] + ["GCN_AS_exec_time"] + ["GCN_AS_iterations"] + 
                    ["GCN_AS_obj_func"] + ["Solution"])


