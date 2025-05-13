import pickle
# import aco_cvrp
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

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

torch.manual_seed(0) # Para reproduzir os resultados
device

torch.set_default_tensor_type(torch.DoubleTensor)

set = os.listdir("test_set")
set.sort()



with open('BestTestLoss.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(["set_name"] +["ACO_time"] + ["ACO_iterations"] + ["ACO_obj_func"] +
                    ["GCN_AS_conv_time"] + ["GCN_AS_exec_time"] + ["GCN_AS_iterations"] + 
                    ["GCN_AS_obj_func"] + ["Solution"])


    for i in range (0, 10, 2) :
        print("test_set/"+set[i+1])
        sol_file = "test_set/"+set[i]
        ins_file = "test_set/"+set[i+1]
        solution = vrplib.read_solution(sol_file)
        instance = vrplib.read_instance(ins_file)
        print(instance)
        for j in range(5):
            #roda o aco_vrp sozinho
            # aco = aco_cvrp.AntColony_CVRP(instance, 10, 1, 100, 0.95, max_no_improv=25, alpha=1, beta=1)

            aco_start = time.time()
            # shortest_path = aco.run()
            aco_end = time.time()

            aco_time = aco_end-aco_start
            # aco_cost = shortest_path[1]
            # aco_iterations = shortest_path[2]

            # print("- ACO\nTotal time spent: {} seconds\nShortest path cost: {}\n".format(aco_time, aco_cost))
            print(solution["cost"])

            
            print("- Best test loss model:\n")

            #inicia modelo
            val_loss_model = model.Model(-1, 20, 5)
            val_loss_model.load_state_dict(torch.load("2540_epocas", weights_only=True, map_location=device))
            val_loss_model.to(device)

            #inicia contagem de tempo, converte a instância, roda no modelo, altera a matriz de prob e joga no aco
            val_loss_start = time.time()

            c = Conversor(instance_file=ins_file, solution_file=sol_file)
            data = c.convert_to_t_geometric()

            gnnData = Data(x=torch.tensor(data.x, dtype=torch.float64),edge_index=data.edge_index,
                                edge_attr=torch.tensor(data["edge_attr"], dtype=torch.float64),
                                y=torch.tensor(data.y, dtype=torch.float64))


            output = val_loss_model(gnnData)
            probMatrix = getProbMatrix(output, gnnData.edge_index)
            print(probMatrix.dtype)
            
            val_loss_conv = time.time()

            # aco = aco_cvrp.AntColony_CVRP(instance, 10, 1, 100, 0.95,
            #                                 max_no_improv=25, alpha=1, beta=1, pheromone=probMatrix)
            # shortest_path = aco.run()

            val_loss_end = time.time()

            val_loss_conv_time = val_loss_conv-val_loss_start
            val_loss_aco_time = val_loss_end-val_loss_conv
            val_loss_time = val_loss_end-val_loss_start

            # val_loss_cost = shortest_path[1]
            # val_loss_iterations = shortest_path[2]

            # print("- GCN_AS\nTotal time spent: {} seconds\nShortest path cost: {}\n".format(val_loss_time, val_loss_cost))
            print(solution["cost"])

            #gera arquivo com os parâmetros desejados
            # writer.writerow([set[i+1]] + [round(aco_time,2)] + [aco_iterations] + [round(aco_cost,2)] + 
            #                 [round(val_loss_conv_time,2)] + [round(val_loss_aco_time, 2)] +
            #                     [val_loss_iterations] + [round(val_loss_cost,2)] + [solution["cost"]])