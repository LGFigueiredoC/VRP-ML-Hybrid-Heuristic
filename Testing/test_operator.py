import math
import time
import model
import csv
import numpy as np
import torch
from torch_geometric.data import Data
import vrplib
from conversor import Conversor
from model import Model
import aco_cvrp_cpp

class Test_operator:
    def __init__(self):
        pass


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
    

    def test (self, subset, file_name, test_config, model_name):

        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            for i in range (0, len(subset)-1, 2):
                sol_file = "test_set/"+subset[i]
                ins_file = "test_set/"+subset[i+1]
                solution = vrplib.read_solution(sol_file)
                instance = vrplib.read_instance(ins_file)
                print(subset[i+1])

                for j in range(5):
                    # conversion
                    conv_start = time.time()
                    model = Model(-1, 20, 5)
                    model.load_state_dict(torch.load(model_name, weights_only=True, map_location=test_config.device))
                    
                    c = Conversor(instance=instance, solution=solution)
                    data = c.convert_to_t_geometric()

                    gnnData  = Data(x=torch.tensor(data.x, dtype=torch.float64),edge_index=data.edge_index,
                                        edge_attr=torch.tensor(data["edge_attr"], dtype=torch.float64),
                                        y=torch.tensor(data.y, dtype=torch.float64))

                    output = model(gnnData)
                    probMatrix = Test_operator.getProbMatrix(output, gnnData.edge_index)
                    conv_end = time.time()

                    # aco
                    aco_start = time.time()
                    aco = aco_cvrp_cpp.ACO_CVRP(10, instance['dimension'], instance['capacity'],
                                                test_config.alpha, test_config.beta, test_config.Q,
                                                test_config.decay, 0, test_config.probNew, test_config.seed)
                    aco.init(instance["edge_weight"], instance["demand"], None)
                    [path_aco, cost_aco, it_aco] = aco.optimize(100, 25)
                    aco_end = time.time()
                    
                    # aco+gnn
                    model_start = time.time()
                    aco_gnn = aco_cvrp_cpp.ACO_CVRP(10, instance['dimension'], instance['capacity'],
                                                    test_config.alpha, test_config.beta, test_config.Q,
                                                    test_config.decay, 0, test_config.probNew, test_config.seed)
                    
                    aco_gnn.init(instance["edge_weight"], instance["demand"], probMatrix)
                    [path_gnn, cost_gnn, it_gnn] = aco.optimize(100, 25)
                    model_end = time.time()

                    # times; writing
                    aco_time = aco_end-aco_start
                    conv_time = conv_end-conv_start
                    model_time = model_end-model_start

                    # print("{} {} {} {} {}".format(subset[i+1], aco_time, conv_time, model_time, solution["cost"]))
                
                    writer.writerow([subset[i+1]] + [round(aco_time, 2)] + [it_aco] + [round(cost_aco, 2)] +
                        [round(conv_time, 2)] + [round(model_time, 2)] + [it_gnn] +
                        [round(cost_gnn, 2)] + [solution["cost"]])