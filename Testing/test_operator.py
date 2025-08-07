import math
import time
from modeler import *
import csv
import numpy as np
import torch
from torch_geometric.data import Data
import vrplib
from conversor import Conversor
import aco_cvrp_cpp
import pickle

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
            probMatrix[edge_index[0][i]][edge_index[1][i]] = matrix[i]    

        #normalização
        normalized_vec = np.zeros(n)
        for i in range(probMatrix.shape[0]):
            normalized_vec[i] = 1/(probMatrix.sum(axis=1)[i])
        
        probMatrix *= normalized_vec

        #transformação do tipo np.array para list do python
        return probMatrix
    

    def test (self, subset, file_name, test_config, model_name):        
        parameters = get_model_parameters(model_name)

        load_start = time.time()
        model = get_model(parameters[0], parameters[1], parameters[2], parameters[3])
        #print("pos modelo")
        model.load_state_dict(torch.load((test_config.model_dir+model_name), weights_only=True, map_location=test_config.device))
        #print("model")
        conversion = True
        load_time = time.time()-load_start

        with open(file_name, "wb") as file:
            for i in range (0, len(subset)-1, 2): #len(subset)-1
                if not conversion:
                    break

                sol_file = test_config.data_set+subset[i]
                ins_file = test_config.data_set+subset[i+1]
                time_readsol = time.time()
                solution = vrplib.read_solution(sol_file)
                instance = vrplib.read_instance(ins_file)
                print(time.time()-time_readsol)
                ## conversion
                conv_start = time.time()
                c = Conversor(instance_file=ins_file, solution_file=sol_file)
                data = c.convert_to_t_geometric()

                gnnData  = Data(x=torch.tensor(data.x, dtype=torch.float64),edge_index=data.edge_index,
                                    edge_attr=torch.tensor(data["edge_attr"], dtype=torch.float64),
                                    y=torch.tensor(data.y, dtype=torch.float64))

                output = model(gnnData)
                probMatrix = Test_operator.getProbMatrix(output, gnnData.edge_index)

                conv_time = time.time()-conv_start + load_time

                if (probMatrix.sum() == 0):
                    print("Model converged to zero")
                    conversion = False
                    break      

                for j in range(5):            
                    ## aco
                    aco_start = time.time()
                    aco = aco_cvrp_cpp.ACO_CVRP(10, instance['dimension'], instance['capacity'],
                                                test_config.alpha, test_config.beta, test_config.Q,
                                                test_config.decay, 0, test_config.probNew, test_config.seed[j])
                    aco.init(instance["edge_weight"], instance["demand"], None)
                    [path_aco, cost_aco, it_aco] = aco.optimize(100, 25)
                    aco_time = time.time()-aco_start

                    ## aco+gnn
                    model_start = time.time()
                    aco_gnn = aco_cvrp_cpp.ACO_CVRP(10, instance['dimension'], instance['capacity'],
                                                    test_config.alpha, test_config.beta, test_config.Q,
                                                    test_config.decay, 0, test_config.probNew, test_config.seed[j])

                    aco_gnn.init(instance["edge_weight"], instance["demand"], probMatrix)
                    [path_gnn, cost_gnn, it_gnn] = aco.optimize(100, 25)
                    model_time = time.time()-model_start

                    cost = solution["cost"]
                    result = f"{ins_file},{round(aco_time, 2)},{it_aco},{round(cost_aco,2)},{round(conv_time,2)},{round(model_time,2)},{it_gnn},{round(cost_gnn,2)},{cost}"
                    print()
                    pickle.dump(result, file)