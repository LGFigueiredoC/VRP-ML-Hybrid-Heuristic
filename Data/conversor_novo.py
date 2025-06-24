import vrplib
# from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
import torch
import numpy as np

class Conversor:

    # A classe conversor é feita para receber uma instância de um cvrp e retornando um
    # grafo do tipo Data do torch_geometric, colocando os atributos Demanda e Coordenada
    # nos nodos e o atributo Distancia nas arestas

    # Para usar basta criar o objeto com o nome do arquivo como parâmetro e utilizar a
    # função convert_to_t_geometric
    def __init__(self, instance_file, solution_file, dst_size=None) -> None:
        self.current_instance = vrplib.read_instance(instance_file)
        self.solution = vrplib.read_solution(solution_file)
        self.optimal_routes = []
        # self.nx_graph = nx.Graph()
        self.n_nodes = self.current_instance['dimension']
        self.n_edges = 0
        self.demand = []
        self.edges = [[],[]]
        self.edge_weight = []
        self.y = []
        self.dst_size = dst_size
        self.far_distance = 10*np.sum(self.demand)


    def add_instance_nodes(self):
        self.demand = self.current_instance['demand'].reshape((-1,1))
        if (self.dst_size != None):
            aux = np.zeros((self.dst_size,1))
            aux[:self.demand.shape[0]] = self.demand
            self.demand = aux
        # i=0
        # for i in range(self.n_nodes):
        #     demand = self.current_instance['demand'][i]
        #     self.nx_graph.add_node(i, demand=demand)
        # if 'node_coord' in self.current_instance.keys():
        #     for node in self.current_instance['node_coord']:
        #         demand = self.current_instance['demand'][i]
        #         new_node_coord = ({'y' : node[1], 'x' : node[0]})

        #         self.nx_graph.add_node(i, coordinates=new_node_coord, demand=demand)
        #         i += 1
        # else:
        #     num_nodes = self.current_instance['demand'].shape[0]
        #     fro


    def add_instance_edges(self):
        if self.dst_size == None:
            for i in range (self.n_nodes):
                for j in range(self.n_nodes):
                    if (i == j):
                        continue
                    self.edges[0].append(i)
                    self.edges[1].append(j)
                    self.n_edges += 1
                    self.edge_weight.append(self.current_instance['edge_weight'][i,j])
        else:
            # self.n_nodes = self.dst_size
            for i in range (self.dst_size):
                for j in range(self.dst_size):
                    if (i == j):
                        continue
                    d = self.far_distance
                    if (i < self.n_nodes) and (j < self.n_nodes):
                        d = self.current_instance['edge_weight'][i,j]
                    self.edges[0].append(i)
                    self.edges[1].append(j)
                    self.n_edges += 1
                    self.edge_weight.append(d)
            # self.n_nodes = self.dst_size
                    

    def add_solution(self):
        self.optimal_routes = np.zeros((self.edges.shape[1],))
        
        for route in self.solution['routes']:
            path = [0]
            for i in range(len(route)):
                if (i == len(route)-1):
                    pass
                path.append(route[i])
            path.append(0)
            for i in range(len(path)-1):
                c0 = self.edges[0,:] == path[i]
                c1 = self.edges[1,:] == path[i+1]
                idx = np.where(c0 & c1)[0]
                c2 = self.edges[1,:] == path[i]
                c3 = self.edges[0,:] == path[i+1]
                idx2 = np.where(c2 & c3)[0]
                self.optimal_routes[idx] = 1
                self.optimal_routes[idx2] = 1
                # compare = [[path[i]],[path[i+1]]]
                # compare = np.asarray(compare)
                # idx = np.where( self.edges == compare)
                # print(idx)

        # size = self.n_nodes*self.n_nodes - self.n_nodes
        # for i in range(size):
        #     self.optimal_routes.append(0)
        
        # for route in self.solution['routes']:
        #     path = [0]
        #     for i in range(len(route)):
        #         if (i == len(route)-1):
        #             pass
        #         path.append(route[i])
        #     path.append(0)


        #     #print(path)
        #     for i in range(len(path)-1):
        #         #print(path[i])
        #         #print(path[i+1])
        #         #print()
                
        #         if (path[i] < path[i+1]):
        #             #print(((self.n_nodes-path[i])*path[i]+path[i+1]) -1)
        #             #print(((self.n_nodes-path[i])*path[i+1]+path[i]) -1)
        #             self.optimal_routes[(path[i]*(self.n_nodes-1))+path[i+1] -1] = 1
        #             self.optimal_routes[(path[i+1]*(self.n_nodes-1))+path[i]] = 1

        #         else:
        #             self.optimal_routes[(path[i+1]*(self.n_nodes-1))+path[i] -1] = 1
        #             self.optimal_routes[(path[i]*(self.n_nodes-1))+path[i+1]] = 1

    # Função feita apenas para verificar o resultado da conversão
    # def print_graph (self):
    #     print("Graph nodes and edges in networkx format:")
    #     print()

    #     for i in range(self.n_nodes):
    #         print("{}: {}".format(i, self.nx_graph.nodes[i]))

    #     print()
    #     print(self.nx_graph.edges)
    #     print()
    #     print(self.optimal_routes)


    def convert_to_t_geometric (self):
        self.add_instance_nodes()
        self.add_instance_edges()
        self.edges = np.asarray(self.edges)
        self.edge_weight = np.asarray(self.edge_weight).reshape((-1,1))
        self.add_solution()
        
        
        # if (self.dst_size != None):
        #     self.n_nodes = self.dst_size
        #     self.optimal_routes = np.asarray(self.optimal_routes)
        #     print(self.optimal_routes.shape)
        #     optim = np.zeros((self.n_nodes,self.n_nodes),dtype=self.optimal_routes.dtype)
        #     optim[:self.optimal_routes.shape[0],:self.optimal_routes.shape[1]] = self.optimal_routes
            

        gnn_graph = Data(x=torch.tensor(self.demand, dtype=torch.float64),edge_index=self.edges,
                       edge_attr=torch.tensor(self.edge_weight, dtype=torch.float64),
                       y=torch.tensor(self.optimal_routes, dtype=torch.float64) + torch.finfo(torch.float64).eps)
        # gnn_graph = from_networkx(self.nx_graph, group_node_attrs=['demand'], group_edge_attrs=['distance'])
        # gnn_graph.y = self.optimal_routes
# 
        #self.print_graph() # visualização do resultado
        #print("Data in torch_geometric format:")
        #print()
        #print(gnn_graph['edge_index'][1])

        return gnn_graph