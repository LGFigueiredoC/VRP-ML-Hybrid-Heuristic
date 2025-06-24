import vrplib
import networkx as nx
from torch_geometric.utils.convert import from_networkx

class Conversor:

    # A classe conversor é feita para receber uma instância de um cvrp e retornando um
    # grafo do tipo Data do torch_geometric, colocando os atributos Demanda e Coordenada
    # nos nodos e o atributo Distancia nas arestas

    # Para usar basta criar o objeto com o nome do arquivo como parâmetro e utilizar a
    # função convert_to_t_geometric
    def __init__(self, instance_file, solution_file) -> None:
        self.current_instance = vrplib.read_instance(instance_file)
        self.solution = vrplib.read_solution(solution_file)
        self.optimal_routes = []
        self.nx_graph = nx.Graph()
        self.n_nodes = self.current_instance['dimension']
        self.n_edges = 0


    def add_instance_nodes(self):
        i=0
        for i in range(self.n_nodes):
            demand = self.current_instance['demand'][i]
            self.nx_graph.add_node(i, demand=demand)
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
        for i in range (self.n_nodes):
            for j in range(i+1, self.n_nodes):
                self.nx_graph.add_edge(i, j, distance=self.current_instance['edge_weight'][i,j])
                self.n_edges += 1


    def resize_graph (self, size):
        quantity = size-self.n_nodes
        for i in range(quantity):
            demand = 0
            distance = 4294967296

            self.nx_graph.add_node(self.n_nodes, demand=demand)
            self.n_nodes += 1

            for j in range(self.n_nodes):
                for k in range (j+1, self.n_nodes):
                    self.nx_graph.add_edge(j, k, distance=distance)
                    self.n_edges += 1


    def add_solution(self):
        size = self.n_nodes*self.n_nodes - self.n_nodes
        for i in range(size):
            self.optimal_routes.append(0)
        
        for route in self.solution['routes']:
            path = [0]
            for i in range(len(route)):
                if (i == len(route)-1):
                    pass
                path.append(route[i])
            path.append(0)


            #print(path)
            for i in range(len(path)-1):
                #print(path[i])
                #print(path[i+1])
                #print()
                
                if (path[i] < path[i+1]):
                    #print(((self.n_nodes-path[i])*path[i]+path[i+1]) -1)
                    #print(((self.n_nodes-path[i])*path[i+1]+path[i]) -1)
                    self.optimal_routes[(path[i]*(self.n_nodes-1))+path[i+1] -1] = 1
                    self.optimal_routes[(path[i+1]*(self.n_nodes-1))+path[i]] = 1

                else:
                    self.optimal_routes[(path[i+1]*(self.n_nodes-1))+path[i] -1] = 1
                    self.optimal_routes[(path[i]*(self.n_nodes-1))+path[i+1]] = 1


    # Função feita apenas para verificar o resultado da conversão
    def print_graph (self):
        print("Graph nodes and edges in networkx format:")
        print()

        for i in range(self.n_nodes):
            print("{}: {}".format(i, self.nx_graph.nodes[i]))

        print()
        print(self.nx_graph.edges)
        print()
        print(self.optimal_routes)


    def convert_to_t_geometric (self, size):
        self.add_instance_nodes()
        self.add_instance_edges()
        self.resize_graph(size)
        self.add_solution()

        gnn_graph = from_networkx(self.nx_graph, group_node_attrs=['demand'], group_edge_attrs=['distance'])
        gnn_graph.y = self.optimal_routes

        #self.print_graph() # visualização do resultado
        #print("Data in torch_geometric format:")
        #print()
        #print(gnn_graph['edge_index'][1])

        return gnn_graph