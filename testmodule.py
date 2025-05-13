import aco_cvrp_cpp
import numpy as np

# help(aco_cvrp_cpp.aco_cvrp_optimize)
# help(aco_cvrp_cpp.ACO_CVRP)

d0 = [0,14,21,33,22,23,12,22,32,32,21,28,30,29,31,30]
d1 = [14,0,12,19,12,24,12,19,21,27,7,19,16,21,33,17]
d2 = [21,12,0,15,22,16,11,9,12,15,11,29,19,9,24,23]
d3 = [33,19,15,0,21,31,25,23,8,24,12,25,9,17,37,16]
d4 = [22,12,22,21,0,36,24,30,26,37,12,7,13,30,44,9]
d5 = [23,24,16,31,36,0,13,8,25,13,26,43,35,16,8,39]
d6 = [12,12,11,25,24,13,0,10,23,20,16,31,26,17,21,28]
d7 = [22,19,9,23,30,8,10,0,18,10,19,37,28,9,15,32]
d8 = [32,21,12,8,26,25,23,18,0,17,15,32,17,10,31,23]
d9 = [32,27,15,24,37,13,20,10,17,0,25,44,31,7,16,37]
d10 = [21,7,11,12,12,26,16,19,15,25,0,19,10,18,34,13]
d11 = [28,19,29,25,7,43,31,37,32,44,19,0,16,37,51,10]
d12 = [30,16,19,9,13,35,26,28,17,31,10,16,0,24,43,6]
d13 = [29,21,9,17,30,16,17,9,10,7,18,37,24,0,21,30]
d14 = [31,33,24,37,44,8,21,15,31,16,34,51,43,21,0,47]
d15 = [30,17,23,16,9,39,28,32,23,37,13,10,6,30,47,0]

distanceMatrix = np.asarray([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15]).astype(np.float64)
	
demand = np.asarray([0,19,30,16,23,11,31,15,28,8,8,7,14,6,19,11]).astype(np.float64)

nAnts = 4  # Nùmero formigas
nCities = 16 #Número Cidades
capac = 35 # Capacidade de cada veículo
alpha = 0.5 # Parâmetro do ACO
beta = 0.8 # Parâmetro do ACO
q = 2 # Parâmetro do ACO (esse é o único que não tinha na outra implementação)
rho = 0.2 # Parâmetro do ACO - (1 - taxa de decaimento)
initCity = 0 # Cidade inicial
probNew = 0.1 # Probabilidade dele buscar uma rota nova
seed = 0 # Semente aleatóri

# aco_cvrp = aco_cvrp_cpp.ACO_VRP(nAnts,nCities,alpha,beta,q,rho,initCity,seed)
aco_cvrp = aco_cvrp_cpp.ACO_CVRP(nAnts,nCities,capac,alpha,beta,q,rho,initCity,probNew,seed)

feromonioInicial = np.ones((nCities,nCities))/(nCities*(nCities-1))
for i in range(nCities):
    feromonioInicial[i,i] = 0

feromonioInicial = feromonioInicial.astype(np.float64)

aco_cvrp.init(distanceMatrix,demand,None)
aco_cvrp.init(distanceMatrix,demand,feromonioInicial)

# print("oi")



# print(feromonioInicial)


[a,b] = aco_cvrp.optimize(100,25)
# [a2,b2] = aco_cvrp.optimize(100,25)



print(a)
print(b)
# print(a2)
# print(b2)
