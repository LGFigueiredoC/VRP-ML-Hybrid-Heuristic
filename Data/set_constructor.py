import pickle
import conversor
import os
import time

data_set = []

#gnn_graph = conv.convert_to_t_geometric()
directories = os.listdir('data_set')
#print(directories)
start = time.time()
for i in range(len(directories)):
    set_start = time.time()
    print("{} set of {}:".format(i, len(directories)))
    sets = os.listdir('data_set/'+directories[i])
    sets.sort()
    #print(sets)
    if (directories[i] == "set_XXL" or directories[i] == "set_E"):
        print("Invalid Set")
    else:
        print(directories[i]+":")
        for j in range(0, len(sets)-1, 2):
            solution = sets[j]
            instance = sets[j+1]

            sol_path = 'data_set/'+directories[i]+'/'+solution
            ins_path = 'data_set/'+directories[i]+'/'+instance

            graph = conversor.Conversor(ins_path, sol_path)

            gnn = graph.convert_to_t_geometric()

            data_set.append(gnn)
            print("{}/{}".format(int(j/2 + 1), int(len(sets)/2)))

        set_end = time.time()
        print("Converted in {} seconds".format(round(set_end-set_start, 2)))

pickle.dump(data_set, open("data_set.bin", "wb"))
end = time.time()
print("Binary file generated successfully in {} seconds".format(round(end-start, 2)))