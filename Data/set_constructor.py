import pickle
import conversor
import os
import time


data_set = []
data_set_name = "test_set"

directories = os.listdir(data_set_name)

start = time.time()
for i in range(len(directories)):
    set_start = time.time()
    print("{} set of {}:".format(i, len(directories)))
    sets = os.listdir(data_set_name+'/'+directories[i])
    sets.sort()
    #print(sets)
    if (directories[i] == "set_XXL" or directories[i] == "set_E"):
        print("Invalid Set")
    else:
        print(directories[i]+":")
        for j in range(0, len(sets)-1, 2):
            solution = sets[j]
            instance = sets[j+1]

            sol_path = data_set_name+'/'+directories[i]+'/'+solution
            ins_path = data_set_name+'/'+directories[i]+'/'+instance

            graph = conversor.Conversor(ins_path, sol_path)

            gnn = graph.convert_to_t_geometric()

            data_set.append(gnn)
            print("{}/{}".format(int(j/2 + 1), int(len(sets)/2)))

        set_end = time.time()
        print("Converted in {} seconds".format(round(set_end-set_start, 2)))

data_set.sort(key=lambda x: len(x["coordinates"]))
pickle.dump(data_set, open("data_set.bin", "wb"))
end = time.time()
print("Binary file generated successfully in {} seconds".format(round(end-start, 2)))