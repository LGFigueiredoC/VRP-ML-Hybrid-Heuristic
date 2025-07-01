import pickle
import conversor_novo
import vrplib
import os
import time

def print_dimension(data):
    print(len(data["coordinates"]))


def set_create(data_set_name, data_type, size=None):
    directories = os.listdir(data_set_name)
    for i in range(len(directories)):
        set_start = time.time()
        print("{} set of {}:".format(i, len(directories)))
        sets = os.listdir(data_set_name+'/'+directories[i])
        sets.sort()
        if (directories[i] == "set_XXL"):
            print("Invalid Set")
        else:
            print(directories[i]+":")
            for j in range(0, len(sets)-1, 2):
                data_set = []
                solution = sets[j]
                instance = sets[j+1]

                sol_path = data_set_name+'/'+directories[i]+'/'+solution
                ins_path = data_set_name+'/'+directories[i]+'/'+instance

                if data_type == "resized":
                    graph = conversor_novo.Conversor(ins_path, sol_path, size)

                elif data_type == "layers":
                    inst = vrplib.read_instance(ins_path)
                    size = (inst["dimension"]//100 + 1) * 100
                    graph = conversor_novo.Conversor(ins_path, sol_path, size)

                elif data_type == "batches":
                    graph = conversor_novo.Conversor(ins_path, sol_path)
                    
                else:
                    print("No data type chosen.")
                    exit(1)

                gnn = graph.convert_to_t_geometric()
                data_set.append(gnn)
                print("{}/{}".format(int(j/2 + 1), int(len(sets)/2)))
                print("Dimension: {}".format(len(gnn.x)))
                pickle.dump(data_set, open("layered_data_set.bin", "ab"))
                del data_set

            set_end = time.time()
            print("Converted in {} seconds".format(round(set_end-set_start, 2)))
    

start = time.time()
size = 0
with open("data_set_batches.bin", "rb") as data_set:
    set = pickle.load(data_set)
    size = len(set[len(set)-1].coordinates)
set_create("data_set", "layers", size)

#data_set.sort(key=lambda x: len(x["coordinates"]))
#for instance in data_set:
    #print_dimension(instance)

end = time.time()
print("Binary file generated successfully in {} seconds".format(round(end-start, 2)))