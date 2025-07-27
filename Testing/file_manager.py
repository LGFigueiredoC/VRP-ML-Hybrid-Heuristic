import os
import csv

class File_manager:
    def __init__(self, data_set_name: str, cpu_num: int):
        self.set = os.listdir(data_set_name)
        self.set.sort()

        self.files = []

        for i in range(cpu_num):
            self.files.append("{}{}.csv".format("result_", i))
        
        step = len(self.set)//cpu_num
        self.subsets = []

        for i in range(cpu_num):
            self.subsets.append(self.set[i*step:(i+1)*step])

        rest = self.set[len(self.set)-len(self.set)%cpu_num::]
        self.subsets[cpu_num-1] = self.subsets[cpu_num-1] + rest
        print(self.subsets)

    

    def file_result (self, model_name):
        with open("results/"+model_name+"_results.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(["set_name"] +["ACO_time"] + ["ACO_iterations"] + ["ACO_obj_func"] +
                    ["GCN_AS_conv_time"] + ["GCN_AS_exec_time"] + ["GCN_AS_iterations"] + 
                    ["GCN_AS_obj_func"] + ["Solution"])
            
            for i in range (len(self.files)):
                with open (self.files[i], newline='') as f:
                    reader = csv.reader(f)
                    
                    for row in reader:
                        writer.writerow(row)
        
        for i in range (len(self.files)):
            os.remove(self.files[i])