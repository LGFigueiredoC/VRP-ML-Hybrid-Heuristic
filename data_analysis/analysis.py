import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import csv

def take_each_model (directory, aco_file_name):
    model_names = os.listdir(directory)
    model_names.sort()

    results = {}
    
    for index in enumerate(model_names[1::], start=1):
        name = index[1].split('_')
        if name[1] == "DeepGCNEncoder":
            new_name = name[2]
        else:
            new_name = name[1]

        new_name = new_name+'_'+name[3]+'_'+name[4]
        df = pd.read_csv(directory+'/'+model_names[index[0]])
        if not df.empty:
            result = df
            mean_results = result.groupby(["set_name"]).mean()
            aco = pd.read_csv(directory+'/'+aco_file_name)
            aco_results = aco.groupby(["set_name"]).mean()

            conv_aco = pd.Series(mean_results["GCN_AS_conv_time"]+mean_results["GCN_AS_exec_time"], name = "GNN+ACO_time")

            results[new_name] = pd.concat([aco_results["ACO_time"], aco_results["ACO_iterations"], aco_results["ACO_obj_func"],
                                conv_aco, mean_results["GCN_AS_iterations"], mean_results["GCN_AS_obj_func"]], axis=1)

    return results
            

def take_models_mean (directory, aco_file_name):
    model_names = os.listdir(directory)
    model_names.sort()
    #print(model_names)

    dataframe = pd.read_csv(directory+'/'+model_names[1])
    
    for model in model_names:
        if model == directory+"/"+aco_file_name:
            continue     
        df = pd.read_csv(directory+'/'+model)
        if not df.empty:
            #print(df)
            dataframe = pd.concat([dataframe, df], axis=0, ignore_index=True)

    mean_results = dataframe.groupby(["set_name"]).mean()
    aco = pd.read_csv(directory+'/'+aco_file_name)
    aco_results = aco.groupby(["set_name"]).mean()

    conv_aco = pd.Series(mean_results["GCN_AS_conv_time"]+mean_results["GCN_AS_exec_time"], name = "GNN+ACO_time")

    result = pd.concat([aco_results["ACO_time"], aco_results["ACO_iterations"], aco_results["ACO_obj_func"],
                           conv_aco, mean_results["GCN_AS_iterations"], mean_results["GCN_AS_obj_func"]], axis=1)

    return result



def take_models_per_arq (directory, aco_file_name):
    model_names = os.listdir(directory)
    model_names.sort()

    results = {}

    for index in enumerate(model_names[1::4], start=1):
        #print(index)
        name = index[1].split('_')[1]
        if name == "DeepGCNEncoder":
            name = index[1].split('_')[2]

        dataframes = []
        for j in range(index[0], index[0]+4):
            df = pd.read_csv(directory+'/'+model_names[j])

            if not df.empty:
                dataframes.append(df)
        
        result = pd.concat(dataframes, ignore_index=True)
        mean_results = result.groupby(["set_name"]).mean()
        aco = pd.read_csv(directory+'/'+aco_file_name)
        aco_results = aco.groupby(["set_name"]).mean()

        conv_aco = pd.Series(mean_results["GCN_AS_conv_time"]+mean_results["GCN_AS_exec_time"], name = "GNN+ACO_time")

        results[name] = pd.concat([aco_results["ACO_time"], aco_results["ACO_iterations"], aco_results["ACO_obj_func"],
                           conv_aco, mean_results["GCN_AS_iterations"], mean_results["GCN_AS_obj_func"]], axis=1)
        #print(results[name])
    
    return results



def take_models_per_nodes (directory, aco_file_name):
    model_names = os.listdir(directory)
    model_names.sort()

    results = {}

    for index in enumerate(model_names[3::2], start=3):
        print(index)
        name = index[1].split('_')[1]
        
        dataframes = []
        for j in range(index[0], index[0]+2):
            df = pd.read_csv(directory+'/'+model_names[j])

            if not df.empty:
                dataframes.append(df)
        
        result = pd.concat(dataframes, ignore_index=True)
        mean_results = result.groupby(["set_name"]).mean()
        aco = pd.read_csv(directory+'/'+aco_file_name)
        aco_results = aco.groupby(["set_name"]).mean()

        conv_aco = pd.Series(mean_results["GCN_AS_conv_time"]+mean_results["GCN_AS_exec_time"], name = "GNN+ACO_time")

        results[name] = pd.concat([aco_results["ACO_time"], aco_results["ACO_iterations"], aco_results["ACO_obj_func"],
                           conv_aco, mean_results["GCN_AS_iterations"], mean_results["GCN_AS_obj_func"]], axis=1)
        print(results[name])

    return results


def take_models_per_layers (directory, aco_file_name):
    model_names = os.listdir(directory)
    model_names.sort()

    results = {}

    for index in enumerate(model_names[1::4], start=1):
        print(index)
        name = index[1].split('_')[1]
        
        dataframes = []
        for j in range(index[0], index[0]+4):
            df = pd.read_csv(directory+'/'+model_names[j])

            if not df.empty:
                dataframes.append(df)
        
        result = pd.concat(dataframes, ignore_index=True)

        mean_results = result.groupby(["set_name"]).mean()
        aco = pd.read_csv(directory+'/'+aco_file_name)
        aco_results = aco.groupby(["set_name"]).mean()

        conv_aco = pd.Series(mean_results["GCN_AS_conv_time"]+mean_results["GCN_AS_exec_time"], name = "GNN+ACO_time")

        results[name] = pd.concat([aco_results["ACO_time"], aco_results["ACO_iterations"], aco_results["ACO_obj_func"],
                           conv_aco, mean_results["GCN_AS_iterations"], mean_results["GCN_AS_obj_func"]], axis=1)

    return results


def print_analysis (name, test, time_eff, time_perc, cost_eff, cost_perc, writer):
    time_eff_size = time_eff.loc[time_eff==True]
    cost_eff_size = cost_eff.loc[cost_eff==True]

    time_mean = float(round(time_perc.mean(), 2))
    time_std = float(round(time_perc.std(), 2))
    time_opt = float(round(((time_eff_size.size/time_eff.size)*100), 2))

    cost_mean = float(round(cost_perc.mean(), 2))
    cost_std = float(round(cost_perc.std(), 2))
    cost_opt = float(round(((cost_eff_size.size/cost_eff.size)*100), 2))

    writer.writerow([name]+[time_mean]+[time_std]+[time_opt]+
                           [cost_mean]+[cost_std]+[cost_opt])

    print(f"Analysis of a sample of {test.size} test instances:")
    print("- Mean relative percentage (nn > aco) {}%".format(time_mean))
    print("- STD of relative percentage (nn > aco) {}%".format(time_std))
    print("- {}% of cases have a decrease in time using AI".format(time_opt))
    print("- Mean relative cost efficiency in percentage (nn > aco) {}".format(cost_mean))
    print("- STD of relative cost efficiency in percentage (nn > aco) {}%".format(cost_std))
    print("- {}% of cases have an increase in cost performance using AI".format(cost_opt))


def plot_pearson (neural_time, neural_cost, pure_aco_time, pure_aco_cost, name):
    time_efficiency = neural_time < pure_aco_time
    time_efficiency_perc = pd.Series(round(((neural_time / pure_aco_time) * 100), 2), name = "NN/ACO time relation")
    cost_efficiency = neural_cost < pure_aco_cost
    cost_efficiency_perc = pd.Series(round(((neural_cost / pure_aco_cost) * 100), 2), name = "NN/ACO cost relation")

    time_mean = time_efficiency_perc.mean()
    cost_mean = cost_efficiency_perc.mean()

    a = b = c = 0

    for i in range (time_efficiency_perc.size):
        a += (time_efficiency_perc.iloc[i]-time_mean)*(cost_efficiency_perc.iloc[i]-cost_mean)
        b += (time_efficiency_perc.iloc[i]-time_mean)**2
        c += (cost_efficiency_perc.iloc[i]-cost_mean)**2

    r = a/(math.sqrt(b)*math.sqrt(c))

    plt.scatter(time_efficiency_perc, cost_efficiency_perc)
    plt.xlabel("Tempo Relativo (%)")
    plt.ylabel("Custo Relativo (%)")
    plt.title(f"{name} time/cost")

    print(r)
    print()


def plot_time_relation (neural_time, pure_aco_time, name):
    time_efficiency = neural_time < pure_aco_time
    time_efficiency_perc = pd.Series(round(((neural_time / pure_aco_time) * 100), 2), name = "NN/ACO time relation")

    time_efficiency_perc = pd.DataFrame(time_efficiency_perc)
    time_efficiency_perc.plot.box()
    plt.title(f"{name} time performance relative to ACO")
    plt.ylabel("Relative performance (%)")

    return [time_efficiency, time_efficiency_perc]


def plot_cost_relation (neural_cost, pure_aco_cost, name):
    cost_efficiency = neural_cost < pure_aco_cost
    cost_efficiency_perc = pd.Series(round(((neural_cost / pure_aco_cost) * 100), 2), name = "NN/ACO cost relation")

    cost_efficiency_perc = pd.DataFrame(cost_efficiency_perc)
    cost_efficiency_perc.plot.box()
    plt.title(f"{name} cost performance relative to ACO")
    plt.ylabel("Relative performance (%)")

    return [cost_efficiency, cost_efficiency_perc]
    


def plot_analysis (test_results, directory, name, writer):
    cost_results = plot_cost_relation(test_results["GCN_AS_obj_func"], test_results["ACO_obj_func"], name)
    plt.savefig(directory+name+"_cost.png")

    time_results = plot_time_relation(test_results["GNN+ACO_time"], test_results["ACO_time"], name)
    plt.savefig(directory+name+"_time.png")

    plt.clf()
    print(f"{name} pearson coefficient:")
    plot_pearson(test_results["GNN+ACO_time"], test_results["GCN_AS_obj_func"], test_results["ACO_time"], test_results["ACO_obj_func"], name)
    plt.savefig(directory+name+"_pearson.png")

    plt.close()

    print_analysis(name, test_results, time_results[0], time_results[1], cost_results[0], cost_results[1], writer)



def wilcoxon_test_all (test_results):
    wilcoxon_test(test_results["ACO_time"], test_results["GNN+ACO_time"])

    wilcoxon_test(test_results["ACO_iterations"], test_results["GCN_AS_iterations"])

    wilcoxon_test(test_results["ACO_obj_func"], test_results["GCN_AS_obj_func"])



def wilcoxon_test (group1, group2):
    diff = pd.Series(group1-group2)
    mark = diff > 0
    diff = diff.abs().sort_values()

    t_plus = 0
    t_minus = 0
    bigger = 0

    for index in enumerate(diff, start=1):
        if mark.iloc[index[0]-1] > 0:
            t_plus += index[0]
            bigger += 1

        else:
            t_minus += index[0]

        #print(f"{index[0]}, {diff.index[index[0]-1]}, {index[1]}, mark = {mark.iloc[index[0]-1]}")

    print(f"{t_plus}, {t_minus}")

    W = min(t_plus, t_minus)

    U_w = (diff.size*(diff.size+1.0)/ 4.0)
    var = (2.0*diff.size+1.0)/6.0
    
    Z = (W - U_w)/(var*U_w)

    print(Z*2)
    print(bigger)