import analysis
import csv
import pandas as pd

graph_dir = "graphs/"
def main ():
    # results = analysis.take_each_model("results", "aco_results.csv")
    # #print(results)

    # with open("analysis_result.csv", 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',', quotechar='|')
    #     writer.writerow(["model_name"]+["time_mean"]+["time_std"]+["time_opt"]+
    #                                    ["cost_mean"]+["cost_std"]+["cost_opt"])

    #     #analysis.wilcoxon_test_all(results)
    #     for result in results:
    #         #print(result)
    #         analysis.plot_analysis(results[result], graph_dir, result, writer)
    
    df = pd.read_csv("analysis_result.csv")
    #print(df)

    print(df["time_mean"].mean())
    print(df["time_opt"].mean())
    print(df["cost_mean"].mean())
    print(df["cost_opt"].mean())

if __name__ == "__main__":
    main()