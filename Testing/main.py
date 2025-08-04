from configurations import Configuration
from file_manager import File_manager
from test_operator import Test_operator
from model_run import model_run
import modeler
import concurrent.futures
import time
import os

model_dir = "trained_models/"
data_set = "test_set/"
model_name = "modelo_DeepGCNEncoder_gen_conv_1_64"
def main ():
    models = os.listdir(model_dir)
    models.sort()
    #print(models)

    i = 1
    for model in models:
        print(model_dir+model)
        model_run(model_dir, model, data_set)
        print(f"Model {i}/{len(models)}")
        i += 1
        break

if __name__ == "__main__":
    main()