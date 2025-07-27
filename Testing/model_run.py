from configurations import Configuration
from file_manager import File_manager
from test_operator import Test_operator
import concurrent.futures
import time

def model_run (model_dir, model_name, data_set):
    t1 = time.time()
    config = Configuration(model_dir=model_dir, data_set=data_set)
    fm = File_manager("test_set", config.cpu_num)
    # print(fm.subsets)
    tester = Test_operator()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(tester.test, fm.subsets[i], fm.files[i], config, model_name) for i in range(config.cpu_num)]
        
        num = 0
        for f in concurrent.futures.as_completed(results):
            num += 1
            print("done {}".format(num))

    fm.file_result(model_name=model_name)
    print("Time elapsed until {} completion: {} seconds".format(model_name, round(time.time()-t1, 2)))