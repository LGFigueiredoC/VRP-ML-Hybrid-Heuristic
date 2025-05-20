from configurations import Configuration
from file_manager import File_manager
from test_operator import Test_operator
import concurrent.futures

def main ():
    config = Configuration()
    fm = File_manager("test_set", config.cpu_num)
    model_name = "2540_epocas"
    print(fm.subsets)
    tester = Test_operator()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(tester.test, fm.subsets[i], fm.files[i], config, model_name) for i in range(config.cpu_num)]
        
        num = 0
        for f in concurrent.futures.as_completed(results):
            num += 1
            print("done {}".format(num))

    fm.file_result()
    

if __name__ == "__main__":
    main()