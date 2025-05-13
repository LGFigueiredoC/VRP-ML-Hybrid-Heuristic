from configurations import Configuration
from file_manager import File_manager
from test_operator import Test_operator
import concurrent.futures

def main ():
    config = Configuration()
    fm = File_manager("test_set", config.cpu_num)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        pass
    

if __name__ == "__main__":
    main()