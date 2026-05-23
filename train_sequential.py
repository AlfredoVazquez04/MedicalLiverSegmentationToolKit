import subprocess
import time
import os
import json
from datetime import datetime
from multiprocessing import Process

from utils import check_gpu_memory



class SequentialTrain:
    '''
    models_2d = {} # TODO
    models_3d = {
        'segformer': 0,
        'swin_unetr': 0,
        'unet': 0,
        'unetr': 0,
        'unetpp': 0,
        'unetrpp': 0,
        'uxlstm': 0,
        'attention_unet': 0
    }
    '''
    config_modelos_3d = {
        'swin_unetr':     {'batch_size': 4, 'accumulate': 6},
        'attention_unet': {'batch_size': 4, 'accumulate': 6},
        'unet':           {'batch_size': 8, 'accumulate': 3},
        'unetpp':         {'batch_size': 8, 'accumulate': 3},
        'uxlstm':         {'batch_size': 8, 'accumulate': 3},
        'segformer':      {'batch_size': 12, 'accumulate': 2},
        'unetr':          {'batch_size': 12, 'accumulate': 2},
        'unetrpp':        {'batch_size': 12, 'accumulate': 2}
    }
    
    dimensions = ['3d'] # '2d',

    dataset = 'amos'
    

    def __call__(self):
        """Method to run the models sequentially."""        
        for dimension in self.dimensions:
            if dimension == '2d':
                models = self.models_2d
            else:
                models = self.config_modelos_3d

            for model, config in models.items():
                
                # Check GPU memory (Optional now, since running sequentially guarantees 
                # the previous model released its memory, but kept for safety)
                gpu_memory = check_gpu_memory() 
                while gpu_memory < 14000:  
                    print("Need more GPU memory. Waiting...")
                    time.sleep(720)  
                    gpu_memory = check_gpu_memory()
               
                print(f"\nTrain {model} ({dimension})...")  
                
                try:
                    # Run the model sequentially. 
                    # This will block and wait for the training to finish completely.
                    self.run_model(
                        model=model, 
                        dimension=dimension, 
                        run_version="0", 
                        dataset=str(self.dataset),
                        batch_size=str(config['batch_size']),
                        accumulate=str(config['accumulate'])
                    )
                except Exception as e:
                    print(f"Error training {model}: {e}")

                # Optional: brief pause between models to ensure memory clears
                time.sleep(10)  

        print("All models are tested")

    @staticmethod
    def run_model(model, dimension, run_version, dataset, batch_size, accumulate):
        """Function to run the model to predict the battery test.
        When the prediction is finished, the time with more parameters are saved in a json file.
        The place for the json file is in the respective folder results.

        Args:
            model (str): Name of the model
            dimension (str): Number of dimensions (2d or 3d)
            run_version (str): Version of the model
        """
        data_path = os.path.join(os.environ['TMPDIR'], 'amos')

        args = [
            '--mode', 'Train',
            '--trainmode', 'init',
            '--model', model, 
            '--dimension', dimension,
            '--dataset', dataset,
            '--data_dir', data_path,
            '--cache_rate', '0.',
            '--max_epochs', '100',
            '--out_channels', '16',
            '--batch_size', batch_size,                  
            '--accumulate_grad_batches', accumulate      
        ]

        cmd = ['python3', 'train.py'] + args

        start_time = datetime.now()
        subprocess.run(cmd)


        # result = {
        #     'model': model,
        #     'dimension': dimension,
        #     'run_version': run_version,
        #     'dataset': SequentialTrain.dataset, 
        #     'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
        # }

        # base_path = f'./logs/{SequentialTrain.dataset}/{model}_{dimension}/lightning_logs/version_{run_version}'
        # file_path = f'{base_path}/time_train_version_{run_version}.json'

        # os.makedirs(base_path, exist_ok=True)
        # with open(file_path, 'w') as json_file:
        #     json.dump(result, json_file, indent=4)
        #     json_file.write(',\n')

        
        # base_path = f'./logs/{SequentialTrain.dataset}/{model}_{dimension}/lightning_logs/version_{run_version}'
        # file_path = f'{base_path}/time_train_version_{run_version}.json'

        # with open(file_path, 'r') as f:
        #     result = json.load(f)

        # end_time = datetime.now()  
        # elapsed_time = end_time - datetime.strptime(result['start_time'], "%Y-%m-%d %H:%M:%S")  

        # result['end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
        # result['elapsed_time'] = str(elapsed_time)

        # with open(file_path, 'w') as f:
        #     json.dump(result, f, indent=4)

        
        

    

if __name__ == "__main__":
    SequentialTrain()()
