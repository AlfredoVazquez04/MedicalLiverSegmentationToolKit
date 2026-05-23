import os
import logging

import scipy.io
import glob
import argparse
from datetime import datetime
from xmlrpc.client import boolean

import gc
import numpy as np
import yaml

import nibabel as nib


import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete


from monai.config import print_config
from monai.metrics import DiceMetric

from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
    list_data_collate,
)

from model.utils import get_model

class Net(pl.LightningModule):
    """Class that defines the Lightning Module that will be used for training, validation and testing.
    """    

    def __init__(self, args):
        """Constructor of the class. Initialize the Lightning Module that will be used for training, validation and testing.

        Args:
            args (argparse.Namespace): Arguments from the command line.
        """        
        super().__init__()
        self.save_hyperparameters()
        
        self.keys = ["image", "label"]

        try:
            self.args = args
            self._model = get_model(args)

            self.post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
            self.post_label = AsDiscrete(to_onehot=args.out_channels)

            # Loss metrics
            self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
            self.dice_metric = DiceMetric(
                include_background=False, reduction="mean", get_not_nans=False)
            self.dice_metric_test = DiceMetric(
                include_background=False, reduction="mean", get_not_nans=False)
            
            self.roi_size = args.roi_size
            self.inference_batch_size = args.inference_batch_size
            self.batch_size = args.batch_size

            self.best_val_dice = 0
            self.best_val_epoch = 0
            self.best_test_dice = 0
            self.best_test_epoch = 0
            self.max_epochs = args.max_epochs 

            self.metric_test_values = []
            self.metric_values = []
            self.epoch_loss_values = []
            self.validation_step_outputs = []
            self.test_step_outputs = []
            self.training_step_outputs = []

        except Exception as e: 
            print(e)
            raise e
            


    def forward(self, x):
        """Function that performs a forward pass on the network.

        Args:
            x (torch.Tensor | monai.data.meta_tensor.MetaTensor): Input data to the network 

        Returns:
            (torch.Tensor | monai.data.meta_tensor.MetaTensor): Output data from the network
        """    
        out = self._model(x)
        
        if isinstance(out, list) or isinstance(out, tuple):
            return out[-1]
            
        return out

    def configure_optimizers(self):
        """Function that configures the optimizer to be used during training.

        Returns:
           torch.optim.adamw.AdamW : Optimizer to be used during training
        """        
        optimizer = torch.optim.AdamW(
            self._model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-5
        )
        return optimizer
    

    def training_step(self, batch, batch_idx):
        """Function that performs a training step on the network. 

        Args:
            batch (dict): The batch of data to be used for training
            batch_idx (int): The index of the batch

        Returns:
            dict: Dictionary containing the loss and the tensorboard logs
        """        
        images, labels = batch[self.keys[0]], batch[self.keys[1]]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        
        self.log(
            'train_loss', 
            loss.item(), 
            prog_bar=True, 
            batch_size=self.batch_size
        )
        
        d_detached = {"loss": loss.detach(), "log": tensorboard_logs}
        self.training_step_outputs.append(d_detached)

        return {"loss": loss, "log": tensorboard_logs}

    def on_training_epoch_end(self):
        """Function that performs an action at the end of the training epoch.
        """        
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())
        self.training_step_outputs.clear()  # free memory


    def validation_step(self, batch, batch_idx):
        """Function that performs a validation step on the network. 

        Args:
            batch (dict): The batch of data to be used for training
            batch_idx (int): The index of the batch

        Returns:
            dict: Dictionary containing the loss and the tensorboard logs
        """  
        images, labels = batch[self.keys[0]], batch[self.keys[1]]
        
        outputs = sliding_window_inference(
            images, 
            self.roi_size, 
            self.inference_batch_size, 
            self.forward
        )
        
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        self.dice_metric(y_pred=outputs, y=labels)
        self.log(
            "val_loss", 
            loss, 
            batch_size=1
            ) 
        d = {"val_loss": loss.detach(), "val_number": len(outputs)}
        self.validation_step_outputs.append(d)

        return d


    def on_validation_epoch_end(self):
        """Function that performs an action at the end of the validation epoch.

        Returns:
            dict: Dictionary containing the tensorboard logs
        """        
        val_loss, num_items = 0, 0

        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        print(
            f"\nCurrent epoch: {self.current_epoch} "
            f"Current mean dice: {mean_val_dice:.4f}"
            f"\nBest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        
        self.metric_values.append(mean_val_dice)
        self.validation_step_outputs.clear()  # free memory

        return {"log": tensorboard_logs}
    

    def test_step(self, batch, batch_idx):
        """Function that performs the test step on the network. 

        Args:
            batch (dict): The batch of data to be used for training
            batch_idx (int): The index of the batch

        Returns:
            dict: Dictionary containing the loss and the tensorboard logs
        """  
        images, labels = batch["image"], batch["label"]
        outputs = sliding_window_inference(
            images, 
            self.roi_size, 
            self.inference_batch_size, 
            self.forward
        )

        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric_test(y_pred=outputs, y=labels)

        self.log(
            "test_loss", 
            loss, 
            batch_size=1
            ) 
        d = {"test_loss": loss.detach(), "test_number": len(outputs)}
        self.test_step_outputs.append(d)

        return d


    def on_test_epoch_end(self):
        """Function that performs an action at the end of the test epoch.

        Returns:
            dict: Dictionary containing the tensorboard logs
        """        
        test_loss, num_items = 0, 0

        for output in self.test_step_outputs:
            test_loss += output["test_loss"].sum().item()
            num_items += output["test_number"]

        mean_test_dice = self.dice_metric_test.aggregate().item()
        self.dice_metric_test.reset()
        mean_test_loss = torch.tensor(test_loss / num_items)
        tensorboard_logs = {
            "test_dice": mean_test_dice,
            "test_loss": mean_test_loss,
        }

        if mean_test_dice > self.best_test_dice:
            self.best_test_dice = mean_test_dice
            self.best_test_epoch = self.current_epoch

        print(
            f"\nCurrent epoch: {self.current_epoch} "
            f"Current mean dice test: {mean_test_dice:.4f}"
            f"\nbest mean dice test: {self.best_test_dice:.4f} "
            f"at epoch: {self.best_test_epoch}"
        )

        self.metric_test_values.append(mean_test_dice)
        self.test_step_outputs.clear()

        return {"log": tensorboard_logs}

def simular_consumo_real_gb(net, device, roi_size=(96, 96, 96), in_channels=1, batch_size=1):
    """
    Simula un step de entrenamiento para medir el pico real de memoria VRAM.
    """
    # 1. Asegurar que la GPU está limpia y resetear el contador de picos
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    
    # 2. Crear un tensor "falso" simulando un parche de tu imagen 3D
    # Formato: (Batch, Channels, Depth, Height, Width)
    dummy_input = torch.randn((batch_size, in_channels, *roi_size), device=device)
    
    # Pasar el modelo a GPU y ponerlo en modo entrenamiento
    net.to(device)
    net.train()
    
    peak_memory_gb = 0.0
    
    try:
        # --- 3. FORWARD PASS (Genera los mapas de activaciones) ---
        output = net.forward(dummy_input)
        
        # Creamos una "falsa pérdida" solo para poder calcular gradientes
        # Usamos .sum() para que genere un escalar
        dummy_loss = output.sum()
        
        # --- 4. BACKWARD PASS (Calcula gradientes - ¡El pico de memoria ocurre aquí!) ---
        dummy_loss.backward()
        
        # 5. Obtener el pico máximo de memoria que se alcanzó en la GPU
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_gb = peak_memory_bytes / (1024 ** 3)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n\t[!] OOM: ¡El modelo requiere más VRAM de la disponible para entrenar!")
            peak_memory_gb = float('inf')
        else:
            raise e
            
    finally:
        # 6. Limpieza extrema: Borrar tensores, sacar el modelo de GPU y vaciar caché
        if 'dummy_input' in locals(): del dummy_input
        if 'output' in locals(): del output
        if 'dummy_loss' in locals(): del dummy_loss
        
        net.cpu() # Devolvemos el modelo a la RAM normal
        torch.cuda.empty_cache()
        gc.collect()
        
    return peak_memory_gb

def encontrar_batch_size_maximo(args, device, max_test_batch=16):
    """
    Sube el batch size de 1 en 1 hasta que la GPU se queda sin memoria.
    Devuelve el último batch size que funcionó correctamente.
    """
    print(f"\n--- Iniciando búsqueda de Batch Size para {args.model.upper()} ---")
    batch_size = 1
    mejor_batch_size = 1
    vram_segura = 0.0
    
    while batch_size <= max_test_batch:
        print(f"\tProbando batch_size = {batch_size}...")
        
        # Instanciamos el modelo en CPU para esta prueba
        net = Net(args)
        
        # Simulamos el consumo
        vram_gb = simular_consumo_real_gb(
            net=net, 
            device=device, 
            roi_size=args.roi_size,
            in_channels=args.in_channels,
            batch_size=batch_size
        )
        
        # Limpiamos
        del net
        torch.cuda.empty_cache()
        
        # Comprobamos si petó la memoria
        if vram_gb == float('inf'):
            print(f"\t[!] Límite alcanzado. El modelo no soporta batch_size = {batch_size}")
            break
        else:
            print(f"\t[OK] Soporta batch_size = {batch_size} (VRAM Peak: {vram_gb:.2f} GB)")
            mejor_batch_size = batch_size
            vram_segura = vram_gb
            batch_size += 1  # Incrementamos para la siguiente prueba
            
    print(f"-> BATCH SIZE MÁXIMO SEGURO: {mejor_batch_size} (Consumiendo {vram_segura:.2f} GB)")
    return mejor_batch_size

class MainModule:
    """Class that defines the main module that will be used to train, test and predict with different medical models.
    """    
    def __init__(self, args):
        self.models = ['segformer',
                       'swin_unetr',
                       'unet',
                       'unetr',
                       'unetrpp',
                       'unetpp',
                       'uxlstm_bot',
                       'attention_unet'
                    ]
        self.args = args

    def __call__(self):
        """Call method that will be used to train, test and predict with different medical models.

        Args:
            args (argparse.Namespace): Arguments from the command line.

        Raises:
            ValueError: Invalid mode. Choose between Train, Test or Predict
        """     

        num_of_gpus = torch.cuda.device_count()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        #print_config()
        print("Number of GPUs available: {}. Device used: {}".format(num_of_gpus, device))
        logging.basicConfig(level=logging.INFO)

        for model in self.models:
            self.args.model = model
            config_path = 'config/%s/%s_%s.yaml'%(self.args.dataset, self.args.model, self.args.dimension)

            if not os.path.exists(config_path):
                raise ValueError("The specified configuration doesn't exist: %s"%config_path)

            print('Loading configurations from %s'%config_path)

            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)

            for key, value in config.items():
                setattr(args, key, value)

            torch.cuda.empty_cache()
            net = Net(self.args)
            batch_maximo = encontrar_batch_size_maximo(
                args=self.args, 
                device=device,
                max_test_batch=10e6  # Ponemos un límite de 8 para no eternizar la prueba
            )
            
            # Borramos el modelo de la RAM
            del net


def get_parser():
    """Function to get the parser with the arguments.

    Raises:
        ValueError: The specified configuration doesn't exist

    Returns:
        argparse.Namespace: Arguments from the command line.
    """    
    parser = argparse.ArgumentParser(description="Framework to train, test and predict with different medical models")

    parser.add_argument("--max_epochs", default=800, type=int, help="Max number of epochs for training")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for training")
    parser.add_argument("--cache_rate", default=1.0, type=float, help="Cache rate for training")
    parser.add_argument("--pin_memory", default=False, type=bool, help="Pin memory for training")

    parser.add_argument("--percentage_train", default=0.8, type=float, help="Percentage of training data")
    
    parser.add_argument("--spatial_dims", default=3, type=int, help="Numero de dimension espaciais (2D ou 3D)")
    parser.add_argument("--in_channels", default=1, type=int, help="Input image channels (i.e. 3 for color images, 1 for gray images)")
    parser.add_argument("--out_channels", default=14, type=int, help="Number of classes")
    parser.add_argument("--data_dir", default='../Datasets/BTCV_/', type=str, help="Training data directory")
    parser.add_argument("--mode", default='Predict', type=str, help="Work mode (Train, Test, Predict)")
    parser.add_argument("--trainmode", default='init', type=str, help="Continue training from checkpoint (cont) or start from scratch (init)")
    parser.add_argument("--roi_size", default=(96, 96, 96), type=tuple, help="Slide window size for inference")
    parser.add_argument("--inference_batch_size", default=1, type=int, help="Batch size for inference")
    parser.add_argument('--folders_img_lbl', type=bool, default=True, help="If images and labels are in different folders")
    
    parser.add_argument("--show", default=False, type=boolean, help="Visualizar resultados on-line")

    parser.add_argument('--model', type=str, default='segformer', help="Network model name. Available models: unet, unetr, swin_unetr, unet++, attention_unet, resunet, medformer, vnet, segformer")
    parser.add_argument('--dimension', type=str, default='3d', help="Dimension of the model (2d or 3d)")
    parser.add_argument('--dataset', type=str, default='btcv', help="Name of the dataset it can be amos (Abdominal Multi-Organ Segmentation)")
    parser.add_argument('--run_version', type=int, default=24, help="Version of the checkpoint for testing or predicting")
    parser.add_argument('--path_prediction', type=str, default="./results/", help="Path to save the predictions")

    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    

    return args




if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.empty_cache()
    
    args = get_parser()

    module = MainModule(args)
    module()