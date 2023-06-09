from torch_directional import FourierDistribution
import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import IPython
from math import pi
import matplotlib.pyplot as plt
from torch.fft import rfft
import pandas as pd
import skimage.io
import os
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from lenet5_no_log_softmax import LeNet5NoLogSoftmax
import argparse
from distutils import util

torch.autograd.set_detect_anomaly(True)

class OrientationsWithSymmDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_modes = 6):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mean = None
        self.var = None
        self.max_modes = max_modes

    def __len__(self):
        return len(self.data_frame)
    
    def set_mean_and_std(self, mean, std):
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        im = skimage.io.imread(img_name)
        im = torch.from_numpy(im).float()
        if not self.mean == None:
            im = (im-self.mean)/self.std
        
        base_ori = torch.tensor(self.data_frame.iloc[idx, 1])
        n_modes = torch.tensor(self.data_frame.iloc[idx, 2])
        assert n_modes<=self.max_modes
        all_oris_padded = torch.zeros(self.max_modes)
        all_oris_padded[0:n_modes] = torch.fmod(torch.linspace(base_ori,base_ori+2*pi,n_modes+1)[:-1],2*pi)

        return [im,all_oris_padded,n_modes]

class SimpleDataSplitDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fulldataset = dataset

    def setup(self, stage):
        noTrain = int(self.fulldataset.__len__()*0.75)
        noVal = int(self.fulldataset.__len__()*0.25)
        self.train_data, self.val_data = random_split(self.fulldataset, [noTrain, noVal])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,num_workers=self.num_workers)

    def test_dataloader(self):
        raise("Test is done using freshly generated data in Matlab.")

class LikelihoodModel(pl.LightningModule):
    
    def __init__(self, n_coeffs, use_real=True, lr = 1e-3):
        super().__init__()
        self.n_coeffs = n_coeffs
        self.lr = lr
        self.net = LeNet5NoLogSoftmax(n_coeffs)
        self.use_real = use_real
        self.val_step_plot_index = 2147483647 # Prevent saving image on initial validation test
        self.save_hyperparameters()


    def forward(self, x):
        return self.net(F.pad(x.view(x.shape[0],1,24,24),(4,4,4,4)))
        
    def training_step(self, batch, batch_idx):
        # training_step for the training loop
        x = batch[0].view(batch[0].shape[0],-1)
        y = batch[1]
        n_symm = batch[2]
        output = self(x)
        
        # Calculate trig moms of WD
        kmax = (self.n_coeffs-1)//2
        ind_range = torch.arange(kmax+1).type_as(y)
        for_comp_exp = ind_range.view(1,-1,1)*y.unsqueeze(1)
        exponent_parts = torch.exp(1j*for_comp_exp)
        trig_moms = (torch.sum(exponent_parts,dim=2)-(y.shape[1]-n_symm).unsqueeze(-1))/torch.sqrt(n_symm.unsqueeze(-1))
        
        loss = torch.empty(n_symm.size(0))
        for i in range(y.size(0)):
            if self.use_real:
                # Interpret outputs as a and b
                a = output[i,0:(output.size(1)+1)//2]
                b = output[i,(output.size(1)+1)//2:]
                fd_model_a_b = FourierDistribution(a=a,b=b,transformation='sqrt')
                fd_model_a_b_norm = fd_model_a_b.normalize()

                fd_gt = FourierDistribution(transformation='sqrt',c=torch.conj(trig_moms[i]))
                fd_gt_approx_a_b_norm = fd_gt.to_real_fd().normalize()

                loss[i] = (fd_model_a_b_norm-fd_gt_approx_a_b_norm).integral()
                
            else:
                c = output[i,0:(output.size(1)+1)//2] + 1j*torch.cat((torch.zeros(1).type_as(output),output[i,(output.size(1)+1)//2:]),dim=-1)
                fd_model = FourierDistribution(c=c,transformation='sqrt',n=self.n_coeffs)
                fd_model_norm = fd_model.normalize()

                fd_gt_c = FourierDistribution(transformation='sqrt',c=torch.conj(trig_moms[i])*self.n_coeffs,n=self.n_coeffs,multiplied_by_n=True)
                fd_gt_c_norm = fd_gt_c.normalize()
                
                loss[i] = (fd_model_norm-fd_gt_c_norm).integral()
        
        if loss.isnan().any():
            print(loss)
            raise Exception('This should not happen.')

        self.log("train_loss",loss.mean())
        return loss.mean()

    def on_validation_epoch_start(self):
        if self.trainer.current_epoch > 0:
            self.val_step_plot_index = torch.randint(low=1, high=self.val_dataloader.__sizeof__(), size=(1,))
    
    def validation_step(self, batch, batch_idx):
        x = batch[0].view(batch[0].shape[0],-1)
        y = batch[1]
        n_symm = batch[2]
        output = self(x)
        # Calculate trig moms
        kmax = (self.n_coeffs-1)//2
        ind_range = torch.arange(kmax+1).type_as(y)
        for_comp_exp = ind_range.view(1,-1,1)*y.unsqueeze(1)
        exponent_parts = torch.exp(1j*for_comp_exp)
        trig_moms = (torch.sum(exponent_parts,dim=2)-(y.shape[1]-n_symm).unsqueeze(-1))/torch.sqrt(n_symm.unsqueeze(-1))
        
        hellinger_like_loss = torch.empty(n_symm.size(0))
        for i in range(y.size(0)):
            if self.use_real:
                # Interpret outputs as a and b
                a = output[i,0:(output.size(1)+1)//2]
                b = output[i,(output.size(1)+1)//2:]
                fd_model_a_b = FourierDistribution(a=a,b=b,transformation='sqrt')
                fd_model_a_b_norm = fd_model_a_b.normalize()

                fd_gt = FourierDistribution(transformation='sqrt',c=torch.conj(trig_moms[i]))
                fd_gt_approx_a_b_norm = fd_gt.to_real_fd().normalize()

                hellinger_like_loss[i] = (fd_model_a_b_norm-fd_gt_approx_a_b_norm).integral()
                
            else:
                c = output[i,0:(output.size(1)+1)//2] + 1j*torch.cat((torch.zeros(1).type_as(output),output[i,(output.size(1)+1)//2:]),dim=-1)
                fd_model = FourierDistribution(c=c,transformation='sqrt',n=self.n_coeffs)
                fd_model_norm = fd_model.normalize()

                fd_gt_c = FourierDistribution(transformation='sqrt',c=torch.conj(trig_moms[i])*self.n_coeffs,n=self.n_coeffs,multiplied_by_n=True)
                fd_gt_c_norm = fd_gt_c.normalize()
                
                hellinger_like_loss[i] = (fd_model_norm-fd_gt_c_norm).integral()
        
        if hellinger_like_loss.isnan().any():
            print(hellinger_like_loss)
            raise Exception('This should not happen.')
        
        loss_likelihood = torch.empty(n_symm.shape).type_as(x)
        for i in range(x.shape[0]):
            if self.use_real:
                a = output[i,0:(output.size(1)+1)//2]
                b = output[i,(output.size(1)+1)//2:]
                fd_model = FourierDistribution(a=a,b=b,transformation='sqrt')
            else:
                c = output[i,0:(output.size(1)+1)//2] + 1j*torch.cat((torch.zeros(1).type_as(output),output[i,(output.size(1)+1)//2:]),dim=-1)
                fd_model = FourierDistribution(c=c,transformation='sqrt',n=self.n_coeffs)
            fd_model_norm = fd_model.normalize()
            dist = fd_model_norm
            pdfs = dist.pdf(y[i,0:n_symm[i]].unsqueeze(1))
            # Seem to work either both or not
            loss_likelihood[i] = - (torch.prod(pdfs)**(1/n_symm[i].float()) * n_symm[i].float())

        if batch_idx==self.val_step_plot_index: #(self.current_epoch %20):#0:
            plot_status(dist,n_symm[-1],y[-1,:])
            self.logger.experiment.add_figure('pdf',plt.gcf(),self.current_epoch)
            self.logger.experiment.add_image('image',x[-1,:].view(24,24),self.current_epoch,dataformats="HW")

        self.log('val_loss_likelihood', loss_likelihood.mean())
        self.log('val_loss_hellinger_like', hellinger_like_loss.mean())

        self.log('hp_metric', loss_likelihood.mean())
        return loss_likelihood.mean()
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            'monitor': 'train_loss'
        }

def plot_status(dist,n_points_curr,y_curr):
    IPython.display.clear_output(wait=True)
    plt.cla()
    dist.plot_grid()
    max_val = dist.plot()
    for y_to_plot in y_curr[0:n_points_curr]:
        plt.plot([y_to_plot.detach().cpu(),y_to_plot.detach().cpu()],[0,max_val.detach().cpu()],':')
    IPython.display.display(plt.gcf())


class SqrtLayer(nn.Module):
    def forward(self, x):        
        return torch.sqrt(x)

class RfftLayer(nn.Module):
    def forward(self, x):        
        return rfft(x)
class NormalizeRealFourierCoeffsLayer(nn.Module):
    def forward(self,x):
        pass

performance_params = {
    "gpus" : -1,
    "num_workers": 0, # for dataloader. 0 means main thread does the work, currently works better most of the time since there is not so much data
    "profiler": 'pytorch',
    "benchmark": True,
    "auto_select_gpus": False,
    "accelerator": 'ddp',
}

if not torch.cuda.is_available(): # Prevent code from failing if no Nvidia GPU or GPU support is installed
    del performance_params["gpus"]
    del performance_params["accelerator"]
    del performance_params["auto_select_gpus"]

debug_params = {
    "deterministic": False,
    "overfit_batches" : 0.0, # To limit to subset. Using full data set with 0.0
}

scenarioParams = {
    "folder": os.path.join("..","Dataset"),
    "standardize": True,
}

default_hparams ={
    "n_coeffs": 81,
    "epochs": 150,
    "use_real": False,
    "lr": 0.0005,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set mode and parameters.')
    parser.add_argument('--net_path', nargs='?', const=1, help='Path to resume from.')
    parser.add_argument('--n_coeffs', nargs='?', const=1, type=int, default=default_hparams['n_coeffs'])
    parser.add_argument('--epochs', nargs='?', const=1, type=int, default=default_hparams['epochs'])
    parser.add_argument('--use_real', nargs='?', const=1, type=util.strtobool, default=default_hparams['use_real'])
    parser.add_argument('--lr', nargs='?', const=1, type=float, default=default_hparams['lr'])
    args = parser.parse_args()
    if debug_params["overfit_batches"] != 0.0:
        input('You are overfitting on batches. Only use this for bebugging and not to get a working network. Press enter to confirm.')

    dataset = OrientationsWithSymmDataset(csv_file=os.path.join(scenarioParams["folder"],"groundtruths.csv"),root_dir=scenarioParams["folder"])
    if scenarioParams["standardize"]:
        loaderTmp = DataLoader(dataset, batch_size=dataset.__len__())
        all_items, _, _= next(iter(loaderTmp))
        mean = all_items.mean()
        std = all_items.std()
        dataset.set_mean_and_std(mean,std)

    hparams = {"n_coeffs": args.n_coeffs, "use_real" : args.use_real, "lr" : args.lr}
    if args.net_path==None:
        current_network = LikelihoodModel(**hparams)
    else:
        current_network = LikelihoodModel.load_from_checkpoint(args.net_path, **hparams)
    
    datamodule = SimpleDataSplitDataModule(batch_size=4,dataset=dataset,num_workers=performance_params.pop('num_workers'))
    log_dir_name = os.path.basename(scenarioParams["folder"])+str(args.n_coeffs)+'coFourier'
    if args.use_real:
        log_dir_name = log_dir_name + 'Real'
    else:
        log_dir_name = log_dir_name + 'Complex'
    print(log_dir_name)
    logger = TensorBoardLogger(save_dir="lightning_logs",name=log_dir_name,log_graph=True)
    trainer = pl.Trainer(max_epochs=args.epochs,logger=logger, reload_dataloaders_every_epoch=False,**performance_params,**debug_params)
    trainer.fit(current_network, datamodule)