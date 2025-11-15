from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import peak_signal_noise_ratio
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from tqdm import tqdm

from .vae import elbo_loss


def validate(model: torch.nn.Module,
             epoch: int,
             log_writer: SummaryWriter,
             dataloader: torch.utils.data.DataLoader,
             test: bool=False
):
    model.eval()
    with torch.no_grad():
        elbo_losses = []
        psnr_scores = []
        mse_scores = []
        for i, (x, _) in enumerate(dataloader):
            x = x.cuda()
            x_hat, mus, log_vars = model(x)
            elbo_losses.append(elbo_loss(x, x_hat, mus, log_vars))
            psnr_scores.append(peak_signal_noise_ratio(x_hat, x))
            mse_scores.append(F.mse_loss(x_hat, x))
            if i == 0:
                x_grid = torchvision.utils.make_grid(
                                x[:8].unsqueeze(0), nrow=2, normalize=True, scale_each=True)
                x_hat_grid = torchvision.utils.make_grid(
                                x_hat[:8].unsqueeze(0), nrow=2, normalize=True, scale_each=True)
                log_writer.add_images('Validation input Images', x_grid, epoch)
                log_writer.add_images('Validation reconstructed Images', x_hat_grid, epoch)
        avg_elbo = torch.stack(elbo_losses).mean()
        avg_mse_score = torch.stack(mse_scores).mean()
        avg_psnr_score = torch.stack(psnr_scores).mean()
        # We have the same loop for validation and testing. Adapt the log names accordingly.
        run_type = 'Test' if test else 'Validation'        
        log_writer.add_scalar(f'{run_type} ELBO', avg_elbo, epoch)
        log_writer.add_scalar(f'{run_type} MSE', avg_mse_score, epoch)
        log_writer.add_scalar(f'{run_type} PSNR', avg_psnr_score, epoch)
        # print(f'    {run_type} metrics, averaged over validation set: '
        #         f'MSE: {avg_mse_score:0.6f}, '
        #         f'PSNR: {avg_psnr_score:0.6f}')


def test(
        model: torch.nn.Module, log_writer: SummaryWriter, batch_size: int=512, num_workers: int=2
):
    log_writer = SummaryWriter()
    transform = T.Compose([T.ToTensor()])
    test_set = CIFAR10(root='./data', download=False, train=False, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(
                    test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    validate(model, 0, log_writer, test_dataloader, test=True)
    print('Testing Done.')


def train(model: torch.nn.Module,
          log_writer: SummaryWriter,
          checkpoint_dir: str,
          pretrained_path: str=None,
          num_epochs: int=1000,
          batch_size: int=512,
          num_workers: int=2,
          val_freq: int=100,
          learn_rate: float=1e-3,
          overfit: bool=False
) -> torch.nn.Module:
    """Train the model, log metrics and save checkpoints.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    log_writer : SummaryWriter
        Tensorboard logger.
    checkpoint_dir: str
        Directory to save checkpoints to.
    pretrained_path: str, optional
        Path to a pre-trained checkpoint, by default None
    num_epochs : int, optional
        Number of epochs to train. If there is a pretrained checkpoint given,
        these epochs are added on top of the checkpoint's epochs. By default 1000.
    batch_size : int, optional
        by default 512
    num_workers : int, optional
        by default 2
    val_freq : int, optional
        Validate every *val_freq* epochs., by default 100

    Returns
    -------
    torch.nn.Module
        Trained model.
    """
    checkpoint_freq = 1000
    transform = T.Compose([T.ToTensor()])
    train_set = CIFAR10(root='./data', download=True, train=True, transform=transform)
    val_set = CIFAR10(root='./data', download=False, train=True, transform=transform)
    # 80% train, 20% validation
    split = int(len(train_set) * 0.8)
    overfit_size = 8
    batch_size = batch_size if not overfit else overfit_size
    train_set.data = train_set.data[:split] if not overfit else train_set.data[:overfit_size]
    val_set.data = val_set.data[split:]
    train_dataloader = torch.utils.data.DataLoader(
                    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(
                    val_set,batch_size=batch_size, shuffle=False, num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    pre_trained_epoch = 0
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
        pre_trained_epoch = int(Path(pretrained_path).stem.split('_')[-1]) + 1
    with tqdm(desc=f'Training...',
              total=pre_trained_epoch + num_epochs,
              initial=pre_trained_epoch
    ) as pbar:
        for epoch in range(pre_trained_epoch, pre_trained_epoch + num_epochs):
            model.train()
            epoch_losses = []
            for i, (x, _) in enumerate(train_dataloader):
                x = x.cuda()
                optimizer.zero_grad()
                x_hat, mus, log_vars = model(x)
                loss = elbo_loss(x, x_hat, mus, log_vars)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss)
            log_writer.add_scalar('Train ELBO loss', torch.stack(epoch_losses).mean(), epoch)
            if epoch % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():0.6f}'})
            if epoch % val_freq == 0:
                x_grid = torchvision.utils.make_grid(
                                x[:8].unsqueeze(0), nrow=2, normalize=True, scale_each=True)
                x_hat_grid = torchvision.utils.make_grid(
                                x_hat[:8].unsqueeze(0), nrow=2, normalize=True, scale_each=True)
                log_writer.add_images('Train input Images', x_grid, epoch)
                log_writer.add_images('Train reconstructed Images', x_hat_grid, epoch)
                validate(model, epoch, log_writer, val_dataloader)
            if epoch % checkpoint_freq == 0:
                save_path = Path(checkpoint_dir) / f'{datetime.now()}_vae_{epoch}.pt'
                torch.save(model.state_dict(), save_path)                            
            pbar.update(1)
    log_writer.flush()
    log_writer.close()
    save_path = Path(checkpoint_dir) / f'{datetime.now()}_vae_{epoch}.pt'
    torch.save(model.state_dict(), save_path)
    print('Training Done.')
    model.eval()
    return model
