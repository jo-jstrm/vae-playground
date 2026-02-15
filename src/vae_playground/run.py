import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import mlflow
import tempfile
from torcheval.metrics.functional import peak_signal_noise_ratio
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from tqdm import tqdm


from vae_playground.vae import elbo_loss, VAE
from vae_playground.config import TrainConfig, DataConfig, TestConfig, Config


def validate(model: torch.nn.Module,
             epoch: int,
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
                                x[:8], nrow=2, normalize=True, scale_each=True)
                x_hat_grid = torchvision.utils.make_grid(
                                x_hat[:8], nrow=2, normalize=True, scale_each=True)
                # Save grids as images and log to MLflow
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    torchvision.utils.save_image(x_grid, f.name)
                    mlflow.log_artifact(f.name, artifact_path='images')
                    os.unlink(f.name)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    torchvision.utils.save_image(x_hat_grid, f.name)
                    mlflow.log_artifact(f.name, artifact_path='images')
                    os.unlink(f.name)
        avg_elbo = torch.stack(elbo_losses).mean()
        avg_mse_score = torch.stack(mse_scores).mean()
        avg_psnr_score = torch.stack(psnr_scores).mean()
        # We have the same loop for validation and testing. Adapt the log names accordingly.
        run_type = 'Test' if test else 'Validation'        
        mlflow.log_metric(f'{run_type} ELBO', float(avg_elbo), step=epoch)
        mlflow.log_metric(f'{run_type} MSE', float(avg_mse_score), step=epoch)
        mlflow.log_metric(f'{run_type} PSNR', float(avg_psnr_score), step=epoch)
        # print(f'    {run_type} metrics, averaged over validation set: '
        #         f'MSE: {avg_mse_score:0.6f}, '
        #         f'PSNR: {avg_psnr_score:0.6f}')


def test(
    model: torch.nn.Module,
    test_cfg: TestConfig,
) -> None:
    """Test the model on the test set.

    Parameters
    ----------
    model : torch.nn.Module
        Model to test.
    config : dict
        Configuration dictionary with testing parameters.
    """
    batch_size = test_cfg.batch_size
    num_workers = test_cfg.num_workers
    transform = T.Compose([T.ToTensor()])
    test_set = CIFAR10(root='./data', download=False, train=False, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(
                    test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    validate(model, 0, test_dataloader, test=True)
    print('Testing Done.')


def train(
    model: torch.nn.Module,
#   checkpoint_dir: str,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    pretrained_path: Optional[str]=None
) -> torch.nn.Module:
    """Train the model, log metrics and save checkpoints.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    checkpoint_dir: str
        Directory to save checkpoints to.
    config_path: str
        Path to YAML configuration file.
    pretrained_path: str, optional
        Path to a pre-trained checkpoint, by default None

    Returns
    -------
    torch.nn.Module
        Trained model.
    """
    # Config is provided via dependency injection as validated sub-configs.
    num_epochs = train_cfg.num_epochs
    batch_size = train_cfg.batch_size
    num_workers = train_cfg.num_workers
    learn_rate = train_cfg.learn_rate
    val_freq = train_cfg.val_freq
    checkpoint_freq = train_cfg.checkpoint_freq
    train_split = data_cfg.train_split
    
    transform = T.Compose([T.ToTensor()])
    train_set = CIFAR10(root='./data', download=True, train=True, transform=transform)
    val_set = CIFAR10(root='./data', download=False, train=True, transform=transform)
    # Train/validation split
    split = int(len(train_set) * train_split)
    batch_size = batch_size
    train_set.data = train_set.data[:split]
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
                epoch_losses.append(loss.detach())
            mlflow.log_metric('Train ELBO loss', float(torch.stack(epoch_losses).mean()), step=epoch)
            if epoch % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():0.6f}'})
            if epoch % val_freq == 0:
                x_grid = torchvision.utils.make_grid(
                    x[:8], nrow=2, normalize=True, scale_each=True
               )
                x_hat_grid = torchvision.utils.make_grid(
                    x_hat[:8], nrow=2, normalize=True, scale_each=True
               )
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    torchvision.utils.save_image(x_grid, f.name)
                    mlflow.log_artifact(f.name, artifact_path='images')
                    os.unlink(f.name)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    torchvision.utils.save_image(x_hat_grid, f.name)
                    mlflow.log_artifact(f.name, artifact_path='images')
                    os.unlink(f.name)
                validate(model, epoch, val_dataloader)
            # if epoch % checkpoint_freq == 0:
                # save_path = Path(checkpoint_dir) / f'{datetime.now()}_vae_{epoch}.pt'
                # torch.save(model.state_dict(), save_path)                            
            pbar.update(1)
    # save_path = Path(checkpoint_dir) / f'{datetime.now()}_vae_{epoch}.pt'
    # torch.save(model.state_dict(), save_path)
    print('Training Done.')
    model.eval()
    return model



def run(cfg: Config) -> None:
    """Training runner."""
    assert torch.cuda.is_available()

    # Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    vae = VAE().to(device)
    # pretrained_path = (checkpoint_dir + pretrained_file) if pretrained_file else None

    mlflow.set_tracking_uri('http://localhost:8080')
    mlflow.set_experiment('VAE CIFAR Training')
    with mlflow.start_run():
        mlflow.log_params({
            'num_epochs': cfg.train.num_epochs,
            'batch_size': cfg.train.batch_size,
            'val_freq': cfg.train.val_freq,
            'learn_rate': cfg.train.learn_rate,
            # 'pretrained': cfg.train.pretrained_file is not None,
        })

        vae = train(vae, train_cfg=cfg.train, data_cfg=cfg.data)

        test(vae, cfg.test)

        mlflow.pytorch.log_model(vae, name="vae_cifar", export_model=True) # pyright: ignore[reportPrivateImportUsage]


    
def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: load config, instantiate model, and start training.

    The function loads the YAML config once, validates it with Pydantic,
    creates the model and checkpoint directory, then calls `train()` with
    only the sub-configs required (lean dependency injection).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_overfit.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint-dir", type=str, default="./data/checkpoints", help="Checkpoint directory")
    parser.add_argument("--pretrained", type=str, default=None, help="Optional path to pretrained checkpoint")
    args = parser.parse_args(argv)

    config = Config().from_yaml(Path(args.config))
    # Call the notebook-style training runner
    run(config)


if __name__ == "__main__":
    main()
