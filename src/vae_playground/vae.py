import torch
import torch.nn.functional as F
import einops


class VAE(torch.nn.Module):
    def __init__(
            self, img_size: int=32, input_dim: int=3, hidden_dims: list=[16, 32, 64], latent_dim: int=64
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Keep track of the spatial dims for the linear layer.
        # This is required when working with the flattened output of the conv layers.
        # For each conv layer, the spatial dims are halved.        
        # Assumptions: square images, stride=2, padding=1.
        self.spatial_dims_inner_conv = img_size // (2 ** len(hidden_dims))
        print(f'Linear spatial dims: {self.spatial_dims_inner_conv}')

        encoder_layers = []
        in_channels = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(
                torch.nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(torch.nn.ReLU())
            in_channels = h_dim        
        self.encoder = torch.nn.Sequential(*encoder_layers)            

        # The input size is the number of channels of the last conv layer 
        # times the spatial dims squared.
        self.linear_mu = torch.nn.Linear(self.hidden_dims[-1] * self.spatial_dims_inner_conv**2,
                                         self.latent_dim)
        self.linear_log_var= torch.nn.Linear(self.hidden_dims[-1] * self.spatial_dims_inner_conv**2,
                                         self.latent_dim)
            

        decoder_layers = []
        in_channels = hidden_dims[-1]
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_hidden_dims) - 1):
            decoder_layers.append(
                torch.nn.ConvTranspose2d(reversed_hidden_dims[i], 
                                         reversed_hidden_dims[i+1],
                                         kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(torch.nn.ReLU())        
        self.decoder = torch.nn.Sequential(*decoder_layers, 
                                           torch.nn.ConvTranspose2d(reversed_hidden_dims[-1], 
                                                                    input_dim,
                                                                    kernel_size=3, 
                                                                    stride=2, 
                                                                    padding=1,
                                                                    output_padding=1),
                                           torch.nn.Tanh())
        self.linear_decoder = torch.nn.Linear(latent_dim, 
                                              hidden_dims[-1] * self.spatial_dims_inner_conv**2)
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        # Keep the batch dim unchanged.
        x = einops.rearrange(x, 'b c h w -> b (c h w)')
        mus = self.linear_mu(x)
        log_vars = self.linear_log_var(x)
        return mus, log_vars
    
    def draw_sample(self, mus: torch.Tensor, log_vars: torch.Tensor):
        eps = torch.randn_like(log_vars)
        # Reparameterization trick:
        # Any normal distribution can be constructed by using 
        # a standard normal distribution (epsilon), scaling it 
        # by the standard deviation (sigma) and then shifting by the mean (mu).
        # We assume a natural logarithm. The point five comes from the square root of the variance.
        z = mus + torch.exp(0.5 * log_vars) * eps
        return z
    
    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_hat = self.linear_decoder(z)
        x_hat = einops.rearrange(x_hat, 'b (c h w) -> b c h w',
                                 c=self.hidden_dims[-1],
                                 h=self.spatial_dims_inner_conv,
                                 w=self.spatial_dims_inner_conv)
        x_hat = self.decoder(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
           x_hat, mus, log_vars. x_hat has range [0,1]
        """
        # Mu and sigma represent the parameters of n = mus.shape[-1] normal distributions.
        mus, log_vars = self.encode(x)
        # Of these n normal distributions, we draw n samples.
        z = self.draw_sample(mus, log_vars)
        # x_hat has range [-1,1]
        x_hat = self.decode(z)
        # Rescale the output to [0, 1] to match the scale of the input data.
        # Relevant for the MSE loss.
        x_hat = (x_hat + 1) / 2
        return x_hat, mus, log_vars


def elbo_loss(
        x: torch.Tensor, x_hat: torch.Tensor, mus: torch.Tensor, log_vars: torch.Tensor
) -> torch.Tensor:
    """Calculates the ELBO loss.

    Parameters
    ----------
    x : torch.Tensor
        Z-score normalized input with range [0,1].
    x_hat : torch.Tensor
        Reconstruction of the input images in range [0,1].
    mus : torch.Tensor
        Mu values of the latent space.
    log_vars : torch.Tensor
        Sigma values of the latent space.

    Returns
    -------
    torch.Tensor        
    """
    mse = F.mse_loss(x_hat, x, reduction='none')
    # We want the distribution of the latent space to be as close as possible to a standard normal distribution.    
    # Taken from https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vanilla_vae.py#L143C105-L143C105
    # KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    # Derivation: https://github.com/AntixK/PyTorch-VAE/issues/69
    # The derived formula using the log_var is better, because it allows for the values to be 
    # negative during training. When treating them as Sigma, they would have to be positive,
    # which is hard to enforce.
    d_kl = torch.mean(-0.5 * torch.sum(1 + log_vars - mus ** 2 - log_vars.exp(), dim = 1), dim = 0)
    beta = 1
    elbo = mse + beta * d_kl
    return elbo.mean()
