import matplotlib.pyplot as plt
import numpy as np
import torch
from tools import progress

import matplotlib.pyplot as plt
import numpy as np
import torch
from tools import progress

from tools import progress

from tools import progress

class VariationalEncoder(torch.nn.Module):
    """
    Conditional VAE Encoder with <layers>+1 fully connected layer
    """
    def __init__(self, in_dims, hidden_dims=[128, 128, 64, 32], latent_dims=5, layers=4, dropout=0, device='cuda'):
        super().__init__()
        self.linears = [torch.nn.Sequential(
                torch.nn.Linear(in_dims, hidden_dims[0]),
                torch.nn.LayerNorm(hidden_dims[0]),
                torch.nn.Dropout(p=dropout))]
        self.add_module('linear0', self.linears[-1])
        for i in range(1, layers):
            self.linears += [torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                torch.nn.LayerNorm(hidden_dims[i]),
                torch.nn.Dropout(p=dropout))
                ]
            self.add_module('linear%d' % i, self.linears[-1])
        self.linear_mean = torch.nn.Linear(hidden_dims[-1], latent_dims)
        self.linear_logstd = torch.nn.Linear(hidden_dims[-1], latent_dims) # log of actual, later exponentiate

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        
        self.device = device
        
    def forward(self, x, return_latent=False):
        if(len(x.shape) > 1):
            x = torch.flatten(x, start_dim=1)
        for linear in self.linears:
            x = torch.nn.functional.relu(linear(x))
        mu = self.linear_mean(x) # mu is g(l(x))
        if return_latent:
            return mu
        else:
            sigma = torch.exp(self.linear_logstd(x)) # sigma is h(l(x))
            z = mu + sigma * self.N.sample(mu.shape) # reparameterization trick
            self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
            return z


class Decoder(torch.nn.Module):
    """
    Conditional VAE Decoder with <layers>+1 fully connected layer
    """
    def __init__(self, out_dims, hidden_dims=[32, 64, 128, 128], latent_dims=3, layers=4, dropout=0):
        super().__init__()
        self.linears = [torch.nn.Sequential(
                torch.nn.Linear(latent_dims, hidden_dims[0]),
                torch.nn.LayerNorm(hidden_dims[0]),
                torch.nn.Dropout(p=dropout))]
        self.add_module('linear0', self.linears[-1])
        
        for i in range(1, layers):
            self.linears += [torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                torch.nn.LayerNorm(hidden_dims[i]),
                torch.nn.Dropout(p=dropout))
                ]
            self.add_module('linear%d' % i, self.linears[-1])        

        self.final_linear1 = torch.nn.Linear(hidden_dims[-1], out_dims)
        # self.final_linear2 = torch.nn.Linear(hidden_dims, out_dims)
        self.final_log_std1 = torch.nn.Linear(hidden_dims[-1], out_dims)
        # self.final_log_std2 = torch.nn.Linear(hidden_dims, out_dims)
        # self.final_lambda = torch.nn.Linear(hidden_dims, 1)
        # self.final_prob_zero = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, z): # outputs distribution
        for linear in self.linears:
            z = torch.nn.functional.relu(linear(z))
        m1 = self.final_linear1(z)
        # m2 = self.final_linear2(z)
        s1 = torch.exp(self.final_log_std1(z))
        # s2 = torch.exp(self.final_std2(z))
        # lam = self.final_lambda
        # p0 = torch.sigmoid(self.final_prob_zero(z))
        return m1, s1


class VariationalEncoderDecoder(torch.nn.Module):
    def __init__(self, beta=0.01, data_dims=124, label_dims=128,
                 latent_dims=5, hidden_dims_input=[128, 128, 64, 32], hidden_dims_output = [32, 64, 128, 128], dropout=0, device='cuda'):
        """
        Conditional VAE
        Encoder: [y x] -> [mu/sigma] -sample-> [z]
        Decoder: [z x] -> [y_hat]

        Inputs:
        -------
        beta - [float] trade-off between KL divergence (latent space structure) and reconstruction loss
        data_dims - [int] size of x
        label_dims - [int] size of y
        latent_dims - [int] size of z
        hidden_dims - [list::int] size of each hidden layers, for both input and output
        layers - [int] number of layers, including hidden layer
        
        """
        super().__init__()
        self.data_dims = data_dims
        self.label_dims = label_dims
        self.latent_dims = latent_dims
        self.label_dims = label_dims
        # Encoder - compresses data into latent dims
        self.encoder = VariationalEncoder(data_dims, hidden_dims_input, latent_dims, layers=len(hidden_dims_input), dropout=dropout, device=device).to(device)
        # Decoder - tries to learn both input and output variables. 
        self.decoder = Decoder(label_dims + data_dims, hidden_dims_output, latent_dims, layers=len(hidden_dims_output), dropout=dropout).to(device)
        self.beta = beta
        self.device = device
        
        self.losses = []

    def forward(self, x, y, return_latent=False, batch_norm=False):
        # expect to be batch_size x data_dims matrix and y to be batch_size x label_dims matrix
        
        # Normalize
        if batch_norm:
            x_m, x_s = x.mean(axis=0), x.std(axis=0)
            y_m, y_s = y.mean(axis=0), y.std(axis=0)
            mx, my = x_s != 0, y_s != 0
            x[:, mx] = x[:, mx] / x_s[mx] - x_m[mx]
            y[:, my] = y[:, my] / y_s[my] - y_m[my]
        z = self.encoder(x, return_latent)
        if return_latent:
            return z
        else:
            y_hat_mean, y_hat_std = self.decoder(z)
            if batch_norm:
                y_hat_mean = (y_hat_mean + y_m) * y_s # undo the batch norm for output
            return y_hat_mean, y_hat_std


    def trainer(self, data, save_params, train_params):
        """
        Train the Conditional VAE

        Inputs:
        -------
        data - [DataLoader] - training data
        epochs - [int] number of epochs
        loss_type - [str] type of loss
        optimizer - [str] type of optimizer
        lr - [float] learning rate
        weight_decay - [float] L2 regularization
        save - [str] file path to save trained model to after training (and after every 20 minutes)
        plot - [boolean] if plots of loss curves and samples should be produced
        """
        
        
        # Training parameters
        if train_params['optimizer'] == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])
        elif train_params['optimizer'] == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])
        else:
            raise ValueError('Unknown optimizer')

        # Train and checkpoint every 20 minutes
        
        if(save_params['save']):
            # calls lambda function every 1200 seconds and at the end
            path = os.path.join(save_params['model_path'], save_params['name']) + save_params['filetype']
            timed=[(save_params['save_interval'], lambda: torch.save(self.state_dict(), path))]
        else:
            timed = None
        
        losses = []
        KL_div = []
        for epoch, (x,y) in progress(range(train_params['epochs']), inner=data, text='Training', timed=timed):
            #x = x.to(self.device)
            #y = y.to(self.device)
            O_real = torch.cat([x, y], 1).to(self.device) # actual output desired 
            O_mean, O_std = self(x, y)
            #y_mean, y_std = self(y, x)
            # y_mean, y_std, p0 = self(y, x)
            if train_params['loss_type'] == 'mse':
                # iid gaussians -> mse
                # loss = ((y - y_hat) ** 2).sum() / self.label_dims + self.beta * self.encoder.kl / self.latent_dims

                # means, so beta' = beta * label_dims / latent_dims
                
                rec_loss = 0.5 * (O_real - O_mean) ** 2 + torch.log(O_std)
                loss = rec_loss.mean() + self.beta * self.encoder.kl
                #loss = (0.5 * (y - y_mean) ** 2 / y_std + torch.log(y_std)).mean() + self.beta * self.encoder.kl
                # model as p0 * N(0, 1/1000) + (1-p0) * N(mean, std)
                # loss = (p0 * y**2).sum() + ((1 - p0) * ((y - y_mean) ** 2 / y_std + torch.log(y_std))).mean() + self.beta * self.encoder.kl
            else:
                raise ValueError('Unknown loss')

            torch.clip(loss, min=-1e5, max=1e5).backward() # backpropagate loss
            KL_div.append(self.encoder.kl)
            losses.append(loss.item()) 
            opt.step()
            opt.zero_grad() # VERY IMPORTANT! Reset gradients. 
        
        print('Last-epoch loss: %.2f' % sum(losses[-len(data):-1]))
        print('Finished Training')

        if save_params['plot']:
            fig_path = os.path.join(save_params['figure_path'], save_params['name'])
            O_hat = O_mean + self.encoder.N.sample([x.shape[0], self.data_dims + self.label_dims]) * O_std
            plt.plot(np.array(losses)[:-1])
            if save_params['savefig']:
                plt.savefig(f'{fig_path}_loss.png')
            fig, ax = plt.subplots(4, 1, sharey=True, figsize=(12, 8))
            ax[0].plot((O_real[0:500] - O_mean[0:500]).detach().cpu().numpy().T, c="C0", alpha=0.3)
            ax[1].plot((O_real[0:500] - O_hat[0:500]).detach().cpu().numpy().T, c="C0", alpha=0.3)
            ax[2].plot((O_real[0:500] - self.sample(O_real.shape[0])).detach().cpu().numpy().T, c="C0", alpha=0.3)
            ax[3].plot((O_real[0:500] - self.sample(O_real.shape[0], random=False)[0]).detach().cpu().numpy().T, c="C0", alpha=0.3)
            ax[0].set_ylabel('y - reconstructed sample')
            ax[1].set_ylabel('y - reconstructed mean')
            ax[2].set_ylabel('y - random sample')
            ax[3].set_ylabel('y - mean')
            ax[0].set_ylim([-0.5, 0.5])
            plt.tight_layout()
            if save_params['savefig']:
                plt.savefig(f'{fig_path}_last_batch.png')
            plt.show()
            plt.close('all')
        return(losses)
    
    

def sample(model, batch_size, random=True):
    """
    Sample from uniform Normal Distribution

    Inputs:
    -------
    x - [BxN array] label
    random - [boolean] if true sample latent variable from prior else use all-zero vector
    """
    if random:
        # Draw from prior
        z = model.encoder.N.sample([batch_size, model.latent_dims])
    else:
        # Set to prior mean, 0 (standard N)
        z = torch.zeros([batch_size, model.latent_dims]).to(model.device)
    mean_y, std_y = model.decoder(z)
    if random:
        # add output noise
        y = mean_y + model.encoder.N.sample(mean_y.shape) * std_y
        # y = torch.zeros_like(mean_y)
        # nz = torch.rand(y.shape).to(self.device) > p0
        # y[nz] = mean_y[nz] + self.encoder.N.sample([(nz == 1).sum()]) * std_y[nz]
        return y
    else:
        return mean_y, std_y
    