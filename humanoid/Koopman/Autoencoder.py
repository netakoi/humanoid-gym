import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim): 
        super(Encoder, self).__init__()

        print('Tanh ver')
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, state_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(hidden_dim, state_dim, bias=False)

    def forward(self, x):
        return self.linear(x)
    

class KoopmanAutoencoder(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(KoopmanAutoencoder, self).__init__()
        self.encoder = Encoder(state_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, state_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        # self.K = torch.randn(hidden_dim+state_dim, hidden_dim+state_dim)  
        self.K = torch.randn(hidden_dim, hidden_dim)  
    
    def forward(self, x):

        z = self.encoder(x)  
        
        if self.K is not None:
            z_next = torch.matmul(z, self.K.T)  # Apply computed Koopman operator
        else:
            z_next = z  

        x_hat = self.decoder(z)  
        return x_hat, z, z_next
        
    def compute_koopman_operator(self, latent_X, latent_Y):
        X_pseudo_inv = torch.linalg.pinv(latent_X)  # Compute pseudo-inverse of latent_X
        # # ###### REPLACE PINV
        # U, S, Vh = torch.linalg.svd(latent_X, full_matrices=False, driver='gesvda')
        # S_inv = 1.0 / S
        # X_pseudo_inv = Vh.T @ torch.diag(S_inv) @ U.T
        # ####################################
        self.K = torch.matmul(latent_Y.T, X_pseudo_inv.T)  # K = Y * X^+
