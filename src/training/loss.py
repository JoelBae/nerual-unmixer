import torch
import torch.nn as nn
import torch.distributions as D
import torchaudio.transforms as T

class HybridLoss(nn.Module):
    """
    Implements the Hybrid Loss Function from the v3 spec.
    L_total = lambda_p * NLL(MDN) + lambda_s * L_Spectral(G_proxy(theta_sample), x)
    """
    def __init__(self, lambda_p=1.0, lambda_s=1.0, sample_rate=44100,
                 fft_sizes=[2048, 1024, 512, 256, 128, 64],
                 hop_sizes=[512, 256, 128, 64, 32, 16],
                 win_lengths=[2048, 1024, 512, 256, 128, 64]):
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, hs, wl))

    def mdn_loss(self, pi, sigma, mu, target):
        """
        Computes Negative Log-Likelihood for MDN.
        target: (batch_size, output_dim)
        """
        # Create Mixture distribution
        # We need to broadcast target to match mixture components
        # target: (batch, 1, dim)
        target = target.unsqueeze(1)
        
        # Normal distribution for each component: N(target | mu, sigma)
        # mu, sigma: (batch, clusters, dim)
        m = D.Normal(loc=mu, scale=sigma)
        
        # Log prob of target under each component
        # log_prob: (batch, clusters, dim)
        log_prob = m.log_prob(target)
        
        # Sum over dimensions (assuming independence across parameter dimensions given the component)
        # log_prob: (batch, clusters)
        log_prob = torch.sum(log_prob, dim=2)
        
        # Combine with mixing coefficients
        # P(y|x) = sum_k pi_k * N(...)
        # LogSumExp trick for numerical stability:
        # log(P) = log(sum(exp(log_pi + log_prob_component)))
        weighted_log_prob = torch.log(pi + 1e-8) + log_prob
        
        # Final NLL
        loss = -torch.logsumexp(weighted_log_prob, dim=1)
        return torch.mean(loss)

    def multi_scale_stft_loss(self, predicted_audio, target_audio):
        """
        Computes Multi-Scale STFT Loss.
        """
        loss = 0.0
        for f in self.stft_losses:
            loss += f(predicted_audio, target_audio)
        return loss

    def forward(self, mdn_out, target_params, proxy, target_audio):
        """
        mdn_out: (pi, sigma, mu)
        target_params: Ground truth parameters
        proxy: NeuralProxy model
        target_audio: Ground truth audio
        """
        pi, sigma, mu = mdn_out
        
        # 1. MDN Loss (NLL)
        loss_nll = self.mdn_loss(pi, sigma, mu, target_params)
        
        # 2. Spectral Loss with Reparameterization
        # Sample theta from the predicted distribution
        # Gumbel-Softmax for cluster selection (Categorical)
        # Reparameterized Normal for parameter value
        
        # Step A: Select component k using Gumbel-Softmax (Approximation)
        # For simplicity in this v1, we can just use the mean of the most likely component 
        # OR sample using the Reparameterization trick for the Normal part.
        
        # Fully Differentiable Sampling:
        # theta = sum_k (softmax_gumbel_prob_k * (mu_k + sigma_k * epsilon))
        
        # Gumbel-Softmax for hard/soft selection
        gumbel_pi = torch.nn.functional.gumbel_softmax(torch.log(pi + 1e-8), tau=1, hard=False)
        
        # Reparameterize Normal
        epsilon = torch.randn_like(mu)
        z_k = mu + sigma * epsilon
        
        # Weighted sum of sampled values from all clusters (Soft Mixture)
        # theta_sample: (batch, dim)
        theta_sample = torch.sum(gumbel_pi.unsqueeze(-1) * z_k, dim=1)
        
        # Pass through Proxy
        pred_audio = proxy(theta_sample)
        
        # Multi-Scale Spectral Loss
        loss_spec = self.multi_scale_stft_loss(pred_audio, target_audio)
        
        total_loss = self.lambda_p * loss_nll + self.lambda_s * loss_spec
        
        return total_loss, {"nll": loss_nll.item(), "spectral": loss_spec.item()}

class STFTLoss(nn.Module):
    """
    Single-Scale STFT Loss (Spectral Convergence + Log Magnitude).
    """
    def __init__(self, fft_size, hop_size, win_length):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def forward(self, pred, target):
        # Ensure correct shape (batch, samples)
        if pred.dim() == 3: pred = pred.squeeze(1)
        if target.dim() == 3: target = target.squeeze(1)

        # Move window to correct device
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)

        # Compute STFT
        pred_stft = torch.stft(pred, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True)
        target_stft = torch.stft(target, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True)

        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        # Safe log
        pred_log = torch.log(pred_mag + 1e-7)
        target_log = torch.log(target_mag + 1e-7)

        # Spectral Convergence Loss
        sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-7)
        
        # Log Magnitude Loss
        mag_loss = nn.L1Loss()(pred_log, target_log)

        return sc_loss + mag_loss
