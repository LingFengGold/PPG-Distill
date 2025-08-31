import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import LRScheduler
from torch.special import gammaln


class MixtureLogitGaussianLoss(nn.Module):
    """
    Mixture of Gaussians on the logit scale.

    Model output: (B, 3*n_components, T)
      1) mixture logits: y[:, 0:n_components, :]
      2) mu:             y[:, n_components:2*n_components, :]
      3) log_sigma:      y[:, 2*n_components:3*n_components, :]

    x: (B,T) or (B,1,T).
    """
    def __init__(self, n_components=3, eps=1e-6,
                 clamp_mu=(-10,10), clamp_log_sigma=(-10,10),
                 reduction='mean'):
        super().__init__()
        self.n_components = n_components
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_sigma = clamp_log_sigma
        self.reduction = reduction
        self.LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

    def forward(self, y, x):
#        print(f'{y.shape=} {x.shape=}')
        B, C, T = y.shape
        expected_c = 3*self.n_components
        assert C == expected_c, f"Expected {expected_c} channels, got {C} {y.shape=} {x.shape=}"

        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x)  # (B,T)

        # reshape for broadcasting => (B,1,T)
        logit_x = logit_x.unsqueeze(1)

        # 1) separate mixture logits, mu, log_sigma
        n = self.n_components
        mix_logits = y[:, 0:n, :]                # (B, n, T)
        mu         = y[:, n:2*n, :]             # (B, n, T)
        log_sigma  = y[:, 2*n:3*n, :]           # (B, n, T)

        mu        = torch.clamp(mu,  self.clamp_mu[0],  self.clamp_mu[1])
        log_sigma = torch.clamp(log_sigma, self.clamp_log_sigma[0], self.clamp_log_sigma[1])
        sigma = torch.exp(log_sigma) + self.eps

        # 2) mixture weights
        weights = F.softmax(mix_logits, dim=1)  # (B, n, T)

        # 3) log gaussian in logit domain
        z = (logit_x - mu) / sigma
        log_gaussian = -0.5*(z**2) - (torch.log(sigma) + self.LOG_SQRT_2PI)

        # 4) jacobian
        jacobian = -torch.log(x*(1.0 - x) + self.eps).unsqueeze(1)  # (B,1,T)
        log_comp = log_gaussian + jacobian  # (B,n,T)

        # 5) mixture: logsumexp( log(weights) + log_comp, dim=1 )
        log_mixture = torch.log(weights + self.eps) + log_comp
        log_prob = torch.logsumexp(log_mixture, dim=1)  # (B,T)

        nll = -log_prob
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class MixtureLogitCauchyLoss(nn.Module):
    """
    Mixture of Cauchy on the logit scale.

    Model output: (B, 3*n_components, T)
      1) mixture logits: y[:, 0:n_components, :]
      2) mu:             y[:, n_components:2*n_components, :]
      3) log_gamma:      y[:, 2*n_components:3*n_components, :]

    x: (B,T).
    """
    def __init__(self, n_components=3, eps=1e-6,
                 clamp_mu=(-10,10), clamp_log_gamma=(-10,10),
                 reduction='mean'):
        super().__init__()
        self.n_components = n_components
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_gamma = clamp_log_gamma
        self.reduction = reduction
        self.LOG_PI = math.log(math.pi)

    def forward(self, y, x):
        B, C, T = y.shape
        expected_c = 3*self.n_components
        assert C == expected_c, f"Expected {expected_c}, got {C}"

        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x)
        logit_x = logit_x.unsqueeze(1)  # => (B,1,T)

        n = self.n_components
        mix_logits = y[:, 0:n, :]               # (B,n,T)
        mu         = y[:, n:2*n, :]
        log_gamma  = y[:, 2*n:3*n, :]

        mu        = torch.clamp(mu, self.clamp_mu[0], self.clamp_mu[1])
        log_gamma = torch.clamp(log_gamma, self.clamp_log_gamma[0], self.clamp_log_gamma[1])
        gamma = torch.exp(log_gamma) + self.eps

        weights = F.softmax(mix_logits, dim=1)

        # log cauchy
        z = (logit_x - mu)/gamma
        log_cauchy = -self.LOG_PI - torch.log(gamma) - torch.log1p(z**2)

        jacobian = -torch.log(x*(1-x) + self.eps).unsqueeze(1)
        log_comp = log_cauchy + jacobian

        log_mixture = torch.log(weights + self.eps) + log_comp
        log_prob = torch.logsumexp(log_mixture, dim=1)

        nll = -log_prob
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class MixtureLogitStudentTFixedNuLoss(nn.Module):
    """
    Mixture of Student's t distributions (fixed nu) on logit scale.

    Model output: (B, 3*n_components, T)
      1) mixture logits: y[:, 0:n_components, :]
      2) mu:             y[:, n_components:2*n_components, :]
      3) log_sigma:      y[:, 2*n_components:3*n_components, :]

    x: (B,T).
    nu is a fixed scalar > 0.
    """
    def __init__(self, n_components=3, nu=3.0, eps=1e-6,
                 clamp_mu=(-10,10), clamp_log_sigma=(-10,10),
                 reduction='mean'):
        super().__init__()
        self.n_components = n_components
        self.nu = float(nu)
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_sigma = clamp_log_sigma
        self.reduction = reduction

    def forward(self, y, x):
        from torch.special import gammaln

        B, C, T = y.shape
        expected_c = 3*self.n_components
        assert C == expected_c, f"Expected {expected_c}, got {C}"

        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x).unsqueeze(1)  # (B,1,T)

        n = self.n_components
        mix_logits = y[:, 0:n, :]
        mu         = y[:, n:2*n, :]
        log_sigma  = y[:, 2*n:3*n, :]

        mu = torch.clamp(mu, self.clamp_mu[0], self.clamp_mu[1])
        log_sigma = torch.clamp(log_sigma, self.clamp_log_sigma[0], self.clamp_log_sigma[1])
        sigma = torch.exp(log_sigma) + self.eps

        weights = F.softmax(mix_logits, dim=1)

        nu = torch.tensor(self.nu, device=y.device)
        half_nu = 0.5*nu
        # Precompute constants:
        log_numer = gammaln(half_nu + 0.5) # scalar
        # We'll incorporate sigma in log_denom below

        z = (logit_x - mu)/sigma
        # log t_k = log_numer - [gammaln(half_nu) + 0.5 ln(nu*pi) + ln(sigma)]
        #           - 0.5*(nu+1) ln(1 + z^2/nu)
        log_denom_const = gammaln(half_nu) + 0.5*torch.log(nu*math.pi)

        # shape => (B,n,T)
        log_denom = log_denom_const + torch.log(sigma)
        log_t = (log_numer - log_denom
                 - 0.5*(nu+1.0)*torch.log1p(z**2 / nu))

        jacobian = -torch.log(x*(1-x) + self.eps).unsqueeze(1)
        log_comp = log_t + jacobian

        log_mixture = torch.log(weights + self.eps) + log_comp
        log_prob = torch.logsumexp(log_mixture, dim=1)

        nll = -log_prob
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class MixtureLogitStudentTAdaptiveNuLoss(nn.Module):
    """
    Mixture of Student's t (adaptive nu) on logit scale.

    Model output: (B, 4*n_components, T)
      1) mixture logits: y[:, 0:n, :]
      2) mu:             y[:, n:2n, :]
      3) log_sigma:      y[:, 2n:3n, :]
      4) log_nu:         y[:, 3n:4n, :]

    x: (B,T).
    """
    def __init__(self, n_components=3, eps=1e-6,
                 clamp_mu=(-10,10), clamp_log_sigma=(-10,10), clamp_log_nu=(-3, 10),
                 reduction='mean'):
        super().__init__()
        self.n_components = n_components
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_sigma = clamp_log_sigma
        self.clamp_log_nu = clamp_log_nu
        self.reduction = reduction

    def forward(self, y, x):
        from torch.special import gammaln

        B, C, T = y.shape
        n = self.n_components
        expected_c = n + 3*n  # 4n
        assert C == expected_c, f"Expected {expected_c} channels, got {C}"

        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x).unsqueeze(1)

        mix_logits = y[:, 0:n, :]
        mu         = y[:, n:2*n, :]
        log_sigma  = y[:, 2*n:3*n, :]
        log_nu     = y[:, 3*n:4*n, :]

        mu        = torch.clamp(mu,  self.clamp_mu[0],  self.clamp_mu[1])
        log_sigma = torch.clamp(log_sigma, self.clamp_log_sigma[0], self.clamp_log_sigma[1])
        log_nu    = torch.clamp(log_nu,    self.clamp_log_nu[0],    self.clamp_log_nu[1])

        sigma = torch.exp(log_sigma) + self.eps
        nu    = torch.exp(log_nu)    + self.eps

        weights = F.softmax(mix_logits, dim=1)

        z = (logit_x - mu) / sigma

        # log Student-t
        #   log_numer = gammaln((nu+1)/2)
        #   log_denom = gammaln(nu/2) + 0.5 ln(nu*pi) + ln(sigma)
        log_numer = gammaln(0.5*(nu + 1.0))
        log_denom = gammaln(0.5*nu) + 0.5*torch.log(nu*math.pi) + torch.log(sigma)

        log_t = log_numer - log_denom - 0.5*(nu+1.0)*torch.log1p(z**2/nu)

        jacobian = -torch.log(x*(1-x) + self.eps).unsqueeze(1)
        log_comp = log_t + jacobian

        log_mixture = torch.log(weights + self.eps) + log_comp
        log_prob = torch.logsumexp(log_mixture, dim=1)  # (B,T)

        nll = -log_prob
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureLogitLaplaceLoss(nn.Module):
    """
    Mixture of Laplace distributions on the logit scale.

    Model output: (B, 3*n_components, T)

      1) mixture logits: y[:, 0:n_components, :]
      2) mu:             y[:, n_components:2*n_components, :]
      3) log_b:          y[:, 2*n_components:3*n_components, :]

    x: (B,T) or (B,1,T) in [0,1].

    The Laplace PDF on logit_x is:
      p(logit_x) = 1/(2*b) * exp(-| (logit_x - mu)/b |),
    plus the Jacobian = -log[x(1-x)].

    We'll combine the components via log-sum-exp with mixture weights
    from softmax(mix_logits).
    """
    def __init__(
        self,
        n_components=3,
        eps=1e-6,
        clamp_mu=(-10,10),
        clamp_log_b=(-10,10),
        reduction='mean'
    ):
        super().__init__()
        self.n_components = n_components
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_b = clamp_log_b
        self.reduction = reduction

    def forward(self, y, x):
        """
        y: (B, 3*n_components, T)
        x: (B,T) or (B,1,T) in [0,1].
        returns negative log-likelihood (scalar if reduction='mean').
        """
        B, C, T = y.shape
        expected_c = 3 * self.n_components
        assert C == expected_c, f"Expected {expected_c} channels, got {C}"

        # 1) Possibly squeeze x if shape is (B,1,T)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # => (B,T)

        # 2) Clamp x to avoid infinite logit
        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x)  # (B,T)
        logit_x = logit_x.unsqueeze(1)  # => (B,1,T) for broadcasting

        n = self.n_components
        # 3) Decompose y into mixture logits, mu, log_b
        mix_logits = y[:, 0:n, :]               # (B,n,T)
        mu        = y[:, n:2*n, :]
        log_b     = y[:, 2*n:3*n, :]

        # 4) clamp mu, log_b
        mu    = torch.clamp(mu, self.clamp_mu[0], self.clamp_mu[1])
        log_b = torch.clamp(log_b, self.clamp_log_b[0], self.clamp_log_b[1])
        b     = torch.exp(log_b) + self.eps  # (B,n,T)

        # 5) Mixture weights
        weights = F.softmax(mix_logits, dim=1)  # (B,n,T)

        # 6) Laplace log-density
        #    let z = (logit_x - mu) / b
        #    log Laplace = -log(2*b) - |z|
        z = (logit_x - mu) / b
        log_laplace = -torch.log(2.0 * b) - torch.abs(z)

        # 7) Jacobian = -log[x(1-x)] => shape (B,T) => (B,1,T)
        jacobian = -torch.log(x*(1.0 - x) + self.eps).unsqueeze(1)

        log_comp = log_laplace + jacobian  # (B,n,T)

        # 8) Combine via log-sum-exp
        log_mixture = torch.log(weights + self.eps) + log_comp
        log_prob = torch.logsumexp(log_mixture, dim=1)  # => (B,T)

        # 9) Negative log-likelihood
        nll = -log_prob

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll  # shape (B,T)



class LogitGaussianLoss(nn.Module):
    """
    Single-component Gaussian on the logit scale.

    Model output: (B, 2, T) => [mu, log_sigma].
    x in [0,1]: (B, T) or (B,1,T).

    PDF(logit_x) = Normal( (logit_x - mu)/sigma ), 
    plus Jacobian: -log[x(1-x)].
    """
    def __init__(self, eps=1e-6, clamp_mu=(-10,10), clamp_log_sigma=(-10,10),
                 reduction='mean'):
        super().__init__()
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_sigma = clamp_log_sigma
        self.reduction = reduction
        self.LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

    def forward(self, y, x):
        # y: (B,2,T) => [mu, log_sigma]
        # x: (B,T) or (B,1,T) in [0,1]
        B, C, T = y.shape
        assert C == 2, f"Expected 2 channels, got {C}"

        # If x is (B,1,T), optionally squeeze:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # => (B,T)

        # 1) clamp x & logit
        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x)  # (B,T)

        mu        = y[:, 0, :]
        log_sigma = y[:, 1, :]

        # 2) clamp
        mu = torch.clamp(mu, self.clamp_mu[0], self.clamp_mu[1])
        log_sigma = torch.clamp(log_sigma, self.clamp_log_sigma[0], self.clamp_log_sigma[1])

        sigma = torch.exp(log_sigma) + self.eps

        # 3) compute z
        z = (logit_x - mu) / sigma

        # 4) log Gaussian
        #    -0.5*z^2 - log(sigma) - log(sqrt(2*pi))
        log_gaussian = -0.5 * (z**2) - (torch.log(sigma) + self.LOG_SQRT_2PI)

        # 5) jacobian
        jacobian = -torch.log(x*(1 - x) + self.eps)

        log_prob = log_gaussian + jacobian
        nll = -log_prob

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class LogitCauchyLoss(nn.Module):
    """
    Single-component Cauchy on the logit scale.
    Model output: (B,2,T) => [mu, log_gamma].
    x in [0,1]: (B,T) or (B,1,T).

    PDF(logit_x) = 1 / [pi * gamma * (1 + ((logit_x - mu)/gamma)^2)]
    plus Jacobian: -log[x(1-x)].
    """
    def __init__(self, eps=1e-6, clamp_mu=(-10,10), clamp_log_gamma=(-10,10),
                 reduction='mean'):
        super().__init__()
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_gamma = clamp_log_gamma
        self.reduction = reduction
        self.LOG_PI = math.log(math.pi)

    def forward(self, y, x):
        B, C, T = y.shape
        assert C == 2, f"Expected 2 channels, got {C}"

        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # => (B,T)

        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x)

        mu        = y[:, 0, :]
        log_gamma = y[:, 1, :]

        mu        = torch.clamp(mu, *self.clamp_mu)
        log_gamma = torch.clamp(log_gamma, *self.clamp_log_gamma)

        gamma = torch.exp(log_gamma) + self.eps

        z = (logit_x - mu) / gamma
        # log Cauchy = - log(pi) - log(gamma) - log(1+z^2)
        log_cauchy = -self.LOG_PI - torch.log(gamma) - torch.log1p(z**2)

        jacobian = -torch.log(x*(1 - x) + self.eps)

        log_prob = log_cauchy + jacobian
        nll = -log_prob

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class LogitStudentTFixedNuLoss(nn.Module):
    """
    Single-component Student's t on the logit scale, with a fixed nu > 0.

    Model output: (B,2,T) => [mu, log_sigma].
    x in [0,1].
    nu is a scalar float set in constructor.
    """
    def __init__(self, nu=3.0, eps=1e-6,
                 clamp_mu=(-10,10), clamp_log_sigma=(-10,10),
                 reduction='mean'):
        super().__init__()
        self.nu = float(nu)  # fixed degrees of freedom
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_sigma = clamp_log_sigma
        self.reduction = reduction

        # Precompute constants that might be used
        # but we need device info if we do gammaln, so be careful in forward.

    def forward(self, y, x):
        B, C, T = y.shape
        assert C == 2, f"Expected 2 channels (mu, log_sigma), got {C}"

        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x)

        mu        = torch.clamp(y[:, 0, :], self.clamp_mu[0], self.clamp_mu[1])
        log_sigma = torch.clamp(y[:, 1, :], self.clamp_log_sigma[0], self.clamp_log_sigma[1])
        sigma     = torch.exp(log_sigma) + self.eps

        nu = torch.tensor(self.nu, device=y.device)
        z = (logit_x - mu) / sigma

        # Student’s t formula:
        # log p_t(z) = ln Gamma((nu+1)/2) - ln[ sqrt(nu*pi)*Gamma(nu/2)*sigma ] 
        #             - (nu+1)/2 * ln[1 + (z^2)/nu ]
        from torch.special import gammaln

        log_numer = gammaln(0.5*(nu + 1.0))
        log_denom = (
            gammaln(0.5*nu) +
            0.5*torch.log(nu * torch.tensor(math.pi, device=y.device)) +
            torch.log(sigma)
        )
        # broadcast shape: (B,T)
        log_t = log_numer - log_denom - 0.5*(nu + 1.0)*torch.log1p(z**2 / nu)

        jacobian = -torch.log(x*(1 - x) + self.eps)
        log_prob = log_t + jacobian
        nll = -log_prob

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class LogitStudentTAdaptiveNuLoss(nn.Module):
    """
    Single-component Student's t on the logit scale, with learned degrees of freedom.
    
    Model output: (B,3,T) => [mu, log_sigma, log_nu].
    x in [0,1].
    """
    def __init__(self, eps=1e-6,
                 clamp_mu=(-10,10), clamp_log_sigma=(-10,10), clamp_log_nu=(-3,10),
                 reduction='mean'):
        super().__init__()
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_sigma = clamp_log_sigma
        self.clamp_log_nu = clamp_log_nu
        self.reduction = reduction

    def forward(self, y, x):
        B, C, T = y.shape
        assert C == 3, f"Expected 3 channels (mu, log_sigma, log_nu), got {C}"

        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x)

        mu        = torch.clamp(y[:, 0, :], self.clamp_mu[0], self.clamp_mu[1])
        log_sigma = torch.clamp(y[:, 1, :], self.clamp_log_sigma[0], self.clamp_log_sigma[1])
        log_nu    = torch.clamp(y[:, 2, :], self.clamp_log_nu[0], self.clamp_log_nu[1])

        sigma = torch.exp(log_sigma) + self.eps
        nu    = torch.exp(log_nu) + self.eps

        z = (logit_x - mu) / sigma

        from torch.special import gammaln

        # log_numer = ln Gamma((nu+1)/2)
        # log_denom = ln Gamma(nu/2) + 0.5 ln(nu*pi) + ln(sigma)
        log_numer = gammaln(0.5*(nu + 1.0))
        log_denom = gammaln(0.5*nu) + 0.5*torch.log(nu*math.pi) + torch.log(sigma)

        log_t = log_numer - log_denom - 0.5*(nu + 1.0)*torch.log1p(z**2 / nu)

        jacobian = -torch.log(x*(1 - x) + self.eps)
        log_prob = log_t + jacobian
        nll = -log_prob

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll



class LogitLaplaceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y, x):
        # Clamp x to avoid extreme values near 0 and 1
        x = torch.clamp(x, min=1e-5, max=1-1e-5)
        
        # Extract mu and b, and clamp the input to exp to prevent it from growing too large
        mu = y[:, 0, :].unsqueeze(1)
        b = torch.clamp(torch.exp(torch.clamp(y[:, 1, :], max=20)), min=1e-4).unsqueeze(1)
        
        # Compute density
        logit_x = torch.logit(x)  # Clamp prevents instability
        density = (1 / (2 * b * x * (1 - x))) * torch.exp(-torch.abs(logit_x - mu) / b)
        
        # Clamp density to ensure it's positive and greater than a small value
        density = torch.clamp(density, min=1e-10)
        
        # Compute loss
        loss = -torch.mean(torch.log(density + 1e-5))

        return loss


class LrScheduler:
    # TODO: maybe need to inherit from Pytorch.nn.Module?
    def __init__(self, optimizer, warmup_scheduler, main_scheduler, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0

    def get_lr(self):
        if self.step_count < self.warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        else:
            return self.main_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.step_count < self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()
        self.step_count += 1

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.main_scheduler.load_state_dict(state_dict['main_scheduler'])

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'main_scheduler': self.main_scheduler.state_dict(),
        }
