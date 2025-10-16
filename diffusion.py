import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def continuous_extract(v, t, x_shape):
    """
    Apply function at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = v(t)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


s = 0.008
def cosine_schedule(t):
    """
    Function implementing cosine diffusion schedule.
    Parameters
    ----------
    t - tensor of timesteps from [0, 1]
    Returns
    -------
    Tensor of \bar{\alpha}_t values
    """
    return (
        torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2 /
        math.cos(s / (1 + s) * math.pi / 2) ** 2
    )


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_bar = lambda t: cosine_schedule(t).sqrt()
        self.sqrt_one_minus_alphas_bar = lambda t: (1 - cosine_schedule(t)).sqrt()

    def forward(self, x_0):
        t = torch.rand(size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            continuous_extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            continuous_extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'ts', torch.linspace(0, 1, T))
        alphas_bar = cosine_schedule(self.ts)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        alphas = alphas_bar / alphas_bar_prev
        self.register_buffer(
            'betas', 1. - alphas)

        # calculations for forward process q(x_t | x_0)
        self.register_buffer(
            'sqrt_alphas_bar_prev', torch.sqrt(alphas_bar_prev))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar_prev', torch.sqrt(1. - alphas_bar_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def predict_eps_from_xstart(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (
            extract(
                self.sqrt_recip_alphas_bar / self.sqrt_recipm1_alphas_bar, t, x_t.shape) * x_t -
            extract(1 / self.sqrt_recipm1_alphas_bar, t, x_t.shape) * x_0
        )

    def predict_eps_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (
            extract(1 / self.betas.sqrt(), t, x_t.shape) * x_t -
            extract(((1 - self.betas) / self.betas).sqrt(), t, x_t.shape) * xprev
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t / self.T)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
            eps = self.predict_eps_from_xprev(x_t, t, xprev=x_prev)
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t / self.T)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            eps = self.predict_eps_from_xstart(x_t, t, x_0)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t / self.T)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        # x_0 = torch.clip(x_0, -1., 1.)

        return x_0, eps, model_mean, model_log_var

    def forward(self, x_T, solver='ddpm'):
        assert solver in ['ddpm', 'ddim', 'jumping']
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            x_0, eps, mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            if solver == 'ddpm':
                x_t = mean + torch.exp(0.5 * log_var) * noise
            elif solver in ['ddim', 'jumping']:
                if solver == 'ddim' and time_step > 0:
                    noise = eps
                x_t = (
                    extract(self.sqrt_alphas_bar_prev, t, x_t.shape) * x_0 +
                    extract(self.sqrt_one_minus_alphas_bar_prev, t, x_t.shape) * noise
                )

        x_0 = x_t
        return torch.clip(x_0, -1, 1)
