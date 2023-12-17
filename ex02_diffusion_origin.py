import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.02)


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule.
    #  Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case.
    #  You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps).to(self.device)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here
        #  so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help

        # TODO: define alphas
        self.alphas = torch.tensor(1. - self.betas).to(self.device)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(self.device)  # alpha_t
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0).to(self.device)
        self.alphas_reciprocal = torch.sqrt(1. / self.alphas).to(self.device)  # 1 / sqrt(alpha_t)

        # TODO: calculations for diffusion q(x_t | x_{t-1}) and others
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_bar).to(self.device)  # sqrt(alpha_bar_t)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_bar).to(self.device)  # sqrt(1 - alpha_bar_t)

        # TODO: calculations for posterior q(x_{t-1} | x_t, x_0)
        # self.posterior_variance = self.betas * (1 - self.alphas_bar_prev) / (1 - self.alphas_bar)
        self.betas_sqrt = torch.sqrt(self.betas).to(self.device)  # sqrt(beta_t)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, class_token, train_mode, weight):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t.
        #  Note that x and t both have a batch dimension
        beta_t = extract(self.betas.cpu(), t, x.shape).to(self.device)
        alpha_reciprocal_t = extract(self.alphas_reciprocal.cpu(), t, x.shape).to(self.device)
        one_minus_alpha_bar_sqrt_t = extract(self.one_minus_alphas_bar_sqrt.cpu(), t, x.shape).to(self.device)
        beta_sqrt_t = extract(self.betas_sqrt.cpu(), t, x.shape).to(self.device)

        # TODO: Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        predicted_noise_uncond = model(x, t, class_label=None, train_mode=False)
        predicted_noise_cond = model(x, t, class_label=class_token, train_mode=False)
        predicted_noise = (1+weight) * predicted_noise_cond - weight * predicted_noise_uncond
        z = torch.randn_like(x) if t_index > 0 else 0

        # TODO (2.2): The method should return the image at timestep t-1.
        x_t_minus_1 = (alpha_reciprocal_t * (x - beta_t * predicted_noise / one_minus_alpha_bar_sqrt_t)
                       + beta_sqrt_t * z)
        return x_t_minus_1

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, class_token=None, class_free_guidance=False, weight=0):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image,
        #  iteratively ''reducing'' the noise in the generated image.

        shape = (batch_size, channels, image_size, image_size)
        # Initialize the image with random noise
        x_zero = torch.randn(shape, device=self.device)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling Loop Time Step", total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_zero = self.p_sample(model, x_zero, t, i, class_token, train_mode=False, weight=weight)
            # imgs.append(x_zero.cpu().numpy())
            imgs.append(x_zero.cpu())
        # TODO (2.2): Return the generated images
        return imgs

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor;
        #  if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero).to(self.device)
        alpha_bar_sqrt_t = extract(self.alphas_bar_sqrt.cpu(), t, x_zero.shape).to(self.device)
        one_minus_alpha_bar_sqrt_t = extract(self.one_minus_alphas_bar_sqrt.cpu(), t, x_zero.shape).to(self.device)
        x_t = alpha_bar_sqrt_t * x_zero + one_minus_alpha_bar_sqrt_t * noise  # (4)
        return x_t

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1", labels=None, train_mode=False):
        # TODO (2.2): compute the input to the network using the forward diffusion process
        #  and predict the noise using the model; if noise is None, you will need to create a new noise vector,
        #  otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)
        noise_image = self.q_sample(x_zero, t, noise)
        noise_pred = denoise_model(noise_image, t, labels, train_mode)
        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(noise, noise_pred)
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(noise, noise_pred)
        else:
            raise NotImplementedError()

        return loss


if __name__ == '__main__':
    timesteps = 100
    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    image_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)


