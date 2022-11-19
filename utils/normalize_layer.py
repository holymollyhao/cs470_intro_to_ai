import torch

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.register_buffer(
            'mu', torch.tensor(means).view(-1, 1, 1))
        self.register_buffer(
            'sigma', torch.tensor(sds).view(-1, 1, 1))

    def forward(self, input: torch.tensor):
        return (input - self.mu) / self.sigma
