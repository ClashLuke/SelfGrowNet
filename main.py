import torch

torch.manual_seed(1)
from model import Trainer

if __name__ == '__main__':
    FEATURES = 4
    WIDTH = 4
    DEVICE = torch.device('cpu')
    dataset_inputs = torch.randn((2 ** 20, 16, FEATURES)) * WIDTH
    dataset_outputs = dataset_inputs.cos().mean(-1) * dataset_inputs.square().mean(-1).sqrt()
    model = Trainer(FEATURES, (dataset_inputs, dataset_outputs), DEVICE, max_loss=(WIDTH*FEATURES)**2)
    model.fit(100)
