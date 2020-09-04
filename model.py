import copy
import math
import os
import random
import time
import typing

import torch

torch.set_num_threads(os.cpu_count())
random.seed(1)


class GrowableLinear(torch.nn.Module):
    def __init__(self, inf: int, outf: int):
        super(GrowableLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(inf, outf))
        self.old_x = 0
        self.old_y = 0
        torch.nn.init.orthogonal_(self.weight)

    def forward(self, inp: torch.Tensor):
        return inp.mm(self.weight)

    def add_feature(self, dim: int):
        size = list(self.weight.size())
        original_size = size.copy()
        size[dim] += 1
        weight = torch.empty(size)
        std = self.weight.std()
        if torch.isnan(std):
            std = 1e-3
        else:
            std = std.item()
        torch.nn.init.normal_(weight[original_size[0]:, original_size[1]:],
                              std=std,
                              mean=self.weight.mean().item())
        weight[0:original_size[0], 0:original_size[1]] = self.weight
        self.old_x = original_size[0]
        self.old_y = original_size[1]
        self.weight = torch.nn.Parameter(weight)

    def no_new(self):
        self.old_x = None
        self.old_y = None


class Mish(torch.nn.Module):
    def forward(self, inp):
        return inp.mul(torch.nn.functional.softplus(inp).tanh())


def no_new(model):
    for layer in model.layers:
        if isinstance(layer, GrowableLinear):
            layer.no_new()


class Model(torch.nn.Module):
    def __init__(self, in_features: int):
        super(Model, self).__init__()
        self.layers = torch.nn.ModuleList([GrowableLinear(in_features, 1)])

    def forward(self, inp: torch.Tensor):
        for mod in self.layers:
            inp = mod(inp)
        return inp.mean(1)

    def add_layer(self, clone=True):
        model = self.clone() if clone else self
        no_new(model)
        model.layers.append(Mish())
        model.layers.append(GrowableLinear(model.layers[-2].weight.size(1), 1))
        return model

    def add_feature(self, layer: int, clone=True):
        model = self.clone() if clone else self
        no_new(model)
        layer = - layer - 1
        if layer < -1:
            model.layers[layer].add_feature(1)
            model.layers[layer + 2].add_feature(0)
        else:
            return False
        return model

    def __int__(self):
        return len(self.layers)

    def __ge__(self, other):
        return hasattr(other, "layers") and len(self.layers) >= len(other.layers)

    def __le__(self, other):
        return not hasattr(other, "layers") or len(self.layers) <= len(other.layers)

    def __lt__(self, other):
        return not hasattr(other, "layers") or len(self.layers) < len(other.layers)

    def __gt__(self, other):
        return hasattr(other, "layers") and len(self.layers) > len(other.layers)

    def __len__(self):
        return int(self)

    def __bool__(self):
        return True

    def clone(self):
        return copy.deepcopy(self)

    def config(self):
        out = [l.weight.size(0) for l in self.layers[::2]]
        out.append(self.layers[-1].weight.size(1))
        out = map(str, out)
        return '-'.join(out)


class Trainer:
    def __init__(self, in_features: int, dataset: typing.Tuple[torch.Tensor, torch.Tensor], device: torch.device,
                 printervall=None, max_loss=100, bagging=0.5, sub_epochs=1):
        self.model = Model(in_features).to(device)
        self.dataset_inputs = dataset[0]
        self.dataset_outputs = dataset[1]
        self.device = device
        self.printervall = printervall
        self.max_loss = max_loss
        self.bagging = bagging
        self.sub_epochs = sub_epochs

    def _train_epoch(self, model, loss_list, loss_idx):
        if not model:
            loss_list[loss_idx] = math.inf
            return
        total_loss = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        # Heavy weight decay is regularization to not overfit on the data
        item_cnt = len(self.dataset_inputs)
        item_str = str(item_cnt)
        item_len = len(item_str)
        start_time = time.time()
        for _ in range(self.sub_epochs):
            # for i in range(1, 1 + item_cnt):
            #     dat = self.dataset_inputs[i]
            #     tgt = self.dataset_outputs[i]
            for i, (dat, tgt) in enumerate(zip(self.dataset_inputs, self.dataset_outputs)):
                if random.random() < self.bagging:
                    continue
                dat = dat.to(self.device)
                out: torch.Tensor = model(dat)
                lss = out.sub(tgt.to(self.device)).square().mean()
                lss.backward()
#                for layer in model.layers:
#                    if isinstance(layer, GrowableLinear):
#                        if layer.old_x is None or layer.old_y is None:
#                            layer.weight.grad[:] = 0
#                            continue
#                        layer.weight.grad[:layer.old_x, :layer.old_y] = 0
                total_loss += lss.item()
                optimizer.step()
                optimizer.zero_grad()
                if self.printervall is not None and i % self.printervall == 0:
                    print(f"[{i:{item_len}d}/{item_str}] Loss: {lss} - Rate: {i / (time.time() - start_time)} Batch/s")
        total_loss /= item_cnt * self.sub_epochs * (1 - self.bagging)
        if total_loss >= self.max_loss:
            total_loss = math.inf
        loss_list[loss_idx] = total_loss

    def fit(self, epochs=1):
        epoch_str = str(epochs)
        epoch_len = len(epoch_str)
        loss_pad = len(str(self.max_loss)) + 6
        manager = torch.multiprocessing.Manager()
        for i in range(epochs):
            start_time = time.time()
            models = [self.model.clone(), self.model.add_layer()] + [self.model.add_feature(layer) for layer in range(0, len(self.model), 2)]
            loss_list = manager.list()
            for _ in range(len(models)):
                loss_list.append(0)
            jobs = [torch.multiprocessing.Process(target=self._train_epoch, args=(mdl,loss_list,i)) for i,mdl in enumerate(models)]
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
            loss, self.model = sorted(zip(loss_list, models))[0]
            print(f'[{i + 1:{epoch_len}d}/{epoch_str}] '
                  f'Best: {loss:{loss_pad}.5f} - '
                  f'Took: {time.time() - start_time:8.1f}s | '
                  f'Config: {self.model.config()}')
