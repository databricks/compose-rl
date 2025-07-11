# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch
from composer.models import ComposerClassifier


class SimpleMLP(torch.nn.Module):
    def __init__(self, num_features: int, device: str = 'cpu'):
        super().__init__()
        fc1 = torch.nn.Linear(
            num_features,
            num_features,
            device=device,
            bias=False,
        )
        fc2 = torch.nn.Linear(
            num_features,
            num_features,
            device=device,
            bias=False,
        )

        self.net = torch.nn.Sequential(fc1, torch.nn.ReLU(), fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleWeightTiedModel(ComposerClassifier):
    """Small classification model with tied weights.

    Args:
        num_features (int): number of input features (default: 1)
        device (str): the device to initialize the model (default: 'cpu')
    """

    def __init__(self, num_features: int = 1, device: str = 'cpu') -> None:
        self.num_features = num_features

        mlp = SimpleMLP(num_features, device)

        net = torch.nn.Sequential(
            mlp,
            torch.nn.Softmax(dim=-1),
        )

        super().__init__(module=net, num_classes=num_features)

        self.module.param_init_fn = self.param_init_fn  # pyright: ignore[reportGeneralTypeIssues]

        # Adding mlp.fc1.weight = mlp.fc2.weight without assignment to self.fc1 and self.fc2
        # since we don't want to create duplicate references to the same module
        # since that will break mixed init.
        mlp.net[0].weight = mlp.net[-1].weight

    def add_fsdp_wrap_attribute_to_children(self):
        for child in self.children():
            child._fsdp_wrap = False  # type: ignore
        for child in self.module.children():
            child._fsdp_wrap = True  # type: ignore

    def param_init_fn(self, module: torch.nn.Module):
        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=0.1)

        if isinstance(module, torch.nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                torch.nn.init.zeros_(module.bias)


class PartialWeightTiedModel(ComposerClassifier):
    """Small classification model with partially tied weights.

    Args:
        num_features (int): number of input features (default: 1)
        device (str): the device to initialize the model (default: 'cpu')
    """

    def __init__(self, num_features: int = 1, device: str = 'cpu') -> None:
        mlp = SimpleMLP(num_features, device)

        # a third fc layer that is not tied to the above mlp
        fc3 = torch.nn.Linear(
            num_features,
            num_features,
            device=device,
            bias=False,
        )

        net = torch.nn.Sequential(
            mlp,
            fc3,
            torch.nn.Softmax(dim=-1),
        )

        # fc1 would be a child module of the Sequential module now but only the mlp should be FSDP wrapped
        # TODO support this or add negative test for this
        # net.fc1 = mlp.fc1

        super().__init__(module=net, num_classes=num_features)
        self.module.param_init_fn = self.param_init_fn  # pyright: ignore[reportGeneralTypeIssues]

        # Adding mlp.fc1.weight = mlp.fc2.weight without assignment to self.fc1 and self.fc2
        # since we don't want to create duplicate references to the same module since that
        # will break mixed init.
        mlp.net[0].weight = mlp.net[-1].weight

    def add_fsdp_wrap_attribute_to_children(self):
        for child in self.children():
            child._fsdp_wrap = False  # type: ignore
        for child in self.module.children():
            child._fsdp_wrap = True  # type: ignore

    def param_init_fn(self, module: torch.nn.Module):
        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=0.1)

        if isinstance(module, torch.nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                torch.nn.init.zeros_(module.bias)
