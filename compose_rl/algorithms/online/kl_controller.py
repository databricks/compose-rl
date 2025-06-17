# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import torch

from compose_rl.interfaces.base_kl_controller import BaseKLController


class AdaptiveKLController(BaseKLController):
    """Adaptive KL controller described in the paper:

    https://arxiv.org/abs/1909.08593 (Sec 2.2)
    KL = clip((kl_curr / target) - 1, -n, n)

    Args:
        init_kl_coef (float): The initial KL coefficient.
        target (float): The target KL coefficient.
        horizon (int): The horizon of the KL controller.
        device (str): The device to use for the KL controller. Default is `cpu`.
    """

    def __init__(
        self,
        init_kl_coef: float,
        target: float,
        horizon: int,
        device: str = 'cpu',
    ):
        super().__init__(device=device)
        self._value = init_kl_coef
        self._target = target
        self._horizon = horizon

    def update(self, current: torch.Tensor, n_steps: int):
        target = self._target
        proportional_error = torch.clamp(
            current / target - 1,
            min=-0.2,
            max=0.2,
        )
        # We need to force cast mult to be a float otherwise it will break
        # PyTorch 2.4 checkpointing.
        mult = float(1 + proportional_error * n_steps / self._horizon)
        self._value *= mult

    @property
    def value(self):
        return self._value

    def state_dict(self):
        return {'value': self.value}

    def load_state_dict(self, state_dict: dict[str, float]):
        self._value = state_dict['value']


class FixedKLController(BaseKLController):
    """Fixed KL controller that always returns same value.

    Args:
        init_kl_coef (float): The initial KL coefficient.
        device (str): The device to use for the KL controller. Default is `cpu`.
    """

    def __init__(self, init_kl_coef: float, device: str = 'cpu'):
        super().__init__(device=device)
        self._value = init_kl_coef

    def update(self, current: torch.Tensor, n_steps: int):
        return

    @property
    def value(self):
        return self._value


class KLPIDController(BaseKLController):
    """Controller based on PID Controls adapted from safe RL literature.

    This is taken from:
    https://arxiv.org/abs/2007.03964 (Equations 18-22)

    Original assumes an update to policy using a constrained optimization
    technique via Lagrangian multiplier. This is approximated via KL penalties
    in rewards (for now).
    KL = KL + -kl_lr * ((kl_curr - target) * trajectory_len)

    Args:
        init_kl_coef (float): The initial KL coefficient.
        target (float): The target KL coefficient.
        horizon (int): The horizon of the KL controller.
        kl_lr (float): The learning rate of the KL controller.
        device (str): The device to use for the KL controller. Default is `cpu`.
    """

    def __init__(
        self,
        init_kl_coef: float,
        target: float,
        horizon: int,
        kl_lr: float,
        device: str = 'cpu',
    ):
        super().__init__(device=device)
        self._value: torch.Tensor = torch.tensor([init_kl_coef],
                                                 requires_grad=True,
                                                 device=self.device)
        self._target = target
        self._horizon = horizon
        self._optim = torch.optim.Adam([self._value], lr=kl_lr)
        self.kl_lr = kl_lr

    def update(self, current: torch.Tensor, n_steps: int):
        self.device = current.device
        self._value.to(self.device)
        diff_to_target = current - self._target  # No clip
        proportional_error = diff_to_target * n_steps / self._horizon
        self._value.sum().backward()
        self._value.grad = torch.tensor([-1 * proportional_error]).float()
        self._optim.step()

    @property
    def value(self):
        return self._value.detach().item()

    def state_dict(self):
        return {'value': self.value}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self._value = state_dict['value']
        # It is fine to reset the optimizer state on reload
        # as grad is based on single step error
        self._optim = torch.optim.Adam([self._value], lr=self.kl_lr)


class BallKLController(BaseKLController):
    """Experimental KL Controller extending PID controller above.

    This controller gives a softer constraint than adaptive.

    Same equations as PID controller  https://arxiv.org/abs/2007.03964
    (Equations 18-22) but clamped. Intuition is that adaptive KL tries to
    constrain policy to being on `surface' of a Ball (loss surface) but this
    allows for policy to be anywhere inside `volume' of said Ball
    KL = clamp(KL + -kl_lr * ((kl_curr - target) * trajectory_len))

    Args:
        init_kl_coef (float): The initial KL coefficient.
        target (float): The target KL coefficient.
        horizon (int): The horizon of the KL controller.
        kl_lr (float): The learning rate of the KL controller.
        device (str): The device to use for the KL controller. Default is `cpu`.
    """

    def __init__(
        self,
        init_kl_coef: float,
        target: float,
        horizon: int,
        kl_lr: float,
        device: str = 'cpu',
    ):
        super().__init__(device=device)
        self._value: torch.Tensor = torch.tensor([init_kl_coef],
                                                 requires_grad=True,
                                                 device=self.device)
        self._target = target
        self._horizon = horizon
        self._optim = torch.optim.Adam([self._value], lr=kl_lr)
        self.kl_lr = kl_lr

    def update(self, current: torch.Tensor, n_steps: int):
        self.device = current.device
        self._value.to(self.device)
        diff_to_target = current - self._target
        proportional_error = diff_to_target * n_steps / self._horizon
        self._value.sum().backward()
        self._value.grad = torch.tensor([-1 * proportional_error]).float()
        self._optim.step()
        with torch.no_grad():
            self._value.clamp_(min=0)

    @property
    def value(self):
        return self._value.detach().item()

    def state_dict(self):
        return {'value': self.value}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self._value = state_dict['value']
        # It is fine to reset the optimizer state on reload
        # as grad is based on single step error
        self._optim = torch.optim.Adam([self._value], lr=self.kl_lr)
