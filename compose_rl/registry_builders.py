# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Builds components for all the components in the Compose RL."""

from compose_rl.registry import KL_CONTROLLER_REGISTRY


def build_kl_controller(cfg: dict):
    """Builds the KL Controller from the config."""
    kl_ctl_type = cfg.get('kl_ctl_type', None)

    # Default KL Controller to Fixed Controller
    if kl_ctl_type is None:
        kl_ctl_type = 'fixed'
        cfg['kl_ctl_type'] = kl_ctl_type
        cfg['init_kl_coef'] = 0.05

    return KL_CONTROLLER_REGISTRY[kl_ctl_type](cfg)
