# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Implements the Composer."""

from typing import TYPE_CHECKING, Any, Optional, Union

from llmfoundry.models.hf.hf_base import BaseHuggingFaceModel
from llmfoundry.models.hf.hf_fsdp import (
    hf_get_hidden_layers,
)
from llmfoundry.utils.config_utils import set_config_overrides  # type: ignore
from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from compose_rl.ppo.hf_utils import AutoModelForCausalLMAsPolicy
from compose_rl.ppo.policy_configuration import HFPolicyConfig

if TYPE_CHECKING:
    from peft import PeftConfig, PeftModel

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class ComposerHFPolicy(BaseHuggingFaceModel):
    """Configures a :class:`.ComposerMosaicPolicy` as a Policy for PPO.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        pretrained_model_name_or_path (str): The pretrained model to use.
        pretrained (bool): Whether to use a pretrained model. Default: ``True``.
        pretrained_lora_id_or_path (str): The pretrained LoRA model to use. Default: ``None``.
        trust_remote_code (bool): Whether to trust remote code. Default: ``True``.
        use_auth_token (bool): Whether to use an auth token. Default: ``False``.
        use_flash_attention_2 (bool): Whether to use Flash Attention 2. Default: ``False``.
        load_in_8bit (bool): Whether to load the model in 8-bit mode. Default: ``False``.
        init_device (str): The device to initialize the model on. Default: ``'cpu'``.
        config_overrides (dict[str, Any]): Additional configuration overrides. Default: ``None``.
        peft_config (PeftConfig): The PEFT configuration. Default: ``None``.
        use_train_metrics (bool): Whether to use training metrics. Default: ``True``.
        allow_embedding_resizing (bool): Whether to allow embedding resizing. Default: ``True``.
        additional_train_metrics (list): Additional training metrics. Default: ``None``.
        additional_eval_metrics (list): Additional evaluation metrics. Default: ``None``.
        should_save_peft_only (bool): Whether to save only the PEFT model. Default: ``False``.
    """
    model_cls: Union[
        type[_BaseAutoModelClass],
        type[PreTrainedModel]] = AutoModelForCausalLMAsPolicy  # type: ignore
    default_train_metrics: tuple = ()
    default_eval_metrics: tuple = ()

    def __init__(
        self,
        tokenizer: Tokenizer,
        pretrained_model_name_or_path: str,
        pretrained: bool = True,
        pretrained_lora_id_or_path: Optional[str] = None,
        trust_remote_code: bool = True,
        use_auth_token: bool = False,
        use_flash_attention_2: bool = False,
        load_in_8bit: bool = False,
        init_device: str = 'cpu',
        config_overrides: Optional[dict[str, Any]] = None,
        peft_config: Optional['PeftConfig'] = None,
        use_train_metrics: bool = True,
        allow_embedding_resizing: bool = True,
        additional_train_metrics: Optional[list] = None,
        additional_eval_metrics: Optional[list] = None,
        should_save_peft_only: bool = False,
    ):
        # Change the model_cls if config_overrides provides joint_actor_critic False
        # if not config_overrides['joint_actor_critic']:
        #     self.model_cls = AutoModelFor
        # Second entrypoint
        super().__init__(
            pretrained_model_name_or_path,
            # tokenizer=tokenizer,
            pretrained=pretrained,
            pretrained_lora_id_or_path=pretrained_lora_id_or_path,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            use_flash_attention_2=use_flash_attention_2,
            load_in_8bit=load_in_8bit,
            init_device=init_device,
            config_overrides=config_overrides,
            shift_labels=True,
            peft_config=peft_config, # type: ignore
            allow_embedding_resizing=allow_embedding_resizing,
            use_train_metrics=use_train_metrics,
            additional_train_metrics=additional_train_metrics,
            additional_eval_metrics=additional_eval_metrics,
            should_save_peft_only=should_save_peft_only,
        )
        # print(f"&&&&&&&&&&&In HFPolicy init: {self.config=}")
        # print(f"&&&&&&&&&&&In HFPolicy init: {self.model.config=}")
        # # Fixed now. Getting false for both
        # breakpoint()
        self.model.config.pretrained = False  # type: ignore

    @classmethod
    def build_config(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool,
        use_auth_token: bool,
        attn_implementation: str,
        config_overrides: dict[str, Any],
    ) -> PretrainedConfig:

        base_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            token=use_auth_token,
            attn_implementation=attn_implementation,
            use_cache=
            False,  # Necessary due to https://github.com/huggingface/transformers/issues/28056
            torch_dtype=config_overrides.get('torch_dtype', 'float32'),
        )

        pretrain_cfg = {
            'trust_remote_code': trust_remote_code,
            'token': use_auth_token,
        }
        # print(f"******************In build config: {base_config=}")
        # print(f"******************In build config: {config_overrides['joint_actor_critic']=}")
        config = HFPolicyConfig(
            base_model=pretrained_model_name_or_path,
            base_config=base_config,
            hidden_size=base_config.hidden_size,
            vocab_size=base_config.vocab_size,
            pretrain_cfg=pretrain_cfg,
        )
        # print(f"******************In build config: {config=}")
        set_config_overrides(config, config_overrides)
        # print(f"******************In build config after set_config_overrides: {config=}")
        # breakpoint()

        return config

    @staticmethod
    def prepare_inner_model(
        model: Union[PreTrainedModel, 'PeftModel'],
        init_device: Optional[str] = None,
    ):
        """Prepare the inner model for FSDP wrapping.

        Args:
            model: The model to prepare.
            init_device: The device to initialize the model on.
        """
        # Note: We need to add the FSDP related attributes to the model AFTER the super init,
        # so that the (possible) embedding resizing doesn't destroy them
        #prepare_hf_model_for_fsdp(model.lm_backbone, init_device)
        try:
            # NOTE: Llama based. Make more general
            base_model = model.lm_backbone.model  # type: ignore
        except:
            base_model = model.lm_backbone.transformer  # type: ignore

        model_block = hf_get_hidden_layers(base_model)  # type: ignore
        
        lm_head = model.lm_backbone.get_output_embeddings()  # type: ignore

        # Try to get input embeddings from the transformer backbone
        # and then from the XXXForCausalLM
        try:
            tied_embeddings = base_model.get_input_embeddings()  # type: ignore
        except:
            tied_embeddings = model.get_input_embeddings()  # type: ignore

        modules = {
            'base_model': base_model,
            'model_block': model_block,
            'tied_embeddings': tied_embeddings,
            'lm_head': lm_head,
        }
        if hasattr(model, 'critic_head'):
            critic_head = model.critic_head
            modules['critic_head'] = critic_head

        for mod_name, module in modules.items():
            if module is None:
                raise ValueError(
                    f'Unable to FSDP-wrap this model! `{mod_name}` does not ' +
                    'follow common layer/weight naming conventions.',
                )
        block_type = type(model_block[0])

        # When using the HF LM models,
        # the weights of the self.lm_head and self.transformer.wte are tied.
        # This tying occurs inside the `self.post_init()` function.
        # This is a hurdle for FSDP because they need to be in the same FSDP block
        # These lines ensures that both modules stay together in the top-most block when
        # the model has this tying enabled (almost all do; this property defaults to True)
        if model.config.tie_word_embeddings:  # type: ignore
            base_model._fsdp_wrap = False  # type: ignore
            tied_embeddings._fsdp_wrap = False  # type: ignore
            lm_head._fsdp_wrap = False

        # PEFT layers should be individually wrapped
        # TODO: Revisit this if we enforce use_orig_params=True, which seems to support
        # mixed frozen/unfrozen FSDP modules
        if hasattr(model, 'peft_type') and model.peft_type is not None:
            peft_type = model.peft_type.lower()  # type: ignore
            active_adapters = [
                adapter.lower()
                for adapter in model.active_adapters  # type: ignore
            ]
            for name, module in model.named_modules():
                if peft_type in name.lower() and any(
                    adapter in name.lower() for adapter in active_adapters
                ):
                    has_parameters = next(module.parameters(), None) is not None
                    has_buffers = next(module.buffers(), None) is not None
                    if has_parameters or has_buffers:
                        module._fsdp_wrap = True  # type: ignore

        # FSDP Wrap and Activation Checkpoint every model block
        model.fsdp_wrap_fn = lambda module: isinstance(  # type: ignore
            module, block_type,
        )
        model.activation_checkpointing_fn = lambda module: isinstance( # type: ignore
            module,
            block_type,
        )

        # This provides support for meta initialization when using FSDP
        model.param_init_fn = lambda module: model._init_weights(  # type: ignore
            module,
        )
