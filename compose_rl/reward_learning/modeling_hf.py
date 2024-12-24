# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Implements AutoModelForCausalLMWithRM wrapped in :class:`.ComposerModel`."""

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)

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

from compose_rl.reward_learning.hf_utils import (
    AutoModelForCausalLMWithRM,
    RewardModelConfig,
)

if TYPE_CHECKING:
    from peft import PeftConfig, PeftModel

__all__ = ['ComposerHFSequenceClassification']

log = logging.getLogger(__name__)

Tokenizer = Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]


class ComposerHFSequenceClassification(BaseHuggingFaceModel):
    """Configures a :class:`.HuggingFaceModel` around a Reward Model.

    Args:
        pretrained_model_name_or_path (str): The name of or local path to
            the HF Causal LM (e.g., `gpt2` to instantiate a GPT2LMHeadModel).
        config_overrides (dict, optional): An optional dictionary of keyword
            arguments that override the default configuration associated with
            cfg.pretrained_model_name_or_path.
        pretrained (bool): Whether to instantiate the model with pre-trained
            weights coming from cfg.pretrained_model_name_or_path. If ``True``,
            cfg.config_overrides must be compatible with the pre-trained weights.
        init_device ('cpu' | 'meta'): Which device, 'cpu' or 'meta', to
            initialize the model on. Currently, `meta` is only supported when
            cfg.pretrained is ``False``. Default: ``'cpu'``.
        peft_config (dict, optional): An optional dictionary of keyword arguments to be
            passed to the PeftConfig constructor. If provided, the model will be wrapped in a PeftModel.
        trust_remote_code (bool, optional): Whether to trust remote code when loading from Hugging Face
            Hub. Default: ``True``.
        use_auth_token (bool, optional): Whether to use the Hugging Face authentication token when
            loading from Hugging Face Hub. Default: ``False``.
        use_train_metrics (bool, optional): Whether to use training metrics. Default: ``True``.
        load_in_8bit (bool, optional): Whether to load the model in 8-bit mode. Default: ``False``.
        init_device (str, optional): Which device to initialize the model on. Default: ``'cpu'``.
        use_flash_attention_2 (bool, optional): Whether to use flash-attention 2. Default: ``False``.
        tokenizer (PreTrainedTokenizer): The tokenizer that the model will use.
    """
    model_cls: Union[
        _BaseAutoModelClass,
        PreTrainedModel] = AutoModelForCausalLMWithRM  # type: ignore
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
        should_save_peft_only: bool = True,
    ):

        config_overrides = config_overrides or {'return_logits': False}

        super().__init__(
            pretrained_model_name_or_path,
            tokenizer=tokenizer,
            pretrained=pretrained,
            pretrained_lora_id_or_path=pretrained_lora_id_or_path,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            use_flash_attention_2=use_flash_attention_2,
            load_in_8bit=load_in_8bit,
            init_device=init_device,
            config_overrides=config_overrides,
            shift_labels=True,
            peft_config=peft_config,
            allow_embedding_resizing=allow_embedding_resizing,
            use_train_metrics=use_train_metrics,
            additional_train_metrics=additional_train_metrics,
            additional_eval_metrics=additional_eval_metrics,
            should_save_peft_only=should_save_peft_only,
        )
        self.model.config.pretrained = False

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
            token=True,
            attn_implementation=attn_implementation,
            use_cache=
            False,  # Necessary due to https://github.com/huggingface/transformers/issues/28056
        )

        pretrain_cfg = {
            'trust_remote_code': trust_remote_code,
            'token': True,
        }

        config = RewardModelConfig(
            base_model=pretrained_model_name_or_path,
            base_config=base_config,
            hidden_size=base_config.hidden_size,
            vocab_size=base_config.vocab_size,
            pretrain_cfg=pretrain_cfg,
        )

        set_config_overrides(config, config_overrides)

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
            base_model = model.lm_backbone.model  # NOTE: Llama based. Make more general
        except:
            base_model = model.lm_backbone.transformer

        model_block = hf_get_hidden_layers(base_model)
        score_head = model.value_head
        lm_head = model.lm_backbone.get_output_embeddings()

        # Try to get input embeddings from the transformer backbone
        # and then from the XXXForCausalLM
        try:
            tied_embeddings = base_model.get_input_embeddings()
        except:
            tied_embeddings = model.get_input_embeddings()

        modules = {
            'base_model': base_model,
            'model_block': model_block,
            'score_head': score_head,
            'tied_embeddings': tied_embeddings,
            'lm_head': lm_head,
        }

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
        if model.config.tie_word_embeddings:
            base_model._fsdp_wrap = False
            tied_embeddings._fsdp_wrap = False  # type: ignore
            score_head._fsdp_wrap = False
            lm_head._fsdp_wrap = False

        # PEFT layers should be individually wrapped
        # TODO: Revisit this if we enforce use_orig_params=True, which seems to support
        # mixed frozen/unfrozen FSDP modules
        if hasattr(model, 'peft_type') and model.peft_type is not None:
            peft_type = model.peft_type.lower()
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
