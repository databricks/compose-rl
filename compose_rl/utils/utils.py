# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import warnings
from typing import Optional, Union

import spacy
import spacy_alignments as tokenizations
import torch
import torch.nn.functional as F
from kubernetes import client, config
from torch.utils.data import DataLoader
from transformers import PretrainedConfig

try:
    from composer.utils import dist
    from omegaconf import DictConfig
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

except ImportError as e:
    raise ImportError(
        'Please make sure to pip install from github.com/mosaicml/llm-foundry ',
        'to get the proper requirements.',
    ) from e

# MegaBlocks MoE load balancing loss
try:
    from megablocks.layers.moe import (
        batched_load_balancing_loss,
        clear_load_balancing_loss,
    )
except:
    batched_load_balancing_loss, clear_load_balancing_loss = None, None

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
log = logging.getLogger(__name__)

MODEL_DTYPES = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def get_mb_load_balancing_loss(
    cfg: PretrainedConfig,
    transformer: torch.nn.Module,
) -> Union[torch.Tensor, None]:
    if cfg.ffn_config['ffn_type'] in (
        'mb_moe',
        'mb_dmoe',
    ) and transformer.training:
        assert batched_load_balancing_loss is not None
        return batched_load_balancing_loss(transformer.mb_args)
    return None


def clear_mb_load_balancing_loss(
    cfg: PretrainedConfig,
    transformer: torch.nn.Module,
) -> None:
    if cfg.ffn_config['ffn_type'] in (
        'mb_moe',
        'mb_dmoe',
    ) and transformer.training:
        assert clear_load_balancing_loss is not None
        clear_load_balancing_loss()


def approx_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Approximates the KL divergence between two distributions P, Q.

    Approximates the KL Divergence between P, Q given the log probabilities,
    log_p and log_q. Taken from: http://joschu.net/blog/kl-approx.html.

    The estimator is unbiased and lower variance than if we were to apporximate
    it by just `-ratio.`

    Args:
        log_p (torch.Tensor): log probabilities for the distribution p.
        log_q (torch.Tensor): log probabilities for the distribution q.
    """
    ratio = log_p - log_q
    approx_kl = (ratio.exp() - 1) - ratio
    return approx_kl


def get_log_probs(
    logits: torch.Tensor,
    actions: torch.Tensor,
    prompt_len: torch.Tensor,
    max_gen_len: Union[torch.Tensor, int],
):
    """Gets the log probs from the generated logits.

    Inputs:
        - logits (torch.Tensor): the logits of the actions. Size (bs, seq_len + gen_len, vocab_size)
        - actions (torch.Tensor): the actions taken, typically tokens generated. Size (bs, gen_len)
        - prompt_len (torch.Tensor): length of the prompt.
        - max_gen_len (int): maximum generation length.
    """
    gen_logits = get_batched_generated_values(logits, prompt_len, max_gen_len)
    return get_log_probs_from_logits(gen_logits, actions)


def switch_left_to_right_padding(
    sequences: torch.Tensor,
    seq_length: torch.Tensor,
    max_gen_len: int,
    pad_token: int,
) -> torch.Tensor:
    """Switches left padding to right padding.

    Inputs:
        - sequences (torch.Tensor): The sequences we want to swap padding (of dimension seq_len + max_gen_len).
        - seq_len (torch.Tensor): the input prompt lengths.
        - max_gen_len (int): the maximum generation length.
        - pad_token (int): the pad token.
    """
    unpadded_sequences = remove_left_padding(sequences, seq_length, max_gen_len)
    max_len = seq_length.max() + max_gen_len  # type: ignore
    return add_right_padding(unpadded_sequences, int(max_len), pad_token)


def remove_left_padding(
    sequences: torch.Tensor,
    seq_length: torch.Tensor,
    max_gen_len: int,
) -> list[torch.Tensor]:
    """Removes left padding from a set of sequences.

    Inputs:
        - sequences (torch.Tensor): the sequences to remove left padding from.
        - seq_len (torch.Tensor): the input prompt lengths.
        - max_gen_len (int): the maximum generation length.
    """
    batch_size, _ = sequences.shape
    unpadded_obs = [
        sequences[i, -(seq_length[i] + max_gen_len):]
        for i in range(batch_size)
    ]
    return unpadded_obs


def add_right_padding(
    unpadded_sequences: list[torch.Tensor],
    max_len: int,
    pad_token: int,
) -> torch.Tensor:
    """Right pad a list of sequences to a given length.

    Inputs:
        - unpadded_sequences (list[torch.Tensor]): a list of unpadded sequences.
        - max_len (int): the maximum length we want the sequences to be padded.
        - pad_token (int): the pad token id that we want to pad sequences.
    """
    right_padded_obs = [
        torch.cat([
            seq,
            torch.ones(max_len - len(seq), device=seq.device, dtype=seq.dtype) *
            pad_token,
        ]) for seq in unpadded_sequences
    ]
    return torch.stack(right_padded_obs, dim=0)


def get_batched_generated_values(
    batched_values: torch.Tensor,
    prompt_len: torch.Tensor,
    max_gen_len: Union[torch.Tensor, int],
) -> torch.Tensor:
    """From a set of batched prompts + max_gen_len, return the generated values.

    Inputs:
        - batched_values (torch.Tensor): The batched generated values.
        - prompt_len (torch.Tensor): A tensor where each entry is the prompt length.
        - max_gen_len (int): the maximum generated length.
    """
    generations = []
    for i in range(batched_values.size(0)):
        curr_max_gen_len = max_gen_len
        if isinstance(max_gen_len, torch.Tensor):
            curr_max_gen_len = max_gen_len[i]  # pyright: ignore
            assert not curr_max_gen_len.is_floating_point()

        generations.append(
            batched_values[i, prompt_len[i] - 1:prompt_len[i] +
                           curr_max_gen_len - 1],
        )
    return torch.stack(generations, dim=0)


def masked_sum(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
):
    """Compute sum of a tensor with masked values."""
    if dim is not None:
        return (values * mask).sum(dim=dim)
    return (values * mask).sum()


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
):
    """Compute mean of a tensor with masked values."""
    if dim is not None:
        return (values * mask).sum(dim=dim) / mask.sum(dim=dim)
    return (values * mask).sum() / mask.sum()


def masked_var(
    values: torch.Tensor,
    mask: torch.Tensor,
    unbiased: Optional[bool] = True,
):
    """Compute variance of tensor with masked values.

    modified from: https://github.com/lvwerra/trl/blob/main/trl/core.py
    """
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        bessel_correction = mask.sum() / (mask.sum() - 1)
        variance = variance * bessel_correction
    return variance


def masked_normalize(
    values: torch.Tensor,
    masked_mean: torch.Tensor,
    masked_var: torch.Tensor,
    shift_mean: Optional[bool] = True,
):
    """Normalize values according to their masked mean and variance."""
    normalized = (values - masked_mean) * torch.rsqrt(masked_var + 1e-8)
    if not shift_mean:
        normalized += masked_mean
    return normalized


def masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    shift_mean: Optional[bool] = True,
):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    return masked_normalize(values, mean, var, shift_mean)


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_gae: float,
):
    """Computes the advantages from rewards, values.

    Note: this function assumes that we have right padded the values with zeros.

    Inputs:
        - rewards (torch.Tensor): The total rewards (environment + non-environment rewards)
        - values (torch.Tensor): The values for the predicted generations per state.
        - gamma (float): The discount factor.
        - lambda_gae (float): lambda value for the generalized advantage estimate.
    """
    assert (values[:, -1] == 0).all()

    # Advantage computation, getting the deltas
    deltas = rewards + gamma * values[:, 1:] - values[:, :-1]
    advantages = torch.zeros_like(deltas)
    advantages[:, -1] = deltas[:, -1]
    discount = gamma * lambda_gae

    # Advantage computation based on deltas
    for t in reversed(range(deltas.size(1) - 1)):
        advantages[:, t] = deltas[:, t] + discount * advantages[:, t + 1]

    return advantages


def dist_compute_masked_mean_and_var(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    unbiased: Optional[bool] = True,
):
    """Computes the distributed masked mean and variance of a tensor."""
    assert len(tensor.shape) == 2

    num_unmasked_elements = mask.sum()

    # Get the masked tensor sum
    masked_tensor_sum = (tensor * mask).sum()

    dist.all_reduce(num_unmasked_elements)
    dist.all_reduce(masked_tensor_sum)

    global_tensor_mean = masked_tensor_sum / num_unmasked_elements

    centered_values = (tensor - global_tensor_mean)**2
    centered_values *= mask
    centered_values = centered_values.sum()

    dist.all_reduce(centered_values)
    global_variance = centered_values / num_unmasked_elements
    if unbiased:
        bessel_correction = num_unmasked_elements / (num_unmasked_elements - 1)
        global_variance *= bessel_correction

    return global_tensor_mean, global_variance


def get_training_dataloader_state_dict(
    dataloader: DataLoader,
    cfg: DictConfig,
    iter_num: int,
) -> Optional[dict]:
    """Gets the training DataLoader state dict.

    Args:
        dataloader (DataLoader): the dataloader to get the state dict from.
        cfg: the training config.
        iter_num (int): The number of iterations of RL training.
    """
    # The iteration batch size
    per_iter_global_train_batch_size = cfg.global_train_batch_size * cfg.num_batches_per_update

    # We can loop through the dataloader multiple times
    # so we need to get the true iteration number in the dataloader
    cur_dataloader_iter = iter_num % len(dataloader)

    dataset = dataloader.dataset

    if hasattr(dataset, 'state_dict') and callable(
        dataset.state_dict,  # pyright: ignore
    ):
        num_samples = per_iter_global_train_batch_size * cur_dataloader_iter
        state_dict: dict = dataset.state_dict(  # pyright: ignore[reportGeneralTypeIssues]
            num_samples, True)
        return state_dict
    else:
        warnings.warn(
            f"The dataloader doesn't have a state dict. This will cause issues with saving and loading the state dict.",
        )


def mask_eos(
    actions: torch.Tensor,
    right_padded_obs: torch.Tensor,
    right_padded_attn_mask: torch.Tensor,
    prompt_len: torch.Tensor,
    generated_len: torch.Tensor,
    max_gen_len: int,
    eos_token: int,
    pad_token: int,
):
    """Mask EOS tokens in a given sequence and returns appropriate values.

    Inputs:
        - actions (torch.Tensor): the actions taken (tokens generated).
        - right_padded_obs (torch.Tensor): the right padded observation.
        - right_padded_attn_mask (torch.Tensor): the right padded attention mask.
        - prompt_len (torch.Tensor): the prompt length.
        - generated_len (torch.Tensor): the generated length for each prompt.
        - max_gen_len (int): the maximum generated length.
        - eos_token (int): the token representing end of sequence token.
        - pad_token (int): the token representing pad token.
    """
    # Creating appropriate masks based upon EOS appearing in sequences
    eos_actions = actions == eos_token
    action_mask = torch.ones_like(actions)
    seen_eos_batches = set()
    for eos_idx in eos_actions.nonzero(as_tuple=False):
        batch_idx = int(eos_idx[0])
        if eos_idx[1] < max_gen_len and batch_idx not in seen_eos_batches:
            action_mask[eos_idx[0], eos_idx[1] + 1:] = 0
            generated_len[eos_idx[0]] = eos_idx[1] + 1
            seen_eos_batches.add(batch_idx)

            # We need to refix all of the padding since we now always generate max_gen_len tokens
            req_pad_start_idx = prompt_len[eos_idx[0]
                                          ] + generated_len[eos_idx[0]]
            right_padded_obs[eos_idx[0], req_pad_start_idx:] = pad_token
            right_padded_attn_mask[eos_idx[0], req_pad_start_idx:] = False

    return right_padded_obs, right_padded_attn_mask, generated_len, action_mask


def get_decoded_sequence(
    sequence: torch.Tensor,
    generated_len: torch.Tensor,
    max_gen_len: int,
):
    decoded_sequence = sequence[:generated_len]
    if generated_len == max_gen_len:
        decoded_sequence = sequence
    return decoded_sequence


def split_text_to_sentences(long_text: str, parser: spacy.Language):
    doc = parser(long_text)
    return [0] + [
        sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0
    ]


def split_text_to_subsentences(
    long_text: str,
    parser: spacy.Language,
    min_subsent_words: int = 5,
):
    """Splits sentence to subsentences.

    This splits sentences to subsentences then checks for clause ends via punctuation.

    Args:
        long_text (string): Long piece of text to split
        parser (spacy.Language): A spacy or nltk sentence level tokenizer
        min_subsent_words (int): Min num of words for a subseq, default ported
        from original fine-grained rewards paper https://arxiv.org/abs/2306.01693
    """

    def get_sub_sentence_starts(
        tokens: torch.Tensor,
        min_subsent_words: int = min_subsent_words,
    ):

        def _is_tok_end_of_subsent(tok: str):
            if re.match('[,;!?]', tok[-1]) is not None:
                return True
            return False

        is_subsent_starts = [True]
        prev_tok = tokens[0]
        prev_subsent_start_idx = 0
        for i, tok in enumerate(tokens[1:]):
            tok_id = i + 1
            if _is_tok_end_of_subsent(
                prev_tok,  # type: ignore
            ) and tok_id + min_subsent_words < len(tokens):
                if tok_id - prev_subsent_start_idx < min_subsent_words:
                    if prev_subsent_start_idx > 0:
                        is_subsent_starts += [True]
                        is_subsent_starts[prev_subsent_start_idx] = False
                        prev_subsent_start_idx = tok_id
                    else:
                        is_subsent_starts += [False]
                else:
                    is_subsent_starts += [True]
                    prev_subsent_start_idx = tok_id
            else:
                is_subsent_starts += [False]
            prev_tok = tok

        return is_subsent_starts

    def tokenize_with_indices(text: str):
        tokens = text.split()
        token_indices = []

        current_index = 0
        for token in tokens:
            start_index = text.find(token, current_index)
            token_indices.append((token, start_index))
            current_index = start_index + len(token)

        return token_indices

    sentence_start_char_idxs = split_text_to_sentences(long_text, parser)

    char_starts = []

    for sentence_idx, sentence_start_char_idx in enumerate(
        sentence_start_char_idxs[:-1],
    ):

        sentence = long_text[
            sentence_start_char_idx:sentence_start_char_idxs[sentence_idx + 1]]

        tokens_with_indices = tokenize_with_indices(sentence)

        tokens = [i[0] for i in tokens_with_indices]
        is_sub_starts = get_sub_sentence_starts(
            tokens,  # type: ignore
            min_subsent_words=min_subsent_words,
        )
        for token_with_idx, is_sub_start in zip(
            tokens_with_indices,
            is_sub_starts,
        ):
            if is_sub_start:
                char_starts.append(sentence_start_char_idx + token_with_idx[1])

    return char_starts + [len(long_text)]


def rescale(tensor: torch.Tensor, scale_min: float, scale_max: float):
    """Rescales tensor between given min and max."""
    return (tensor - scale_min) / (scale_max - scale_min)


def format_reward_input(
    prompt: str,
    generated_sequences: list[str],
    tokenizer: Tokenizer,
):
    """Formatting for reward strings before they go into reward model.

    Args:
        prompt (str): prompt
        generated_subsequences (list[str]): list of split subsequences of generated text

    Returns:
        _type_: formatted input to reward fn
    """
    # Add formatting for reward input here, separate with SEPs depending on training
    join_token = ''
    if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token is not None:
        join_token = tokenizer.sep_token
    formatted_sequence = prompt + join_token.join(generated_sequences)
    return formatted_sequence


def process_fine_granularities(
    prompt: str,
    prompt_len: int,
    generated: str,
    generated_len: int,
    original_obs: Union[torch.Tensor, list[int]],
    granularity: str,
    parser: spacy.Language,
    tokenizer: Tokenizer,
    max_seq_len: int,
):
    """Processes a single generation for fine-grained rewards.

    Takes a single generation, processes it by parsing and figuring out
    indices where to apply rewards. Then aligns indices between parser tokens
    and current tokens, and current tokens and previously encoded tokens
    (because tokenizers are weird)

    Args:
        prompt (str): Original prompt
        prompt_len (int): Total length of original prompt
        generated (str): Original generated text
        generated_len (int): Total length of original generated text
        original_obs (list[int]): Original observation in token ids
        granularity (str): Granularity type
        parser (): Spacy or other sent/subsent parser
        tokenizer (Tokenizer): Tokenizer
        max_seq_len (int): Maximum total sequence length

    Returns:
        reward_input (str): string formatted reward inputs
        end_indices_aligned_gather (list(int)): indices at which to gather rewards from reward
        end_indices_aligned_scatter: indices at which to scatter rewards to for RL
    """
    if granularity == 'subsentence':
        end_char_idxs = split_text_to_subsentences(generated, parser)
    elif granularity == 'sentence':
        end_char_idxs = split_text_to_sentences(generated, parser)
    elif granularity == 'document':
        end_char_idxs = [0, len(generated)]
    else:
        raise NotImplementedError(f'{granularity=} is not supported.')
    generated_sequences = [
        generated[end_char_idxs[i]:end_char_idxs[i + 1]]
        for i in range(len(end_char_idxs) - 1)
    ]

    # Initialize an empty list to store the end token indices of each sentence
    unaligned_end_indices = []
    concatenated_subseq_tokens = []

    # Calculate the end indices of where to take rewards after parsing
    for sent_idx in range(len(generated_sequences)):
        tokens = tokenizer.tokenize(generated_sequences[sent_idx])
        concatenated_subseq_tokens.extend(tokens)
        token_count = len(concatenated_subseq_tokens)
        unaligned_end_indices.append(token_count - 1)

    # Reencode the prompt, encode -> decode -> encode is not stable
    # but text -> encode is stable
    reward_prompt_tokens = tokenizer.tokenize(prompt)
    reward_prompt_len = len(reward_prompt_tokens)

    # Format the reward
    reward_input = format_reward_input(
        prompt,
        generated_sequences=generated_sequences,
        tokenizer=tokenizer,
    )
    tokenized_reward_input = tokenizer.tokenize(reward_input)
    tokenized_reward_input = [
        t for t in tokenized_reward_input if t is not None
    ]
    concatenated_subseq_tokens = [
        t for t in concatenated_subseq_tokens if t is not None
    ]
    # Truncate here to prevent scatter gather indices from going over
    tokenized_reward_input = tokenized_reward_input[:max_seq_len]

    # Alternate solution incase tokenizer text -> encode isn't stable
    # reward_tokens_alignment_forward, reward_tokens_alignment_backward = \
    #     tokenizations.get_alignments(tokenized_reward_input, reward_prompt_tokens)
    # reward_prompt_len = reward_tokens_alignment_backward[-1][-1] # + 1 if rewards end in wspace

    reward_seq_len = len(tokenized_reward_input)
    tokenized_generated_input = tokenized_reward_input[reward_prompt_len:]
    reward_generated_len = len(tokenized_generated_input)

    # Align indices of where to gather the rewards from the reward formatted input
    # this alignment is between the spacy tokenizer and the MPT tokenizer on reward input

    parser_alignment_forward, parser_alignment_backward = tokenizations.get_alignments(
        tokenized_generated_input,
        concatenated_subseq_tokens,
    )
    del parser_alignment_forward  # unused

    end_indices_aligned_gather = []
    for end_idx in unaligned_end_indices:
        if len(parser_alignment_backward[end_idx]) > 0:
            end_indices_aligned_gather.append(
                parser_alignment_backward[end_idx][-1],
            )
        else:
            # Document weird edge cases where unicode based sequence alignment fails
            log.warning(
                f'Tokenizer sequence alignment failed. You\'ve been hit by a smooth {concatenated_subseq_tokens[end_idx]}',
            )

    # The original tokenized obses RL training sees, without the decode step
    original_generated_token_ids = original_obs[prompt_len:prompt_len +
                                                generated_len]
    original_generated_tokens = tokenizer.convert_ids_to_tokens(
        original_generated_token_ids,
    )

    # Align between the tokenized outputs of the policy and the tokenized inputs to reward
    original_generated_tokens = [
        t for t in original_generated_tokens if t is not None
    ]
    tokenized_generated_input = [
        t for t in tokenized_generated_input if t is not None
    ]
    og_alignment_forward, og_alignment_backward = tokenizations.get_alignments(
        original_generated_tokens,
        tokenized_generated_input,
    )
    del og_alignment_forward  # unused

    end_indices_aligned_scatter = []
    # Track the smooth criminal tokens that break sequence alignment
    failed_align_end_idxs = []
    for i, end_idx in enumerate(end_indices_aligned_gather):
        if len(og_alignment_backward[end_idx]) > 0:
            end_indices_aligned_scatter.append(
                og_alignment_backward[end_idx][-1],
            )
        else:
            # Document weird edge cases where unicode based sequence alignment fails
            log.warning(
                f'Tokenizer sequence alignment failed. You\'ve been struck by a smooth {i, tokenized_generated_input[end_idx]}',
            )
            failed_align_end_idxs.append(i)
    # Get rid of indices in the final gather where the sequence alignment fails
    end_indices_aligned_gather = [
        u for i, u in enumerate(end_indices_aligned_gather)
        if i not in failed_align_end_idxs
    ]
    assert len(end_indices_aligned_gather) == len(end_indices_aligned_scatter)

    # last token cutoffs and additions
    end_indices_aligned_gather = [
        min(item, reward_generated_len - 1)
        for item in end_indices_aligned_gather
    ]

    end_indices_aligned_scatter = [
        min(item, generated_len - 1) for item in end_indices_aligned_scatter
    ]

    # Special edge case for document level rewards
    if granularity == 'document':
        end_indices_aligned_gather = [reward_generated_len - 1]
        end_indices_aligned_scatter = [generated_len - 1]

    # Special edge case when all sequence alignment fails, fall through to document level rewards
    # TODO Note that we might want to change this behavior in the future depending on vendor data
    if (
        len(
            end_indices_aligned_gather,
        ) == 0 or reward_generated_len - 1 not in end_indices_aligned_gather
    ):
        end_indices_aligned_gather.append(reward_generated_len - 1)
    if (
        len(
            end_indices_aligned_scatter,
        ) == 0 or generated_len - 1 not in end_indices_aligned_scatter
    ):
        end_indices_aligned_scatter.append(generated_len - 1)

    return reward_input, reward_prompt_len, reward_generated_len, reward_seq_len, \
        end_indices_aligned_gather, end_indices_aligned_scatter


def batch_process_fine_granularities(
    raw_untokenized_texts: list[tuple],
    granularity_types: list[str],
    generated_lens: list[int],
    parser: spacy.Language,
    tokenizer: Tokenizer,
    prompt_lens: list[int],
    original_obses: Union[torch.Tensor, list[list[int]]],
    max_seq_len: int,
    device: Optional[str] = 'cpu',
):
    """Processes batch into inputs and end indices for fine-grained rewards.

    Args:
        raw_untokenized_texts (list[tuple]): tuples of raw str (prompt, generation) pairs
        granularity_types (list[str]): All granularities you want to process
        generated_lens (list[int]): Generated lengths
        parser (): Spacy parser (or other sentence tokenizers)
        tokenizer (Tokenizer): Tokenizer
        prompt_lens (list[int]): Original prompt lengths
        original_obses (tensor, list[list[int]]): Origin observations seen during RL training
        max_seq_len (int): Max sequence length for rewards to see
        device (str, optional): Device, defaults to 'cpu'.

    Returns:
        <end_idxs_gather_dict, end_idxs_scatter_dict, end_reward_inputs_dict,
        reward_seq_lens_dict, reward_prompt_lens_dict, reward_generated_lens_dict> :
        end indexs of indices to gather reward scores from reward model and to scatter to RL
        and overall prompt, generated, seq lengths, all dicts keyed on granularity type
    """
    # TODO (raj) MPT specific, make custom with data loader changes
    tokenizer.padding_side = 'right'

    end_idxs_gather_dict = {}
    end_idxs_scatter_dict = {}

    end_reward_inputs_dict = {}

    reward_seq_lens_dict = {}
    reward_prompt_lens_dict = {}
    reward_generated_lens_dict = {}

    for granularity in granularity_types:
        formatted_reward_inputs = []

        gather_all_sequence_end_idxs = []
        scatter_all_sequence_end_idxs = []

        reward_prompt_lens = []
        reward_generated_lens = []
        reward_seq_lens = []
        for i, (raw_untokenized_text) in enumerate(raw_untokenized_texts):
            prompt, generated = raw_untokenized_text
            reward_input, reward_prompt_len, reward_generated_len, reward_seq_len, end_indices_aligned_gather, \
                end_indices_aligned_scatter = process_fine_granularities(
                prompt=prompt, prompt_len=prompt_lens[i], generated=generated,
                granularity=granularity, generated_len=generated_lens[i],
                parser=parser, tokenizer=tokenizer, original_obs=original_obses[i],
                max_seq_len=max_seq_len,
            )

            reward_prompt_lens.append(reward_prompt_len)
            reward_generated_lens.append(reward_generated_len)
            reward_seq_lens.append(reward_seq_len)

            formatted_reward_inputs.append(reward_input)

            gather_all_sequence_end_idxs.append(
                torch.LongTensor(end_indices_aligned_gather).to(device),
            )
            scatter_all_sequence_end_idxs.append(
                torch.LongTensor(end_indices_aligned_scatter).to(device),
            )

        end_idxs_gather_dict[granularity] = gather_all_sequence_end_idxs
        end_idxs_scatter_dict[granularity] = scatter_all_sequence_end_idxs

        reward_prompt_lens_dict[granularity] = torch.LongTensor(
            reward_prompt_lens,
        ).to(device)
        reward_generated_lens_dict[granularity] = torch.LongTensor(
            reward_generated_lens,
        ).to(device)
        reward_seq_lens_dict[granularity] = torch.LongTensor(
            reward_seq_lens,
        ).to(device)

        outputs = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=formatted_reward_inputs,
            padding='longest',
            truncation=True,
            max_length=max_seq_len,
            return_attention_mask=True,
        )
        end_reward_inputs_dict[granularity] = outputs

    return {
        'end_idxs_gather_dict': end_idxs_gather_dict,
        'end_idxs_scatter_dict': end_idxs_scatter_dict,
        'end_reward_inputs_dict': end_reward_inputs_dict,
        'reward_seq_lens_dict': reward_seq_lens_dict,
        'reward_prompt_lens_dict': reward_prompt_lens_dict,
        'reward_generated_lens_dict': reward_generated_lens_dict,
    }


def scatter_gather_rewards(
    temp_rews: torch.Tensor,
    curr_rewards: torch.Tensor,
    reward_prompt_lens: torch.Tensor,
    prompt_lens: torch.Tensor,
    reward_generated_lens: torch.Tensor,
    generated_lens: torch.Tensor,
    reward_seq_lens: torch.Tensor,
    seq_lens: Union[torch.Tensor, list[torch.Tensor]],
    end_idxs_gather: list[torch.Tensor],
    end_idxs_scatter: list[torch.Tensor],
):
    """Gather rewards from reward output and scatter to use with RL.

    Takes raw reward output from model and scatters only rewards from
    end_idxs to the into 0s tensor.

    Args:
        temp_rews (torch.Tensor): Initially a 0s tensor
        curr_rewards (torch.Tensor): Raw scores from model
        prompt_lens (torch.Tensor): Prompt lengths
        generated_lens (torch.Tensor): Generated lengths
        seq_lens (torch.Tensor): The whole seq lengths
        reward_prompt_lens (torch.Tensor): Prompt lengths from reward input
        reward_generated_lens (torch.Tensor): Generated lengths from reward input
        reward_seq_lens (torch.Tensor): The whole sequence lengths from reward input
        end_idxs (torch.Tensor): Indices at which we want to scatter rewards

    Returns:
        torch.Tensor: rewards
    """
    batch_size = curr_rewards.shape[0]
    for i in range(batch_size):
        # Prompt length and generated length together should gives us sequence length
        assert (prompt_lens[i] + generated_lens[i]).item() == seq_lens[i].item()
        assert (reward_prompt_lens[i] +
                reward_generated_lens[i]).item() == reward_seq_lens[i].item()
        # The number of indices you gather rews from outputs is same as scatter
        assert end_idxs_scatter[i].shape[-1] == end_idxs_gather[i].shape[-1]
        batch_curr_rewards = curr_rewards[
            i, reward_prompt_lens[i]:reward_prompt_lens[i] +
            reward_generated_lens[i]]
        gathered_rewards = batch_curr_rewards.gather(
            dim=0,
            index=end_idxs_gather[i],
        )
        temp_rews[i] = temp_rews[i].scatter(
            0,
            end_idxs_scatter[i],
            gathered_rewards,
        )
    return temp_rews


def flip_pad_token_usage_for_generate(model: torch.nn.Module):
    """Determines the pad token usage and flips if necessary for generate.

    This function assumes that the model is wrapped by FSDP. If we don't flip the usage from `False`
    to `True` before generate, then the kernel we use to remove pad tokens will throw an error.

    Args:
        model (torch.nn.Module): a torch model
    Returns:
        needs_flipping (bool): represents if we needed to flip the pad token usage.
    """
    needs_flipping = False
    if not hasattr(model, 'transformer'):
        return needs_flipping
    assert len(model.transformer.blocks) > 0
    block = model.transformer.blocks[0]
    # Logic takes care of the activation checkpointing case w/ FSDP
    if hasattr(block._fsdp_wrapped_module, '_checkpoint_wrapped_module'):
        needs_flipping = not block._fsdp_wrapped_module._checkpoint_wrapped_module.use_pad_tok_in_ffn
    else:
        # Otherwise we avoid the activation checkpointing and toggle the flag here
        needs_flipping = not block._fsdp_wrapped_module.use_pad_tok_in_ffn

    if needs_flipping:
        flip_pad_token_usage_in_ffn(model)

    return needs_flipping


def flip_pad_token_usage_in_ffn(model: torch.nn.Module):
    """Flips the pad token usage for a given model.

    This function assumes that the model is wrapped by FSDP.

    Args:
        model (torch.nn.Module): a torch model.
    """
    for block in model.transformer.blocks:

        # Logic takes care of the activation checkpointing case w/ FSDP
        if hasattr(block._fsdp_wrapped_module, '_checkpoint_wrapped_module'):
            block._fsdp_wrapped_module._checkpoint_wrapped_module.use_pad_tok_in_ffn = not block._fsdp_wrapped_module._checkpoint_wrapped_module.use_pad_tok_in_ffn
        else:
            # Otherwise we avoid the activation checkpointing and toggle the flag here
            block._fsdp_wrapped_module.use_pad_tok_in_ffn = not block._fsdp_wrapped_module.use_pad_tok_in_ffn


def get_remote_name(pod_name: str):
    """Gets the remote name given the kubernetes pod name.

    This function gets the appropriate API address to query the remote model given the last resumption ID of the kubernetes pod that is hosting the remote server.
    NOTE: for Mosaic clusters, if using a dependent deployment, the dependent deployment must be in the same cluster as the training run for this function to work.

    Args:
        pod_name (str): The pod name of the last resumption ID from kubernetes.
    """
    NAMESPACE_FILE = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    with open(NAMESPACE_FILE, 'r') as file:
        namespace = file.read().replace('\n', '')

    # You get this by doing mcli describe run, getting the Last Resumption ID and appending -0 to it
    api_response = v1.read_namespaced_pod(pod_name, namespace, pretty=True)
    return f'http://{api_response.status.pod_ip}:8080/v2/completions'  # pyright: ignore


def get_log_probs_from_logits(logits: torch.Tensor, actions: torch.Tensor):
    """Gets the log probabilities from a set of logits.

    This code is taken from:
    https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, actions.unsqueeze(2).long()).squeeze(-1)
    return logpy


def extract_packed_chosen_rejected(
    input_tensor: torch.Tensor,
    chosen_len: torch.Tensor,
    rejected_len: torch.Tensor,
    max_seq_len: int,
    pad_token_id: int,
):
    """Extracts the chosen and rejected values from the tensor.

    Args:
        input_tensor (torch.Tensor): the tensor with concatenated chosen and rejected value
            that we want to extract values from.
        chosen_len (torch.Tensor): the length of the chosen responses (batch_size,)
        rejected_len (torch.Tensor): the length of the rejected responses (batch_size,)
        max_seq_len (int): the maximum sequence length that we want to pack the sequences
        pad_token_id (int): the token id we should be padding our values with
    """
    batch_size = input_tensor.size(0)
    chosen_values = []
    rejected_values = []
    for i in range(batch_size):
        unpadded_chosen = input_tensor[i, :chosen_len[i]]
        padded_chosen = make_padded_tensor(
            unpadded_chosen,
            max_seq_len,
            pad_token_id,
        )
        chosen_values.append(padded_chosen)

        unpadded_rejected = input_tensor[i, chosen_len[i]:chosen_len[i] +
                                         rejected_len[i]]
        padded_rejected = make_padded_tensor(
            unpadded_rejected,
            max_seq_len,
            pad_token_id,
        )
        rejected_values.append(padded_rejected)
    return torch.stack(chosen_values), torch.stack(rejected_values)


def make_padded_tensor(
    input_tensor: torch.Tensor,
    max_seq_len: int,
    pad_token_id: int,
):
    """Pads tensor to max_seq_len using pad_token_id.

    First dimension is always seq_len, pad tensor to max_seq_len.

    Args:
        input_tensor (torch.Tensor): the tensor we want to pad.
        max_seq_len (int): the maximum sequence length we want to pad the tensor
        pad_token_id (int): the token id we should be padding our values with
    """
    seq_len = input_tensor.size(0)
    pad_len = max_seq_len - seq_len
    if len(input_tensor.shape) == 1:
        pad_tensor = torch.ones(
            pad_len,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        ) * pad_token_id
    elif len(input_tensor.shape) == 2:
        pad_tensor = torch.ones((pad_len, input_tensor.size(1)),
                                device=input_tensor.device,
                                dtype=input_tensor.dtype) * pad_token_id
    else:
        raise NotImplementedError(
            f'Making a pad tensor of shape {input_tensor.shape} is not supported.',
        )
    return torch.cat([input_tensor, pad_tensor], dim=0)


def get_batch_logp(
    labels: torch.Tensor,
    logits: torch.Tensor,
    prompt_len: torch.LongTensor,
    prompt_gen_len: torch.LongTensor,
    average_log_prob: bool,
):
    """Gets the log probability for given labels and logits.

    Args:
        labels (torch.LongTensor): the labels to get the logits (batch_size, seq_len)
        logits (torch.FloatTensor): the logits over the entire sequence (batch_size, seq_len, vocab_size)
        prompt_len (torch.LongTensor): the length of the prompt (batch_size,)
        prompt_gen_len (torch.LongTensor): the length of the prompt and generated sequence (batch_size,)
        average_log_prob (bool): whether or not we should average the log prob
    """
    batch_size, _ = labels.shape
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1]

    # Dummy tokens, will mask this out later
    for i in range(batch_size):
        labels[i, prompt_gen_len[i]:] = 0

    log_p = get_log_probs_from_logits(logits, labels)

    action_mask = make_action_mask(
        prompt_len,
        prompt_gen_len,
        log_p.shape,
        device=log_p.device,
    )

    num_actions = action_mask.sum(dim=-1)
    if (num_actions == 0).any():
        num_actions[num_actions == 0] = 1

    if average_log_prob:
        return (log_p * action_mask).sum(-1) / num_actions
    else:
        return (log_p * action_mask).sum(-1)


def make_action_mask(
    prompt_len: torch.LongTensor,
    prompt_gen_len: torch.LongTensor,
    mask_shape: torch.Size,
    device: torch.device,
):
    """Makes a mask on what were 'generated' tokens for DPO (offline RL).

    Args:
        prompt_len (torch.LongTensor): A tensor representing lengths of the prompts
        prompt_gen_len (torch.LongTensor): A tensor representing the length of prompts and generated.
            Note: this includes the EOS token, so we need to mask it out
        mask_shape (torch.Size): the shape of the mask to create
        device (torch.device): the device to make the mask
    """
    batch_size, _ = mask_shape
    mask = torch.ones(mask_shape, device=device)
    for i in range(batch_size):
        # This is becuase the last token should be EOS, so
        # we don't want the probability of the next token
        mask[i, prompt_gen_len[i] - 1:] = 0

        # The last token in the prompt generates loss since
        # it's considered the first token the `model` generated.
        # All other prompt tokens should be masked.
        mask[i, :prompt_len[i] - 1] = 0

    return mask
