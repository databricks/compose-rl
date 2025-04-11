# Data Processing Scripts

The `unified_tokenize_dataset.py` script writes out MDS datasets that can be used for training models with DPO or PPO.

## Verifiable Rewards

For enabling [Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2411.15124) (RLVR) with the above script you can run the script as follows:

```bash
python unified_tokenize_dataset.py --dataset_name <hf_dataset_name> --splits train [--subset <hf_dataset_subset>] --local_dir <save_path> --tokenizer_name <hf_tokenizer_name> --max_length <sequence_length_for_data> --dataset_type verifiable_answers
```

We currently support two datasets for RLVR:

- GMS8k: `openai/gsm8k`
- MATH: `DigitalLearningGmbH/MATH-lighteval`
