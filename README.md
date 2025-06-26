# Compose RL

**Compose RL** is a framework for Reinforcement Learning with Human Feedback (RLHF), designed to streamline the integration of various RLHF techniques. By leveraging the flexibility of MosaicML's [Composer](https://github.com/mosaicml/composer) and [LLM Foundry](https://github.com/mosaicml/llm-foundry/tree/main), Compose RL enables modular and composable experimentation, catering to researchers and practitioners exploring RLHF methodologies.

**Note**: This repository is currently in **alpha**. Expect frequent changes and potential instabilities. Use at your own risk!

## Key Features

- **Composable RLHF Training**:
  - Experiment with Policy Proximal Optimization (PPO), Direct Preference Optimization (DPO) and its variants, and reward model training.
  - Seamlessly integrate custom components or replace default implementations.

- **Powered by Databricks MosaicML**:

This repo utilizes [Composer](https://github.com/mosaicml/composer), an open-source deep learning training library optimized for scalability and usability. Composer simplifies the implementation of distributed training workflows on large-scale clusters, abstracting complexities like parallelism techniques, distributed data loading, and memory optimization. Additionally, this repo leverages [LLM Foundry](https://github.com/mosaicml/llm-foundry/tree/main), an open-source repository containing code for training large language models (LLMs) to help enable rapid experimentation with the latest techniques.

- **Modularity**:

  - Decoupled components for policy optimization, preference modeling, and reward model training.
  - Flexible configurations to adapt to a variety of RLHF workflows.

You'll find in this repo:
* `compose-rl` - source code for models, datasets, utilities
* `scripts` - scripts for generating data
* `yamls` - yamls for training runs

## Installation

Clone the repository and install the dependencies:

<!--pytest.mark.skip-->
```bash
git clone https://github.com/databricks/compose-rl.git
cd compose-rl
pip install -e .[gpu]
python3 -m spacy download en_core_web_sm
```

We *highly recommend* you use the [llm-foundry images](https://github.com/mosaicml/llm-foundry/?tab=readme-ov-file#mosaicml-docker-images) and install Compose RL and LLM Foundry and its dependencies on top of the images.

**Note**: when using the LLM Foundry images, please install LLM Foundry as it's not natively included with the images.


## Quickstart
Here is an end-to-end workflow of performing data preparation for training along with how to launch model training.

### Data preparation

Below is the set of commands to run to prepare datasets into the appropriate Mosaic Data Shard (MDS) format, which is either a pre-tokenized version of the data or the raw chat-message data, that we will use for training.

Below is the command to prepare preference data -- which can be used for reward model or offline RL (e.g. DPO) training:

<!--pytest.mark.skip-->
```bash
cd scripts
python data/unified_tokenize_dataset.py --dataset_name allenai/ultrafeedback_binarized_cleaned \
--local_dir pref_data \
--dataset_type preference \
--tokenizer_name meta-llama/Llama-3.1-8B-Instruct \
--split train_prefs
```

Below is the command to prepare single-turn message data -- which can be used for online RL (e.g. PPO) training:

<!--pytest.mark.skip-->
```bash
cd scripts
python data/messages_dataset_to_mds.py --dataset_name allenai/ultrafeedback_binarized_cleaned \
--local_dir ultrafeedback_summarization_data \
--split train_prefs
```

To further enable online RL with [verifiable rewards](https://arxiv.org/abs/2411.15124) you can use the following command to prepare the chat-message data and their corresponding verifiable answers:

<!--pytest.mark.skip-->
```bash
cd scripts
python data/messages_dataset_to_mds.py --dataset_path <hf_dataset_name> \
--local_dir verifiable_data \
--split train \
```

For RLVR, We currently support the following two HuggingFace datasets for verifiable rewards:

- GMS8k: `openai/gsm8k`
- MATH: `DigitalLearningGmbH/MATH-lighteval`

The data preparation scripts also supports additional arguments for specifying the subset of the HuggingFace dataset `--subset <str>` and max sequence length `--max_length <int>`

For custom datasets, you can create a custom preprocessing function in the `compose-rl/scripts/data/messages_preprocessing_utils.py` file, or you can preprocess your own dataset directly, save it locally as a .jsonl file, and then use the following command to convert it to the MDS format:
<!--pytest.mark.skip-->
```bash
cd scripts
python data/messages_dataset_to_mds.py --dataset_path <path_to_jsonl_file> \
--local_dir custom_dataset \
```


### Model training

Below are the scripts to launch training runs assuming you ran the data preparation scripts above. Additionally, these scripts assume that we are in the root directory where Compose RL and LLM Foundry were cloned. This is because we utilize [LLM Foundry's Registry System](https://github.com/mosaicml/llm-foundry/?tab=readme-ov-file#registry) in order to take advantage of existing features in LLM Foundry.

**Reward Model Training**

Below is the command to run reward model training:

<!--pytest.mark.skip-->
```bash
composer llm-foundry/scripts/train/train.py \
compose-rl/yamls/local_reward.yaml \
train_loader.dataset.local=/compose-rl/scripts/pref_data/ \
train_loader.dataset.split=train_prefs
```

**DPO Training**

Below is the command to run for DPO training (along with its variants):

<!--pytest.mark.skip-->
```bash
composer llm-foundry/scripts/train/train.py \
compose-rl/yamls/local_dpo.yaml \
train_loader.dataset.local=/compose-rl/scripts/pref_data/ \
train_loader.dataset.split=train_prefs
```

For DPO we support other variants of DPO including: [Reward Aware Preference Optimization (RPO)](https://arxiv.org/pdf/2406.11704v1), [REgression to RElative REward Based RL (REBEL)](https://arxiv.org/pdf/2404.16767), [Identity-PO (IPO)](https://arxiv.org/abs/2310.12036), and [Kahneman-Tversky Optimization (KTO)](https://arxiv.org/abs/2402.01306).

**PPO Training**

Below is the command to run Online PPO training:

<!--pytest.mark.skip-->
```bash
composer llm-foundry/scripts/train/train.py \
compose-rl/yamls/local_ppo.yaml \
train_loader.dataset.local=/compose-rl/scripts/ultrafeedback_summarization_data/ \
train_loader.dataset.split=train_prefs
```

## Helpful code pointers

**Adding new data processing**
In the [`unified_tokenize_dataset.py`](https://github.com/databricks/compose-rl/blob/d59a63baecdadc7444b85edc39101c6237441bd6/scripts/data/unified_tokenize_dataset.py)  script, we can add new capabilities to process data by modifying the [`__iter__`](https://github.com/databricks/compose-rl/blob/d59a63baecdadc7444b85edc39101c6237441bd6/scripts/data/unified_tokenize_dataset.py#L51)function. Typically all datasets are pre-tokenized before model training.

**Creating new reward models**
In order to modify the training loss for reward models we need to define a new `Enum` [here](https://github.com/databricks/compose-rl/blob/d59a63baecdadc7444b85edc39101c6237441bd6/compose_rl/reward_learning/model_methods.py#L30) and update the `pairwise_loss` function. Modifying the loss will result in a different `loss_type` in the model yaml [here](https://github.com/databricks/compose-rl/blob/d59a63baecdadc7444b85edc39101c6237441bd6/yamls/local_reward.yaml#L16)

Additionally, in order to modify the reward model architecture for huggingface models we need to update `AutoModelForCausalLMWithRM` [here](https://github.com/databricks/compose-rl/blob/d59a63baecdadc7444b85edc39101c6237441bd6/compose_rl/reward_learning/hf_utils.py#L127). Additionally, any model config updates, we can update the config [here](https://github.com/databricks/compose-rl/blob/d59a63baecdadc7444b85edc39101c6237441bd6/compose_rl/reward_learning/hf_utils.py#L91). Config updates will apply to the model yaml [here](https://github.com/databricks/compose-rl/blob/d59a63baecdadc7444b85edc39101c6237441bd6/yamls/local_reward.yaml#L11)

**Creating new Offline RL Variants**
In order to add new offline RL based algorithms, we need to add a new `Enum` to `DPOEnum` [here](https://github.com/databricks/compose-rl/blob/d59a63baecdadc7444b85edc39101c6237441bd6/compose_rl/dpo/model_methods.py#L29) and then we can add the new loss function in the `dpo_loss` method. The loss function field will be updated in the model section of the DPO yaml [here](https://github.com/databricks/compose-rl/blob/d59a63baecdadc7444b85edc39101c6237441bd6/yamls/local_dpo.yaml#L6)

**Creating new Online RL Variants**
In order to create new online RL based algorithms, we need to add a new `forward` and `loss` function into `ppo/modeling_utils.py`. From here we need to define a new model in `ppo/model.py`.

**A high level overview of LLM Foundry Plugins**
LLM Foundry plugins allows us to take advantage of many of the functions within LLM foundry, while augmenting the code with other models and methods. Plugins requires us to define registry entrypoints into LLM foundry which is done in the `pyproject.toml` file in this repo. See the commented code around `entry points` in the file [here](https://github.com/databricks/compose-rl/blob/d59a63baecdadc7444b85edc39101c6237441bd6/pyproject.toml#L41), where we define various entrypoints for models, dataloaders, callbacks, and metrics. For more details on plugins and its registry system see [here](https://github.com/mosaicml/llm-foundry/?tab=readme-ov-file#registry).

**How to write HuggingFace checkpoints**
Since we use LLM Foundry's plugins, we are able to export our models as HuggingFace Models and Checkpoints. We are able to do this by using the `hf_checkpointer` [callback from LLM Foundry](https://github.com/mosaicml/llm-foundry/blob/a27c720058bcdf08bfbd51a1e76b17097012fe26/llmfoundry/callbacks/hf_checkpointer.py#L245), which can be defined in a yaml as the following:
```
callbacks:
    hf_checkpointer:
	    # Specify an appropriate save interval (e.g. 1 epoch)
        save_interval: 1ep
        # Specify the appropriave save path (this can be a local checkpoint, s3, oci, or any other
        # saving mechanism supported by Composer.)
        save_folder:
        # Specify the precision to save the checkpoint.
        precision: bfloat16
```
