import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Optional

import horovod.tensorflow as hvd

import datasets
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split

import transformers
from transformers import (
    CONFIG_MAPPING,
    CONFIG_NAME,
    TF2_WEIGHTS_NAME,
    TF_MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TFAutoModelForMaskedLM,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
)
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/tensorflow/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(TF_MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


class SavePretrainedCallback(tf.keras.callbacks.Callback):

    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)


def on_state_reset(model, lr):
    tf.keras.backend.set_value(model.optimizer.lr, lr * hvd.size())


@hvd.elastic.run
def train_op(state):
    model = state.model
    history = model.fit(
        state.tf_train_dataset,
        validation_data=state.tf_eval_dataset,
        epochs=int(state.training_args.num_train_epochs),
        steps_per_epoch=len(state.train_dataset) // state.training_args.per_device_train_batch_size,
        callbacks=state.callbacks,
        verbose=1 if hvd.rank() == 0 else 0
    )
    return history


def main():

    hvd.init()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):

        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.dataset_name is None and data_args.train_file is None and data_args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if data_args.train_file is not None:
            extension = data_args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if data_args.validation_file is not None:
            extension = data_args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if training_args.output_dir is not None:
        training_args.output_dir = Path(training_args.output_dir)
        os.makedirs(training_args.output_dir, exist_ok=True)

    if isinstance(training_args.strategy, tf.distribute.TPUStrategy) and not data_args.pad_to_max_length:
        logger.warning("We are training on TPU - forcing pad_to_max_length")
        data_args.pad_to_max_length = True

    checkpoint = None
    if len(os.listdir(training_args.output_dir)) > 0 and not training_args.overwrite_output_dir:
        config_path = training_args.output_dir / CONFIG_NAME
        weights_path = training_args.output_dir / TF2_WEIGHTS_NAME
        if config_path.is_file() and weights_path.is_file():
            checkpoint = training_args.output_dir
            logger.warning(
                f"Checkpoint detected, resuming training from checkpoint in {training_args.output_dir}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        else:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to continue regardless."
            )

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    if training_args.seed is not None:
        set_seed(training_args.seed)

    if data_args.dataset_name is not None:

        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if checkpoint is not None:
        config = AutoConfig.from_pretrained(checkpoint)
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can reduce that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset line_by_line",
        )
    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )

    train_dataset = tokenized_datasets["train"]

    if data_args.validation_file is not None:
        eval_dataset = tokenized_datasets["validation"]
    else:
        logger.info(
            f"Validation file not found: using {data_args.validation_split_percentage}% of the dataset as validation as provided in data_args"
        )
        train_indices, val_indices = train_test_split(
            list(range(len(train_dataset))), test_size=data_args.validation_split_percentage / 100
        )

        eval_dataset = train_dataset.select(val_indices)
        train_dataset = train_dataset.select(train_indices)

    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
        start = hvd.rank() * len(train_dataset) // hvd.size()
        end = (hvd.rank() + 1) * len(train_dataset) // hvd.size()
        train_dataset = train_dataset[start : end]

    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
        start = hvd.rank() * len(eval_dataset) // hvd.size()
        end = (hvd.rank() + 1) * len(eval_dataset) // hvd.size()
        eval_dataset = eval_dataset[start : end]

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    with training_args.strategy.scope():
        if checkpoint is not None:
            model = TFAutoModelForMaskedLM.from_pretrained(checkpoint, config=config)
        elif model_args.model_name_or_path:
            model = TFAutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, config=config)
        else:
            logger.info("Training new model from scratch")
            model = TFAutoModelForMaskedLM.from_config(config)

        model.resize_token_embeddings(len(tokenizer))
        num_replicas = training_args.strategy.num_replicas_in_sync
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=data_args.mlm_probability, return_tensors="tf"
        )
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        tf_train_dataset = train_dataset.to_tf_dataset(
            columns=[col for col in train_dataset.features if col != "special_tokens_mask"] + ["labels"],
            shuffle=True,
            batch_size=num_replicas * training_args.per_device_train_batch_size,
            collate_fn=data_collator,
            drop_remainder=True,
        ).with_options(options)

        tf_eval_dataset = eval_dataset.to_tf_dataset(
            columns=[col for col in eval_dataset.features if col != "special_tokens_mask"] + ["labels"],
            shuffle=False,
            batch_size=num_replicas * training_args.per_device_train_batch_size,
            collate_fn=data_collator,
            drop_remainder=True,
        ).with_options(options)

        batches_per_epoch = len(train_dataset) // (num_replicas * training_args.per_device_train_batch_size)
        optimizer, lr_schedule = create_optimizer(
            init_lr=training_args.learning_rate,
            num_train_steps=int(training_args.num_train_epochs * batches_per_epoch),
            num_warmup_steps=training_args.warmup_steps,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
            weight_decay_rate=training_args.weight_decay,
        )

        optimizer = hvd.DistributedOptimizer(optimizer)

        model.compile(optimizer=optimizer)

        state = hvd.elastic.KerasState(model, batch=100, epoch=0, tf_eval_dataset=tf_eval_dataset, tf_train_dataset=tf_train_dataset, training_args=training_args)
        callbacks = [
            hvd.elastic.CommitStateCallback(state),
            hvd.elastic.UpdateBatchStateCallback(state),
            hvd.elastic.UpdateEpochStateCallback(state),
            SavePretrainedCallback(output_dir=training_args.output_dir)
        ]
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size = {training_args.per_device_train_batch_size * num_replicas}")

        history = train_op(state)
        try:
            train_perplexity = math.exp(history.history["loss"][-1])
        except OverflowError:
            train_perplexity = math.inf
        try:
            validation_perplexity = math.exp(history.history["val_loss"][-1])
        except OverflowError:
            validation_perplexity = math.inf
        logger.warning(f"  Final train loss: {history.history['loss'][-1]:.3f}")
        logger.warning(f"  Final train perplexity: {train_perplexity:.3f}")
        logger.warning(f"  Final validation loss: {history.history['val_loss'][-1]:.3f}")
        logger.warning(f"  Final validation perplexity: {validation_perplexity:.3f}")

        if training_args.output_dir is not None:
            model.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        model.push_to_hub()


if __name__ == "__main__":
    main()
