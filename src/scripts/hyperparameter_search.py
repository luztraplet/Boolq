import psutil
import ray
import wandb
from datasets import load_dataset, load_metric, ClassLabel
from ray import tune
from ray.tune import Stopper
from ray.tune.suggest.hyperopt import HyperOptSearch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# set up Wandb
wandb.login()
wandb.init(project="Hyperopt", entity="luztraplet")

# patch Raytune bug
ray._private.utils.get_system_memory = lambda: psutil.virtual_memory().total

model_checkpoint = "roberta-base"

# load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
dataset = load_dataset('boolq')


# tokenize dataset
def preprocess(batch):
    return tokenizer(batch['question'], batch['passage'], truncation='longest_first', padding='max_length',
                     max_length=256)


# format dataset
def rename_remove_cast_split_dataset(encoded_dataset):
    encoded_dataset = encoded_dataset.rename_column("answer", "labels")

    encoded_dataset = encoded_dataset.remove_columns(["question", "passage"])

    new_features = encoded_dataset['train'].features.copy()
    new_features["labels"] = ClassLabel(names=['0', '1'])
    encoded_dataset = encoded_dataset.cast(new_features)
    train_dataset = encoded_dataset["train"].shuffle(seed=42)
    eval_dataset = encoded_dataset["validation"]
    return train_dataset, eval_dataset


encoded_dataset = dataset.map(preprocess, batched=True)
train_dataset, eval_dataset = rename_remove_cast_split_dataset(encoded_dataset)


# set up model initialisation
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, return_dict=True)


# load metric accuracy
metric = load_metric('accuracy')


# define usage of metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)


# create custom stopper that terminates runs that don't reach the threshold
class TrialStopper(Stopper):
    def __init__(
            self,
            metric_threshold=0.64
    ):
        self._metric_threshold = metric_threshold

    def __call__(self, trial_id, result):
        metric_result = result.get("eval_accuracy")

        if metric_result < self._metric_threshold:
            return True

        return False

    def stop_all(self):
        return False


# set up training arguments
training_args = TrainingArguments(
    "Roberta Base Hyperopt",
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="wandb",
    disable_tqdm=True
)

# set up trainer
trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    model_init=model_init,
    compute_metrics=compute_metrics,
)

# set up search space
hp_space = {
    "learning_rate": tune.loguniform(1e-4, 1e-6),
    "per_device_train_batch_size": tune.choice([8]),
    "gradient_accumulation_steps": tune.choice([1, 2, 4, 8]),
    "lr_scheduler_type": tune.choice(["linear", "cosine", "constant_with_warmup"]),
    "warmup_steps": tune.qrandint(0, 200, 25),
    "num_train_epochs": 3,
}

# do hyperopt
trainer.hyperparameter_search(
    hp_space=lambda _: hp_space,
    direction="maximize",
    backend="ray",
    n_trials=20,
    search_alg=HyperOptSearch(metric="objective", mode="max"),
    stop=TrialStopper())
