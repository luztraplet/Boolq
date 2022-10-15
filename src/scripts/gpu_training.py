import wandb
from datasets import load_dataset, load_metric, ClassLabel
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

# set up Wandb
wandb.login()
wandb.init(project="Boolq", entity="luztraplet")

model_checkpoint = "roberta-base"

# load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
dataset = load_dataset('boolq')


# tokenize dataset
def preprocess_dataset(batch):
    return tokenizer(batch['question'], batch['passage'], truncation='longest_first', padding='max_length',
                     max_length=256)


# format dataset
def rename_remove_cast_split_dataset(encoded_dataset):
    encoded_dataset = encoded_dataset.rename_column("answer", "labels")

    encoded_dataset = encoded_dataset.remove_columns(["question", "passage"])

    new_features = encoded_dataset['train'].features.copy()
    new_features["labels"] = ClassLabel(names=['0', '1'])
    encoded_dataset = encoded_dataset.cast(new_features)
    encoded_dataset.set_format("torch")

    train_dataset = encoded_dataset["train"].shuffle(seed=42)
    eval_dataset = encoded_dataset["validation"]
    return train_dataset, eval_dataset


encoded_dataset = dataset.map(preprocess_dataset, batched=True)
train_dataset, eval_dataset = rename_remove_cast_split_dataset(encoded_dataset)

# load model and optimizer
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
optimizer = AdamW(model.parameters())

# load metric accuracy
metric = load_metric('accuracy')


# define usage of metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)


# set up training arguments
training_arguments = TrainingArguments(
    "Roberta Base finetuned on boolq",
    evaluation_strategy="steps",
    save_strategy="no",
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    lr_scheduler_type="linear",
    eval_steps=50,
    warmup_steps=50,
    logging_strategy="steps",
    logging_steps=50,
    gradient_accumulation_steps=1,
    report_to="wandb",
)

# set up trainer
trainer = Trainer(
    model,
    training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# do training
trainer.train()

wandb.finish()
