import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)

# Încarcă fișiere preprocesate
train_df = pd.read_csv("trainC_preprocesat.csv")
val_df = pd.read_csv("valC_preprocesat.csv")

# Conversie la HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Inițializare model și tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Tokenizare
def tokenize(example):
    inputs = tokenizer(
        example["input_text"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    targets = tokenizer(
        example["target_text"],
        max_length=256,
        padding="max_length",
        truncation=True
    )
    inputs["labels"] = [
        label if label != tokenizer.pad_token_id else -100
        for label in targets["input_ids"]
    ]
    return inputs

train_tok = train_dataset.map(tokenize, batched=False)
val_tok = val_dataset.map(tokenize, batched=False)

# Argumente pentru antrenare
args = TrainingArguments(
    output_dir="./t5_note2dialogue_model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=1,
    logging_dir="./logs",
    report_to="none",
    fp16=torch.cuda.is_available()
)

# Setup Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)

# Rulează antrenarea
trainer.train()

# Salvează modelul antrenat
model.save_pretrained("./t5_note2dialogue_model_TASK_C")
tokenizer.save_pretrained("./t5_note2dialogue_model_TASK_C")
