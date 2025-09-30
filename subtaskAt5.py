import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import evaluate
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


class SummaryDataset(Dataset):
    def __init__(self, df, tokenizer, source_col, target_col, max_length=512):
        self.inputs = tokenizer(
            list("summarize: " + df[source_col]),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.targets = tokenizer(
            list(df[target_col]),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        return {
            "input_ids":      self.inputs.input_ids[idx],
            "attention_mask": self.inputs.attention_mask[idx],
            "labels":         self.targets.input_ids[idx],
        }

    def __len__(self):
        return len(self.inputs.input_ids)


# ─── Setup ────────────────────────────────────────────────────────────────

# Load tokenizer & model (legacy behavior)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Resize embeddings in case HF added any special tokens
model.resize_token_embeddings(len(tokenizer))
model.to("cuda")

# Load full CSVs
train_df = pd.read_csv("../date/preprocessed_data/train_cleaned.csv")
val_df   = pd.read_csv("../date/preprocessed_data/validation_cleaned.csv")
test_df  = pd.read_csv("../date/preprocessed_data/test1_cleaned.csv")

# Create datasets over the full DataFrames
train_ds = SummaryDataset(train_df, tokenizer, "dialogue", "section_text")
val_ds   = SummaryDataset(val_df,   tokenizer, "dialogue", "section_text")
test_ds  = SummaryDataset(test_df,  tokenizer, "dialogue", "section_text")

# Data-collator to pad & mask labels correctly
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100
)

# Load metrics
rouge = evaluate.load("rouge")
bleu  = evaluate.load("bleu")


def manual_evaluate(dataset, split_name, batch_size=4):
    model.eval()
    all_preds, all_refs = [], []

    # 1) Generate
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            input_ids = torch.stack([b["input_ids"] for b in batch]).to("cuda")
            attention_mask = torch.stack([b["attention_mask"] for b in batch]).to("cuda")
            preds = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
            all_preds.extend(preds.cpu().tolist())

    decoded_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]

    # 2) Decode references
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        labels = [b["labels"] for b in batch]
        cleaned = [torch.where(l == -100, tokenizer.pad_token_id, l) for l in labels]
        decoded = tokenizer.batch_decode(cleaned, skip_special_tokens=True)
        all_refs.extend([r.strip() for r in decoded])

    # Early bail if empty
    if not decoded_preds or not all_refs:
        return decoded_preds, all_refs, {"rouge1": 0.0, "rougeL": 0.0, "bleu": 0.0}

    # 3) ROUGE
    rouge_res = rouge.compute(predictions=decoded_preds, references=all_refs, use_stemmer=True)

    # 4) BLEU with smoothing + empty-ref filter
    filtered = [(p, r) for p, r in zip(decoded_preds, all_refs) if p.strip() and r.strip()]
    if not filtered:
        print("BLEU computation skipped due to all empty predictions/references.")
        bleu_score = 0.0
    else:
        preds_filt, refs_filt = zip(*filtered)
        bleu_res = bleu.compute(
            predictions=list(preds_filt),
            references=list(refs_filt),
            smooth=True
        )
        bleu_score = bleu_res["bleu"]

    print(f"\n📊 {split_name} → ROUGE-1: {rouge_res['rouge1']:.4f}, ROUGE-L: {rouge_res['rougeL']:.4f}, BLEU: {bleu_score:.4f}")

    return decoded_preds, all_refs, {
        "rouge1": rouge_res["rouge1"],
        "rougeL": rouge_res["rougeL"],
        "bleu":   bleu_score,
    }


# ─── Training ─────────────────────────────────────────────────────────────

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    evaluation_strategy="epoch",    # or "epoch" to auto-evaluate
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,        # optional: enables trainer.evaluate()
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.evaluate()


# ─── Evaluate & Save ─────────────────────────────────────────────────────

val_preds, val_refs, val_metrics = manual_evaluate(val_ds,  "Validation")
test_preds, test_refs, test_metrics = manual_evaluate(test_ds, "Test")

# Save raw predictions over *all* test rows
pd.DataFrame({
    "dialogue":   test_df["dialogue"],
    "reference":  test_refs,
    "prediction": test_preds
}).to_csv("results/SubtaskA_T5FlanBase_predictions_fullDataset.csv", index=False)

# Save metrics
pd.DataFrame([
    {"split": "Validation", **val_metrics},
    {"split": "Test",       **test_metrics},
]).to_csv("results/metrics_subtaskA_T5FlanBase_fullDataset.csv", index=False)
