import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Forțează rularea pe CPU
device = torch.device("cpu")

# Încarcă modelul și tokenizerul
model_dir = "./t5_note2dialogue_model_TASK_C"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)
model.to(device)
model.eval()

# Încarcă setul de test
test_df = pd.read_csv("testC_preprocesat.csv")

# Generează predicții
predictions = []
for text in test_df["input_text"]:
    inputs = tokenizer(
        text, return_tensors="pt",
        padding="max_length", truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    predictions.append(pred)

# Salvează rezultatele într-un fișier CSV
test_df["generated_text"] = predictions
test_df[["input_text", "target_text", "generated_text"]].to_csv("taskC_predictions.csv", index=False)

print("✅ Predicții generate și salvate în: taskC_predictions.csv")
