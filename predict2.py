# generate_predictions.py
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Încarcă modelul antrenat
model = T5ForConditionalGeneration.from_pretrained("./t5_dialogue2note_model_clean")
tokenizer = T5Tokenizer.from_pretrained("./t5_dialogue2note_model_clean")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Încarcă datele de test
df = pd.read_csv("testset1_preprocesat.csv")

# Generează predicții
predictions = []
model.eval()
for input_text in df["input_text"]:
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_length=128)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(pred)

df["prediction"] = predictions
df.to_csv("subtaskA1_predictions.csv", index=False)
print("✅ Predicții salvate în subtaskA1_predictions.csv")
