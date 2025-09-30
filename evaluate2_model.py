import pandas as pd
import evaluate
import matplotlib.pyplot as plt

# Încarcă fișierul cu predicții
df = pd.read_csv("subtaskA1_predictions.csv")

# Pregătește datele
preds = df["prediction"].astype(str).tolist()
refs = df["target_text"].astype(str).tolist()

# Încarcă metricile
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

# Calculează scorurile
rouge_result = rouge.compute(predictions=preds, references=refs)
bleu_result = bleu.compute(predictions=preds, references=[[r] for r in refs])
meteor_result = meteor.compute(predictions=preds, references=refs)

# Structurare scoruri într-un dict
scores = {
    "ROUGE-1": rouge_result["rouge1"],
    "ROUGE-2": rouge_result["rouge2"],
    "ROUGE-L": rouge_result["rougeL"],
    "BLEU": bleu_result["bleu"],
    "METEOR": meteor_result["meteor"]
}

# Creează DataFrame
df_scores = pd.DataFrame.from_dict(scores, orient="index", columns=["Score"])
df_scores = df_scores.round(4)

# ✅ Afișează scorurile
print("📊 Evaluation Scores:\n")
print(df_scores)

# ✅ Grafic bară
plt.figure(figsize=(8, 4))
df_scores["Score"].plot(kind="bar", rot=0)
plt.ylim(0, 1)
plt.title("Evaluation Metrics for T5 Predictions")
plt.ylabel("Score")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
