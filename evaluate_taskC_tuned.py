import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns

# Încarcă fișierul de predicții
df = pd.read_csv("taskC_predictions_tuned.csv")

# Încarcă metricile
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

# Calculează scorurile
rouge_result = rouge.compute(predictions=df["generated_text"], references=df["target_text"])
bleu_result = bleu.compute(predictions=df["generated_text"], references=df["target_text"])
meteor_result = meteor.compute(predictions=df["generated_text"], references=df["target_text"])

# Dicționar cu toate scorurile
all_scores = {
    "ROUGE-1": rouge_result["rouge1"],
    "ROUGE-2": rouge_result["rouge2"],
    "ROUGE-L": rouge_result["rougeL"],
    "BLEU": bleu_result["bleu"],
    "METEOR": meteor_result["meteor"]
}

# Afișare într-un DataFrame
scores_df = pd.DataFrame.from_dict(all_scores, orient="index", columns=["Score"])
print(scores_df)

# Plot grafic
plt.figure(figsize=(8, 5))
sns.barplot(x=scores_df.index, y=scores_df["Score"])
plt.title("Evaluation Metrics for Note2Dialogue")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
