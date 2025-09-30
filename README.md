# Dialogue2Note & Note2Dialogue with T5

Acest repository conține cod pentru antrenarea, generarea de predicții și evaluarea modelelor **T5** pe două sarcini principale:

- **Subtask A (Dialogue2Note)** – generarea notițelor clinice structurate pe baza dialogurilor medic–pacient.  
- **Task C (Note2Dialogue)** – generarea dialogurilor pe baza notițelor medicale.  

## 📂 Structura repository-ului

- `subtaskAt5.py` – antrenare și evaluare pentru **Subtask A** (T5-Flan).  
- `TASKC_train.py` – antrenare pentru **Task C** (T5-Base).  
- `predict.py`, `predict2.py` – generare predicții pentru Subtask A (testset-uri diferite).  
- `generate_predictC.py`, `generate_predict_tuned.py` – generare predicții pentru Task C (model standard și model tunat).  
- `evaluate_model.py`, `evaluate2_model.py` – evaluare Subtask A (ROUGE, BLEU, METEOR).  
- `evaluate_taskC.py`, `evaluate_taskC_tuned.py` – evaluare Task C (ROUGE, BLEU, METEOR).  

## ⚙️ Cerințe

Se instalează dependențele necesare:

```bash
pip install torch transformers datasets evaluate pandas matplotlib seaborn

# Subtask A – Dialogue2Note
python subtaskAt5.py

# Subtask C – Note2Dialogue
python TASKC_train.py


Modelele antrenate vor fi salvate în directoarele:

./t5_dialogue2note_model_clean

./t5_note2dialogue_model_TASK_C

## 📝 Generare predicții

# Subtask A
python predict.py
python predict2.py

# Subtask C
python generate_predictC.py
python generate_predict_tuned.py


## 📊 Evaluare

# Subtask A
python evaluate_model.py
python evaluate2_model.py

# Subtask C
python evaluate_taskC.py
python evaluate_taskC_tuned.py

