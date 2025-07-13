#utils 

### ✅ **1. Loss and Accuracy Curves per Client (Training)**

**Plot:**

- **Training loss vs. local epoch**
    
- **Training accuracy vs. local epoch**
    

**Why:**  
Shows how fast and how well each client converges individually. Use it to spot:

- Clients learning slower/faster than others
    
- Overfitting (flat or increasing loss while accuracy plateaus)
    

**How:**  
Log `loss` and `accuracy` in `client.train_self()` and store per-epoch results in:

```python
client.training_history = {
    'loss': [...],  # One per local epoch
    'accuracy': [...]
}
```

Then plot:

```python
for i, client in enumerate(clients):
    plt.plot(client.training_history['loss'], label=f'Client {i}')
plt.title('Training Loss per Client')
```

---

### ✅ **2. Global Accuracy Over Rounds**

**Plot:**

- **Global (server) accuracy vs. communication round**
    

**Why:**  
This measures the **overall effect** of collaboration. Useful for comparing:

- FedAvg vs. baseline
    
- Convergence rate
    
- How pseudo-labeling (FixMatch) affects generalization
    

---

### ✅ **3. Client Weight Drift (Optional)**

**Plot:**

- **L2 distance** between each client's weights and the global average
    

**Why:**  
This visualizes **how different clients are** in their learning. In heterogeneous settings or with non-IID data, some clients might drift more than others.

---

### ✅ **4. Test Accuracy per Client**

If you give each client a **local test set** (or split the main test set across clients), you can compare:

- How well each client performs on their own test distribution
    
- Generalization across clients
    

---

### ✅ **5. Confusion Matrices (Optional)**

**Plot:** Confusion matrix for each client or the server model

**Why:**  
Helps uncover _what kinds of mistakes_ are common, or if a client consistently misclassifies certain classes.

---

### ✅ **6. Pretraining vs. Fine-tuning Plots**

If clients do **FixMatch pretraining**, you can split plots:

- Pretraining loss vs. epochs
    
- Fine-tuning loss vs. epochs
    

---



Good source:
https://www.evidentlyai.com/classification-metrics/explain-roc-curve



BONUS:
We are considering between AUROC, AUC, F1 (wweighted & macro), recall, precision, accuracy


### ✅ **1. Precision**: _How many of the predicted positives were correct?_

**Formula**:

Precision=True PositivesTrue Positives + False Positives \text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}

**Interpretation**:

- Out of everything the model predicted as **positive**, how many were actually correct?
    
- High precision = **few false positives**.
    

**When important**:

- When **false alarms are costly** (e.g., cancer diagnosis, fraud detection).
    
- You want to **trust your model's positive predictions**.
    

---

### ✅ **2. Recall (Sensitivity)**: _How many of the actual positives were found?_

**Formula**:

Recall=True PositivesTrue Positives + False Negatives\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}

**Interpretation**:

- Out of all real **positives**, how many did the model catch?
    
- High recall = **few false negatives**.
    

**When important**:

- When **missing a positive is very costly** (e.g., failing to detect a disease).
    
- In **semi-supervised learning**, recall shows how well you’re generalizing to unseen examples.
    

---

### ✅ **3. F1 Score**: _Balance between precision and recall_
- If **macro is low but weighted is high**, your model struggles on rare classes.
    
- If both are high, your model is performing well and fairly across all classes.
    
- If weighted is low, your model is failing even on frequent classes — a bigger concern.

**Formula**:

F1=2×Precision×RecallPrecision + Recall\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}}

**Interpretation**:

- Harmonic mean of precision and recall.
    
- **Penalizes extreme imbalances** (e.g., high precision but low recall).
    
- It’s a **single score** that captures both correctness and completeness.
    

**When important**:

- In **imbalanced datasets**, where accuracy can be misleading.
    
- To get an overall sense of performance when **both precision and recall matter**.
    

---

### ✅ **4. AUROC (Area Under ROC Curve)**: _How well does the model rank its predictions?_

**ROC Curve**: plots **True Positive Rate (Recall)** vs **False Positive Rate** at various thresholds.  
**AUC** = area under this curve (ranges from 0.5 to 1.0)

**Interpretation**:

- Measures **confidence**: how well the model ranks positive vs negative examples.
    
- Higher AUROC = model is **better at separating classes**, even if threshold isn’t fixed.
    

**When important**:

- When you care about how confident the model is, not just whether it's right.
    
- In **semi-supervised** or **federated** setups where models improve slowly and confidence is key.
    

---

### 📊 Example:

Imagine you're building a model to detect "Digit 5" from MNIST:

|Metric|Value|What it tells you|
|---|---|---|
|Accuracy|98%|Great, but could be due to class imbalance (many digits ≠5).|
|Precision|70%|70% of predicted 5s were correct. Some false alarms.|
|Recall|30%|Caught only 30% of actual 5s. Missed many.|
|F1 Score|42%|Tells you performance is actually poor — bad balance.|
|AUROC|0.85|Model ranks 5s higher than others most of the time. It’s learning!|

---

### 🧠 Summary Table

|Metric|Focuses On|When Useful|
|---|---|---|
|Precision|Avoiding false positives|You want high _purity_ of positive predictions|
|Recall|Avoiding false negatives|You want to _catch as many positives_ as possible|
|F1 Score|Balancing both|General-purpose evaluation on imbalanced data|
|AUROC|Confidence and ranking|Measuring how well model **separates** classes, especially in early stages|

---

These metrics together give you a **multi-dimensional view**:

- Precision & recall help you **understand trade-offs**.
    
- F1 gives a **single balanced measure**.
    
- AUROC shows **how confident and consistent** your model is, even if not perfect.
    

Let me know if you want visual examples or want to compare how these change across federated rounds or clients.