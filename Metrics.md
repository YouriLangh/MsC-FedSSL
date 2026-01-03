
# ✅ **Metric Interpretations (High vs. Low) in Imbalanced Settings**

## **1. Accuracy**

**What it measures:**  
Overall proportion of correct predictions.

**High accuracy means:**

- The model gets many predictions right **overall**, but **this may be dominated by the majority class**.
    
- Not reliable in imbalanced datasets.
    

**Low accuracy means:**

- The model is failing broadly on both majority and minority classes.
    

**In imbalanced datasets:**

- You can have **high accuracy but terrible minority-class recall**.
    
- Treat accuracy as a weak metric.
    

---

## ✅ **2. Confusion Matrix**

**What it measures:**  
A full breakdown of predictions vs. ground truth for every class.

**Interpretation in imbalance:**

- The **only metric that exposes the failure modes** hidden by accuracy.
    
- Look for:
    
    - Minority class rows: **low true positives** + **many false negatives** → the model cannot detect rare classes.
        
    - Majority class: usually inflated true positives.
        

**High-quality confusion matrix:**

- Balanced diagonal entries (true positives).
    

**Low-quality confusion matrix:**

- Sparse diagonal for minority classes.
    

---

## ✅ **3. F1 Macro**

**Definition:**  
Unweighted average of F1 for each class.

**High F1_macro means:**

- The model performs **similarly well across all classes**, including minorities.
    
- Indicates **class-balanced performance**.
    

**Low F1_macro means:**

- Model fails on minority classes (most common case).
    
- Highly sensitive to **class imbalance**.
    

**In imbalanced datasets:**

- Usually low unless you explicitly address imbalance (loss weighting, data balancing, etc.).
    

---

## ✅ **4. F1 Weighted**

**Definition:**  
Class-wise F1 weighted by class frequencies.

**High F1_weighted means:**

- Strong performance on the majority class.
    
- Could still be failing on minority classes.
    

**Low F1_weighted means:**

- The model is failing broadly across classes.
    

**In imbalanced datasets:**

- Much **closer to accuracy** than to macro-F1.
    
- Can be **misleading**, because minority classes have little impact.
    

---

## ✅ **5. Recall Macro**

Recall_macro = mean recall across classes (unweighted).

**High recall_macro means:**

- The model detects **all classes**, including rare ones.
    
- Excellent indicator for imbalance-sensitive problems.
    

**Low recall_macro means:**

- The model **misses minority classes** frequently (high false negatives).
    
- Typical for models trained on imbalanced data.
    

**In imbalanced datasets:**

- One of the most important metrics (especially if missing minority classes is costly).
    

---

## ✅ **6. AUROC (Macro, OVR)**

**Definition:**  
Average AUROC over all one-vs-rest binary problems, treating each class equally.

**High AUROC (≈ 0.9–1.0):**

- Strong ranking ability across all classes.
    
- The model assigns **higher probabilities to correct classes** even before thresholding.
    

**Medium AUROC (≈ 0.7–0.85):**

- Reasonable class separability but imperfect.
    

**Low AUROC (< 0.7):**

- Poor class separability.
    
- Model probabilities are not distinguishing classes, especially minorities.
    

**In imbalanced datasets:**

- AUROC_macro is **threshold-independent**, so it’s more robust than F1/recall.
    
- Still, minority classes with poor separability will pull the score down.
    

---

# ✅ **7. Error Rate**

Simply: 1 − accuracy.

Same limitations as accuracy:

- **Low error rate** can still hide failure on minor classes.
    

---

# ✅ Summary Table (Imbalanced Dataset Context)

|Metric|High Value Means|Low Value Means|Notes|
|---|---|---|---|
|**Accuracy**|Good overall performance (mostly majority class)|Bad overall performance|Misleading under imbalance|
|**Confusion Matrix**|Strong diagonal, all classes represented|Missing diagonal for minority classes|Best for inspection|
|**F1_macro**|Good balanced performance|Poor minority-class performance|Key metric for imbalance|
|**F1_weighted**|Good majority-class performance|General poor performance|Masked by class imbalance|
|**Recall_macro**|Good detection of all classes|High FN on minorities|Important for rare classes|
|**AUROC_macro**|Good separability for each class|Poor probability quality|Threshold-independent|

---

# ✅ If you want, I can also:

- Evaluate your current metric outputs and tell you which classes fail
    
- Suggest imbalance strategies (loss weighting, oversampling, focal loss)
    
- Simulate how these metrics change under extreme imbalance
    

Just send your metric printouts.