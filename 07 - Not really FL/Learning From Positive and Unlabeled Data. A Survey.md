
### 🔹 What is PU Learning?

PU learning = **Positive + Unlabeled learning**.  
You only know which examples are **positive** (e.g., people with a disease), but the rest are **unlabeled** (they might be healthy or sick — you don’t know).

- Unlike normal machine learning, you **don’t have clear negative examples**.
    

---

### 🔹 Why is this useful?

This happens in **real life a lot**:

- Medical data → you know who has a disease, but not everyone who _doesn’t_.
    
- Ads/recommendations → you know what people clicked, but not everything they ignored.
    
- Knowledge bases → only facts are recorded, not everything false.
    

So PU learning helps build classifiers **even when negatives are missing**.

---

### 🔹 The Main Challenges

1. **How are positives chosen to be labeled?**
    
    - Sometimes they’re labeled completely at random (**SCAR**).
        
    - Sometimes labeling depends on hidden factors (e.g., richer patients more likely to get diagnosed — **SAR**).
        
    - Sometimes positives that look like negatives are harder to label (**PGPU**).
        
2. **How do we know how many positives exist in total (class prior)?**
    
    - You need a way to **estimate the overall proportion of positives** in the data, even though many are hidden.
        
3. **How do we train without negatives?**
    
    - Treat unlabeled as “probably negative but noisy.”
        
    - Try to find **reliable negatives** (examples far away from positives).
        
    - Or adjust learning methods to **work directly with positive + unlabeled data**.
        

---

### 🔹 How Do Models Learn in PU?

Three big strategies:

- **Two-step**: First guess which unlabeled examples are _likely_ negatives, then train normally.
    
- **Biased learning**: Pretend unlabeled = negative, but don’t trust it fully (penalize mistakes differently).
    
- **Class prior methods**: Estimate the fraction of positives in the data and bake it into training.
    

---

### 🔹 Evaluation Problem

Normally, you’d test models using both positives & negatives. But here, negatives aren’t labeled.

- Workaround: Use **math tricks** (e.g., under SCAR) to estimate standard metrics like accuracy and precision.
    

---

### 🔹 Applications

- **Healthcare**: finding undiagnosed diseases.
    
- **Marketing**: predicting what people will buy/click.
    
- **Biology**: identifying disease-related genes.
    
- **Fraud & anomaly detection**: spotting rare events without clear negatives.
    
- **Search/recommendations**: knowing what people like, but not what they dislike.
    

---

### 🔹 Takeaway

PU learning is about **making the best use of incomplete data**:

- You only know some positives.
    
- You don’t know the negatives.
    
- With assumptions + clever tricks, you can still train models that perform well in the real world.
    

---

👉 In **one sentence**:  
PU learning is a way to train machine learning models when you only know which things are _positive_ and everything else is _uncertain_, which happens a lot in real-life data.
