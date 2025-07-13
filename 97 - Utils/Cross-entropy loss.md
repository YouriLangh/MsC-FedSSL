#utils 

1. **The Model Predicts Probabilities**
    - A neural network outputs a probability for each class (e.g., for an ECG classifier, it predicts the likelihood of different heart conditions).
    - Example: Suppose we have 3 possible classes:
        - **Normal Beat** (70% confidence → 0.7)
        - **Atrial Premature Beat** (20% confidence → 0.2)
        - **Ventricular Escape Beat** (10% confidence → 0.1)
2. **The True Label is One-Hot Encoded**
    
    - If the real class is **"Atrial Premature Beat"**, the correct label is:
        - Normal Beat → **0**
        - Atrial Premature Beat → **1**
        - Ventricular Escape Beat → **0**
3. **Cross-Entropy Calculates the Error**
    
    - The formula for cross-entropy loss: 
			`` Loss=−∑(True Label)×log⁡(Predicted Probability)``
	
    - Since only the **correct class contributes**, we only use the predicted probability for "Atrial Premature Beat" (0.2): Loss=−log⁡(0.2)=1.61
    - Lower loss = better prediction.
    - If the model had predicted **0.9** for "Atrial Premature Beat" instead of 0.2, the loss would be: −log⁡(0.9)=0.10 (much better!)

- **If the model is very confident and correct (e.g., predicts 0.99 for the right class), the loss is close to 0.**
- **If the model is very confident but wrong (e.g., predicts 0.99 for the wrong class), the loss is very high.**
- **If the model is uncertain (e.g., predicts 0.33 for each class), the loss is moderate.**

## NT-Xent (Normalized Temperature-Scaled Cross Entropy)
NT-Xent is a loss function used in **contrastive learning**, which helps a model learn **useful representations** by pulling similar data points together and pushing different ones apart. In the case of ECG classification, it would help the model learn a **good feature space** before actual classification.


We want the model to **understand the underlying patterns** before using labels.
##### **Step 1: Create Two Augmented Versions of the Same ECG**

- Take an **ECG signal** (e.g., a "Normal Beat").
- Apply **two different transformations** to create two versions:
    
    - **Weak augmentation:** Slight shifting of the signal.
    - **Strong augmentation:** Adding noise.
    
    → These are called **positive pairs** because they represent the same underlying heartbeat.

##### **Step 2: Compare the ECG Representations**

- The model processes both augmented ECGs through a neural network (like ResNet-9).
- Instead of predicting a class label, the model **maps** them to a feature space (a vector of numbers).
- The goal is to **make the feature representations of positive pairs similar**.



- NT-Xent includes a **temperature parameter (τ)**, which controls how much the model focuses on hard-to-distinguish examples.
- If τ is low, only **very similar pairs are strongly encouraged to be close**.
- If τ is high, the model is more **lenient** and allows slightly different signals to be grouped together.



In contrastive learning, the original sample would typically be projected along with augmented versions, but **this paper does not use contrastive loss**.