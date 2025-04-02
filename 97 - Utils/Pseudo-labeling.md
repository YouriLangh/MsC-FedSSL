#utils

Pseudo-labeling employs a trained model to annotate the unlabelled
data and then retrains the model on both the original
labeled data and newly pseudo-labeled data.

Pseudo-labeling is a technique in **semi-supervised learning**, which is a method used when we have only a small amount of labeled data but a lot of unlabeled data. Instead of wasting the unlabeled data, we use a trained model to make predictions on it and treat those predictions as if they were real labels (pseudo-label). This allows the model to learn from more data, improving its accuracy.


#### **tep 3: Pseudo-Labeling Process**

1. The client receives an **unlabeled** ECG signal.
2. The model **makes a prediction** and assigns a class (e.g., "Atrial Premature Beat").
3. If the prediction confidence is above a threshold (e.g., 95%), the label is accepted as a **pseudo-label**.
4. If the confidence is low, the sample is **discarded** or remains unlabeled.
Weak augmentation (% per class, rather than One-Hot Encoding)