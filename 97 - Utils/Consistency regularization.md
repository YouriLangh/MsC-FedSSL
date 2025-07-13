#utils 

Consistency regularization is a key technique in **semi-supervised learning (SSL)** that leverages the assumption that a model’s predictions should be consistent under small perturbations of the input. The idea is that even if a data point is unlabeled, the model should output similar predictions when the input is slightly altered.

### **How Consistency Regularization Works:**

1. **Unlabeled Data Augmentation** – Apply small transformations (e.g., noise, rotations, cropping, or adversarial perturbations) to the input data.
    
2. **Prediction Consistency** – The model is trained to produce the same (or similar) output for both the original and perturbed versions of an input.
    
3. **Loss Function** – A consistency loss term (such as Mean Squared Error or KL Divergence) is added to the total loss to enforce this behavior.