#FedSSL #SEMI

inspects sources 6 & 22

Traditional self-training vs BN FedAvg achieves way better result; explanation of difference:
Great question! Let’s simplify the key idea behind **traditional self-training (Equation 3)** vs. the improved method using **consistency regularization (Equation 4)** in the paper _"Benchmarking Semi-supervised Federated Learning"_.

---

### 🧠 **What’s going on conceptually?**

#### **Traditional self-training (Equation 3):**

- Each user device (client) has **only unlabeled data**.
    
- The user model makes a prediction on a data point.
    
- If it’s confident enough (e.g., prediction probability > 95%), it treats the predicted label as **“pseudo ground truth”** and trains on that.
    
- This is simple, but error-prone—bad pseudo-labels can reinforce mistakes.
    

> Think of it like a student teaching themselves using flashcards: if they _think_ the answer is right, they move on. But if they’re wrong and too confident, they just memorize the wrong answer.

---

#### **Consistency Regularization (Equation 4):**

- Still no true labels on users.
    
- But instead of trusting just **one prediction**, the model:
    
    1. Applies a **weak data augmentation** (like a flip or shift) to get a confident prediction.
        
    2. Applies a **strong augmentation** (like cropping, rotating, color shift) to the same image.
        
    3. Trains the model so that the prediction on the strong version **matches** the confident prediction from the weak one.
        
- This **enforces consistency**—if an image still looks like a dog even when rotated or cropped, it should still be labeled as a dog.
    

> This is like a student checking if their answer still makes sense when the question is rephrased or presented differently. It’s a more robust way to learn.

---

### 📈 **So why does this help in Federated Learning?**

1. **Equation 3** leads to lots of noise: bad pseudo-labels → bad learning.
    
2. **Equation 4** uses **consistency between augmentations**, so it doesn’t rely solely on a potentially bad label—it learns from the agreement.
    
3. When this change is combined with **Group Normalization (GN)** instead of **Batch Norm (BN)** (which handles non-iid data better), it boosts accuracy _massively_.
    

---

### 🔍 Real example from the paper:

|Method|Accuracy on CIFAR-10|
|---|---|
|Traditional self-training (Eq 3)|**52.10%**|
|Consistency regularization (Eq 4)|**85.11%**|
|+ Group Norm|**92.83%**|

That’s **a 40% improvement** just by switching from Eq 3 → Eq 4 and using better normalization!

---

### Summary

|Aspect|Traditional Self-training (Eq 3)|Consistency Regularization (Eq 4)|
|---|---|---|
|Requires confidence?|✅ Yes|✅ Yes|
|Relies on one prediction?|✅ Yes|❌ Uses weak + strong augmentation|
|Robust to noise?|❌ Not very|✅ Yes, avoids learning from uncertain data|
|Accuracy|🚫 Low|✅ Much higher with GN|

Let me know if you want an analogy with diagrams or to see how this looks in pseudocode!

![[Benchmarking-Semi-supervised-Federated-Learning.pdf]]