#SSL 


How does SSL work? 
##### **1. Data Augmentation (Early Stage – Input Processing)**

**When:** Before feeding data into the model.  
**Why:** Creates diverse versions of the same data so the model learns robust representations.  
**Example in SSL:**

- Two different augmented views of an image are generated before being fed into a contrastive learning model (e.g., SimCLR).
---

##### **2. Pretext Task (Self-Supervised Training Phase)**

**When:** After augmentation, during the model’s training phase.  
**Why:** Encourages the model to learn meaningful representations by solving a task.  
**Example in SSL:**

- The model is trained to identify whether two augmented images come from the same original image (contrastive learning).
- The model predicts masked words in a sentence (language modeling).


**Pretext Task in SimCLR:** "Given two augmented views of an image, learn that they belong to the same object while distinguishing them from other images."

MoCo and SimCLR achieve comparable results to state of the art supervised methods.


How to identify the right [[Pretext tasks]]? Depends on downstream task, dont use image rotations if downstream task is to identify which way is up.

**Section 3**
Different type of contrastive learning architectures:

- End-to-End Learning: SimCLR
- Memory Bank
- Momentum Encoder
- Clustering

**Section 4**
"Most of the works in CL utilize ResNet"

To train the encoder we use a pretext task with contrastive loss for backpropagation.

![[A Survey on Contrastive Self-Supervised Learning.pdf]]