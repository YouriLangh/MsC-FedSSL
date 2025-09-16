#SEMI-SL 

**MixMatch (Berthelot et al., 2019)** – unified pseudo-labeling + consistency + mixup.

Semi-supervised learning [6] (SSL) seeks to largely alleviate the need for labeled data by allowing
a model to leverage unlabeled data. Many recent approaches for semi-supervised learning add a
loss term which is computed on unlabeled data and encourages the model to generalize better to
unseen data. In much recent work, this loss term falls into one of three classes (discussed further
in Section 2): entropy minimization [18, 28]—which encourages the model to output confident
predictions on unlabeled data; consistency regularization—which encourages the model to produce
the same output distribution when its inputs are perturbed; and generic regularization—which
encourages the model to generalize well and avoid overfitting the training data.
In this paper, we introduce MixMatch, an SSL algorithm which introduces a single loss that combines all of these 3 and averages them i think.

% “Pseudo-Label” [28] does entropy minimization implicitly by constructing hard
% (1-hot) labels from high-confidence predictions on unlabeled data and using these as training targets
% in a standard cross-entropy loss (LOOK AT MIXMATCH SECTION 2)



### 1. Generate pseudo-labels for unlabeled data

- For each unlabeled sample u, apply **K augmentations** and average the model’s predictions.
    
- Apply a **sharpening function** (temperature scaling) to make the distribution more confident.
    
- That gives you a “guessed label” $y^u$. (y hat)
    

---

### 2. Combine labeled and unlabeled

- Now you have:
    
    - Labeled set: (xl,yl)
        
    - Unlabeled set with pseudo-labels: (u,y^$_u$)
        

---

### 3. Apply **MixUp**

MixUp (Zhang et al., 2017) linearly interpolates both **inputs** and **labels**:
![[MixMatch - mixup.png]]

So the model sees a **mixture of real + pseudo-labeled data** and learns from the interpolated labels.

---

### 4. Why MixUp helps in SSL

- **Smooths the decision boundary**: prevents the model from memorizing noisy pseudo-labels.
    
- **Regularizes training**: encourages linear behavior between data points.
    
- **Bridges labeled & unlabeled data**: forces the network to treat pseudo-labeled data more like labeled data by mixing them together.
    

---

🔑 In short:  
In **MixMatch**, mixup is applied on the union of **(labeled data + unlabeled data with guessed labels)**, producing mixed examples with **soft labels**. This prevents overconfidence on noisy pseudo-labels and gives the algorithm much better generalization than pseudo-labeling alone.


So the new examples (x′,y′)(x', y')(x′,y′) are “hybrids”:

- Sometimes a **blend of two labeled examples**
    
- Sometimes a **blend of two unlabeled+pseudo-labeled examples**
    
- Sometimes a **blend of a labeled and an unlabeled+pseudo-labeled example**


The **MixUp step** blends both the **inputs** (images) and their **labels** (true labels or pseudo-labels).

So instead of training on:

- a single labeled image with a one-hot label,
    
- or a single unlabeled image with a guessed label,
    

you train on a **synthetic example** that is a smooth interpolation of two data points.

That forces the model to behave **linearly between examples** instead of making abrupt jumps.

So a minibatch, you take 2 samples and then combine these into a new combined sampled.

![[MixMatch.pdf]]