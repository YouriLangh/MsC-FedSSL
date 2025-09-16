#SEMI-SL 

Can be quoted for:
`Medical data being harder due to lack of annotations & filtering labels with fixed = discarding good labels & IMPORTANTLY: Choice of threshold inconsistenly affects the results. Also that our softmax solution is kinda bad.`
![[Feedback-Driven Pseudo-Label Reliability Assessment. Redefining Thresholding for Semi-Supervised Semantic Segmentation.png|300]]

They do multi class segmentation?

Its impractical/unrealistic to perform hyperparameter search over confidence threshold as it contradicts the real-world scenario.

CAC & ACT:

**ENCORE (Ensemble-of-Confidence Reinforcement)** is a semi-supervised semantic segmentation strategy that **dynamically refines pseudo-labels** to improve performance when labeled data is scarce. Its main goal is to avoid the pitfalls of using a fixed confidence threshold for pseudo-label filtering, which can be suboptimal across classes and training stages.

### 1. Class-Aware Confidence Calibration (CAC)

- **Problem**: Different classes naturally have different prediction confidences; a single global threshold can discard correct pseudo-labels for “low-confidence” classes or accept incorrect ones for “high-confidence” classes.
    
- **How it works**:
    
    1. Train the network on the labeled data.
        
    2. For each class, compute the average confidence of correctly predicted instances.
        
    3. Use this class-specific average as the **initial confidence threshold (R)** for that class.
        
- **Benefit**: Accounts for class-specific confidence variability, improving initial pseudo-label reliability.
    

### 2. Adaptive Confidence Thresholding (ACT)

- **Problem**: Even class-specific thresholds may become suboptimal during training; manual tuning is impractical.
    
- **How it works**:
    
    1. Initialize three threshold controllers (τ1, τ2, τ3) around each class’s R.
        
    2. In each training iteration, create three **assessor networks** (copies of the student model) and train them briefly using pseudo-labels filtered with each τ.
        
    3. Evaluate each assessor on a small labeled batch (e.g., using Dice score).
        
    4. Select the threshold that performs best (τ*) for the next training cycle.
        
    5. Dynamically adjust τ1–τ3 if extremes are consistently chosen, preventing suboptimal fixation.
        
- **Benefit**: Continuously adapts thresholds based on real-time feedback, keeping reliable pseudo-labels and filtering out unreliable ones.
    

---

**Integration of CAC + ACT**

- CAC gives **class-specific starting thresholds**, and ACT **dynamically refines them** during training.
    
- This feedback-driven loop **automates pseudo-label selection**, reduces manual tuning, and significantly improves semi-supervised segmentation performance.
    

It does t his for each minibatch? Doesnt this make it extremely unreliable?

![[Feedback-Driven Pseudo-Label Reliability Assessment.pdf]]