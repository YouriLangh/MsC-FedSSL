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








ENCORE integrates two key components: **Class-Aware Confidence Calibration (CAC)** and **Adaptive Confidence Thresholding (ACT)**.

Here's how they work in simple terms:

1. **Class-Aware Confidence Calibration (CAC)**

    ◦ **Problem:** Standard pseudo-labeling often uses a single, overall confidence threshold for _all_ classes. However, different classes naturally have varying levels of prediction certainty. If the threshold is too high, it might discard correct pseudo-labels for classes that the model is inherently less confident about (false negatives). If it's too low, it might introduce incorrect pseudo-labels for classes where the model is usually very confident (false positives).

    ◦ **How it works:** CAC addresses this by estimating prediction reliability _for each class individually_. It uses the available **labeled training data** as a guide. For each class, CAC calculates the **average confidence** that the model has when it makes a _correct_ prediction for that specific class.

    ◦ **Result:** This process generates **class-specific reliability indicators (R)**. These indicators act as the initial, tailored confidence thresholds for filtering pseudo-labels for each class, rather than using a single, global threshold. This makes the initial filtering more accurate and accounts for the unique confidence characteristics of each class.

2. **Adaptive Confidence Thresholding (ACT)**

    ◦ **Problem:** Even with class-specific thresholds from CAC, the ideal threshold might change as the model learns and its performance evolves throughout training. Manually adjusting these thresholds would require a large validation dataset and significant effort, which goes against the goal of semi-supervised learning.

    ◦ **How it works:** ACT is a dynamic system that continuously adjusts these confidence thresholds during training based on real-time feedback.

   ▪ **Multiple "Assessor Networks":** At each training step, ENCORE creates three "assessor networks" which are essentially copies of the main "student" model currently being trained.

   ▪ **Testing Different Thresholds:** Each of these three assessor networks applies a _different confidence threshold controller_ (derived from the CAC reliability indicators) to filter the pseudo-labels generated for the unlabeled data.

   ▪ **Performance Evaluation:** After each assessor network trains briefly with its assigned threshold, its performance is immediately evaluated on a small batch of the _labeled data_ using a metric like the Dice score (which measures segmentation accuracy).

   ▪ **Selecting the Best Threshold:** The threshold controller that leads to the **highest Dice score** on the labeled data is chosen as the "optimal" one for the _next cycle of the student model's training_.

   ▪ **Dynamic Adjustment of Threshold Range:** If a particular threshold controller (e.g., consistently the lowest or consistently the highest) is selected for a certain number of consecutive times, it signals that the overall range of thresholds needs to shift. ENCORE then automatically adjusts the values of the threshold controllers to be more lenient or stricter, accordingly. This feedback loop ensures that the pseudo-label filtering adapts to the model's learning progress.


2. **ACT Initializes** **Three Sets** **of Class-Specific Thresholds:** The **Adaptive Confidence Thresholding (ACT)** module then uses this class-wise reliability indicator `R` to initialise its three "confidence threshold controllers" ($\tau_1, \tau_2, \tau_3$).

    ◦ These controllers are defined as: `(τ1, τ2, τ3) = (α1R, R, α2R)`.

    ◦ Since `R` is a vector of class-specific reliability scores, `τ1`, `τ2`, and `τ3` are also **vectors of class-specific thresholds**.

    ◦ For example, if `R = [R_A, R_B, R_C]`:

        ▪ `τ1` would be `[α1 * R_A, α1 * R_B, α1 * R_C]`

        ▪ `τ2` would be `[R_A, R_B, R_C]` (the original CAC values)

        ▪ `τ3` would be `[α2 * R_A, α2 * R_B, α2 * R_C]`

    ◦ Here, `α1` and `α2` are fixed scalar "threshold adaptors" (e.g., 0.98 and 1.02) that define an initial search space around the CAC-derived reliability `R`. So, `τ1` represents a slightly stricter set of class-wise thresholds, `τ2` is the original set, and `τ3` is a slightly more lenient set.

3. **Assessor Networks Use These** **Sets** **of Class-Wise Thresholds:** At each iteration, three "assessor networks" are created as copies of the current student model. Each assessor network is assigned one of these three **threshold controllers** (`τ1`, `τ2`, or `τ3`).

So t1 lowers the thresholds and t3 increases them and t2 keeps them the same.

![[Feedback-Driven Pseudo-Label Reliability Assessment.pdf]]