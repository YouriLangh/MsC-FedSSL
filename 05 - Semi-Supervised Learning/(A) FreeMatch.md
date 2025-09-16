#SEMI-SL 

Has a good overview of all the adaptive threshold solutions & just SSL solutions.


Commonly used datasets for SSL:
We evaluate FreeMatch on common benchmarks: CIFAR-10/100 (Krizhevsky et al., 2009),
SVHN (Netzer et al., 2011), STL-10 (Coates et al., 2011) and ImageNet (Deng et al., 2009).

We can actually see that in EXTREMELY low labeled cases, Freematch >>> Fixmatch, otherwise its pretty close.


FreeMatch uses **two types of self-adaptive thresholds**:

1. **Self-adaptive Global Threshold (τt)**:
    
    - This threshold reflects the **overall learning status of the model across the entire dataset** at a given time step `t`.
    - It is calculated as the **Exponential Moving Average (EMA) of the maximum prediction confidence** (`max(qb)`) observed from the unlabeled data samples in each training batch.
    - Initially, at the very start of training (`t=0`), this global threshold `τt` is set to `1/C`, where `C` is the total number of classes.
    - **The global threshold is designed to stably increase during training**. This means it starts relatively low to encourage the use of more unlabeled data in the early stages when the model is less confident, and then rises as the model becomes more accurate to filter out potentially incorrect pseudo-labels and reduce confirmation bias.
2. **Self-adaptive Local Threshold (p̃t(c))**:
    
    - This threshold is **class-specific**, meaning there's a unique local threshold for each class `c`. Its purpose is to modulate the global threshold based on the **individual learning difficulty and diversity within each class**.
    - It's computed as the **EMA of the model's predicted probability for each specific class `c`** (`qb(c)`) from the unlabeled data in a batch. This gives `p̃t(c)` for each class `c`.
    - Similar to the global threshold, it's initialised as `1/C` for each class at `t=0`.

**Combining Global and Local Thresholds for the Final Self-Adaptive Threshold (τt(c))**:

- The final self-adaptive threshold for a particular class `c` at iteration `t`, denoted `τt(c)`, is derived by **multiplying the global threshold (`τt`) by a normalised version of the local threshold (`p̃t(c)`)**.
- The normalisation, called `MaxNorm`, ensures that the class with the highest `p̃t(c)` effectively sets the upper bound for the modulation. The formula is: **τt(c) = (p̃t(c) / max{p̃t(c) : c ∈ [C]}) · τt**.

**How the Thresholding Dynamically Works During Training**:

- **Early Training Stages**: When the model is still learning and its predictions are less reliable, the global and local thresholds will be **relatively low**. This allows **more unlabeled data samples to be included in the training** process, even if their confidence is not extremely high. This initial "learning boom" helps speed up convergence and improve the utilisation of unlabeled data.
- **Later Training Stages**: As the model progresses and becomes more confident in its predictions, both the global and local thresholds **adaptively increase**. This stricter filtering ensures that only **higher-quality pseudo-labels** (those with high confidence) are used for training, which helps to **alleviate confirmation bias** (where the model reinforces its own mistakes).

The `τt(c)` calculated this way is then used to filter the unlabeled data when computing the unsupervised loss. Only unlabeled samples whose maximum predicted confidence is **greater than or equal to the self-adaptive threshold specific to their predicted class** (`τt(arg max (qb))`) are used to compute the loss.

This dual adaptive mechanism allows FreeMatch to adjust thresholds without manual tuning, making it more robust and effective, especially when labeled data is scarce. This is a key distinction from methods that rely on fixed high thresholds (like FixMatch and UDA) or ad-hoc global threshold adjustments (like Dash and AdaMatch), or even FlexMatch, which maps local thresholds from a pre-defined fixed global threshold.



Sure! Here’s a **very simple explanation** of FreeMatch’s two thresholds:

---

## 🧠 Two thresholds in FreeMatch

**1. Global threshold (τt)**

- One threshold for **all classes together**.
    
- Shows **how confident the model is overall**.
    
- Starts low (`1/C`) and **slowly increases** as the model gets better.
    
- Low at first → accept more data.
    
- High later → only accept confident data.
    

---

**2. Local threshold (p̃t(c))**

- One threshold for **each class c**.
    
- Shows **how well the model is doing on that class**.
    
- Also starts at `1/C` and grows as the model gets better at that class.
    
- If a class is harder, its threshold grows slower.
    

---

## ⚙️ How they work together

- Final threshold for class `c`:
    
    τt(c)=p~t(c)max⁡cp~t(c)×τt\tau_t(c) = \frac{p̃_t(c)}{\max_c p̃_t(c)} \times \tau_t
- So:
    
    - **Global threshold** controls the overall level.
        
    - **Local threshold** adjusts it up or down per class.
        

---

## 📈 Why this helps

- **Early on**: thresholds are low → use lots of unlabeled data.
    
- **Later**: thresholds go up → only use high-confidence data → fewer mistakes.
    


Perfect — this is a very good question.  
Let’s walk through **how the formula can actually go up or down** for each class as training progresses.

---

## 📌 The formula again

τt(c)=p~t(c)max⁡c′p~t(c′)×τt\tau_t(c) = \frac{p̃_t(c)}{\max_{c'} p̃_t(c')} \times \tau_t

- p~t(c)p̃_t(c) = EMA of predicted probability for class cc
    
- max⁡c′p~t(c′)\max_{c'} p̃_t(c') = the largest local value among all classes
    
- τt\tau_t = global threshold (EMA of max confidence in batches)
    

**So**:

- If class cc gets better → p~t(c)p̃_t(c) increases
    
- If another class gets better faster → max⁡c′p~t(c′)\max_{c'} p̃_t(c') increases more
    
- So the ratio can **go up or down depending on relative progress**
    

---

## 📈 Example: class A increases, class B decreases

Let’s say at time **t**:

- Global threshold: τt=0.4\tau_t = 0.4
    
- Local scores:
    
    - p~t(A)=0.3p̃_t(A) = 0.3
        
    - p~t(B)=0.25p̃_t(B) = 0.25
        
    - max = 0.3
        

Then:

τt(A)=(0.3/0.3)×0.4=0.4\tau_t(A) = (0.3 / 0.3) \times 0.4 = 0.4 τt(B)=(0.25/0.3)×0.4≈0.33\tau_t(B) = (0.25 / 0.3) \times 0.4 \approx 0.33

---

Now at **t+1** the model improves at A but struggles with B:

- Global threshold: τt+1=0.5\tau_{t+1} = 0.5
    
- Local scores:
    
    - p~t+1(A)=0.45p̃_{t+1}(A) = 0.45 ↑
        
    - p~t+1(B)=0.2p̃_{t+1}(B) = 0.2 ↓
        
    - max = 0.45
        

Then:

τt+1(A)=(0.45/0.45)×0.5=0.5    (increased from 0.4)\tau_{t+1}(A) = (0.45/0.45) \times 0.5 = 0.5 \;\; (\text{increased from 0.4}) τt+1(B)=(0.2/0.45)×0.5≈0.22    (decreased from 0.33)\tau_{t+1}(B) = (0.2/0.45) \times 0.5 \approx 0.22 \;\; (\text{decreased from 0.33})

✅ So A’s threshold goes **up** (model is more confident),  
while B’s threshold goes **down** (model is struggling).

---

## 📝 Key point

- The **global part τt\tau_t** pulls all thresholds **up over time** as the whole model gets better.
    
- The **local ratio** makes some classes **rise faster or slower** depending on how well they are learned compared to the others.
    


## 🧠 What “EMA of predicted probability for a class” means

- For each unlabeled batch, the model gives a probability distribution over classes for each sample.  
    Example: for one sample, it might predict  
    [0.1,0.7,0.2][0.1, 0.7, 0.2][0.1,0.7,0.2] → 70% chance class 2.
    
- For a class ccc, we look at **how confident the model is on that class across the batch**.  
    So for class ccc:
    
    - Collect all predicted probabilities qb(c)q_b(c)qb​(c) for that class from this batch.
        
    - Compute the **average predicted probability** for class ccc in this batch.
        
- Then we **smooth these batch averages over time** using an **Exponential Moving Average (EMA)**:
    
    p~t(c)=α⋅p~t−1(c)+(1−α)⋅avg_prob_in_batch(c)p̃_t(c) = \alpha \cdot p̃_{t-1}(c) + (1 - \alpha) \cdot \text{avg\_prob\_in\_batch}(c)p~​t​(c)=α⋅p~​t−1​(c)+(1−α)⋅avg_prob_in_batch(c)

Here:

- p~t(c)p̃_t(c)p~​t​(c) is the smoothed "confidence trend" for class ccc
    
- α\alphaα is a smoothing factor (like 0.9)
    

---

## 💡 Why do this

- The EMA makes the local threshold signal **stable and less noisy**.
    
- Instead of jumping up and down from one batch to another, it gives a **slow trend** showing how confident the model is becoming about class ccc over time.

![[FreeMatch.pdf]]