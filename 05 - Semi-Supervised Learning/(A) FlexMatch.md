#SEMI-SL 

Proposes the idea of using adaptive thresholds based on how well learned a certain class is, aka how many samples are above a threshold. Worse class learning status
--> class isnt very represented therefore lower threshold.


Process:
In simple terms, adaptive thresholds in FlexMatch, introduced by Curriculum Pseudo Labeling (CPL), are computed by dynamically adjusting them for each class based on the model's current learning status, rather than using a single fixed threshold for all classes. This process does not add new parameters or significant computational overhead.

Here's a breakdown of how it works:

1. **Estimating Learning Status (σt(c))**:
    
    - The core idea is that the "learning effect" for a particular class `c` at a given time step `t` can be estimated by **counting the number of unlabeled data samples** whose predicted confidence is **above a pre-defined fixed threshold (τ)** and are also **classified into that specific class `c`**.
    - This count is denoted as `σt(c)`.
    - If a class has fewer samples meeting these criteria (`σt(c)` is small), it's considered to have a **greater learning difficulty or a worse learning status**.
2. **Normalising the Learning Effect (βt(c))**:
    
    - To make these learning effect estimates useful for scaling, `σt(c)` is **normalized** to a value `βt(c)` that ranges between 0 and 1.
    - This is done by dividing `σt(c)` by the maximum `σt` observed across all classes at that time step: **`βt(c) = σt(c) / max_c σt`**.
    - Consequently, the **best-learned class** (the one with the highest `σt(c)`) will have its `βt(c)` equal to 1.
3. **Calculating the Flexible Threshold (Tt(c))**:
    
    - The adaptive threshold for each class `c` at time `t`, denoted `Tt(c)`, is then computed by **multiplying the normalized learning effect (`βt(c)`) by the pre-defined fixed threshold (`τ`)**: **`Tt(c) = βt(c) · τ`**.
    - This means that for classes that are **hard to learn** (lower `βt(c)`), their thresholds are **lowered**, allowing more unlabeled samples from those classes to be used in training.
    - Conversely, for **well-learned classes** (higher `βt(c)`), their thresholds are **raised**, ensuring only higher-quality samples contribute to training.
    - As learning progresses and classes become reliably accurate, their thresholds will gradually approach `τ`. Thresholds are updated at each iteration and can also decrease if predictions change.
4. **Threshold Warm-up**:
    
    - At the very beginning of training, the model's predictions can be unstable. To prevent unreliable threshold adjustments, a "warm-up" process is introduced.
    - During warm-up, the normalization denominator in `βt(c)` is adjusted to include the number of **unused unlabeled data**. This ensures that the estimated learning effects for all classes gradually rise from zero, preventing early fluctuations and creating a "learning boom" where more unlabeled data can be utilised.
5. **Non-linear Mapping (Optional)**:
    
    - Optionally, a non-linear mapping function `M(·)` can be applied to `βt(c)` before multiplying by `τ`, so `Tt(c) = M(βt(c)) · τ`. This allows the thresholds to grow at a non-linear rate, for example, growing slowly when `βt(c)` is small and becoming more sensitive as it gets larger. A convex function like `x/(2-x)` has been found to perform well.

The calculated `Tt(c)` is then used to filter unlabeled data for each class when computing the unsupervised loss in FlexMatch, ensuring that only confident predictions (above `Tt(c)`) contribute to training.


The warm-up process with normalisation in FlexMatch addresses a specific problem that occurs at the **early stage of training**: the model's predictions for unlabeled data can be unstable and prone to **confirmation bias**. This means the model might "blindly predict most unlabeled samples into a certain class" based on how its parameters were initially set up.

If the standard normalisation formula (Eq. 6) were used immediately, where `βt(c) = σt(c) / max_c σt`, these unreliable early predictions could lead to heavy fluctuations in the flexible thresholds for each class. This could also cause some class thresholds to prematurely reach the maximum (`τ`), filtering out most unlabeled data from those classes even if they are not truly well-learned.

To prevent these issues, FlexMatch introduces a **warm-up process** by modifying the denominator of the normalisation formula:

**Original normalisation (without warm-up)**: `βt(c) = σt(c) / max_c σt`

**Warm-up normalisation (Eq. 11)**: `βt(c) = σt(c) / max { max_c σt, N - Σc σt }`

Let's break down the warm-up part:

1. **What** **N - Σc σt(c)** **means**:

    ◦ `N` is the total number of unlabeled data.

    ◦ `Σc σt(c)` is the sum of `σt(c)` for all classes `c` at time `t`. Remember `σt(c)` represents the count of unlabeled samples for class `c` that have a prediction confidence above the fixed threshold `τ`.

    ◦ Therefore, `N - Σc σt(c)` can be understood as the **number of unlabeled data that have** **not yet been used** for pseudo-labeling because their confidence predictions haven't reached the fixed threshold `τ` or they haven't been confidently assigned to any specific class yet.

2. **How the warm-up normalisation works**:

    ◦ In the early stages of training, when the model's predictions are unreliable, most unlabeled data will not have a confidence above `τ`. This means `Σc σt(c)` will be small, and consequently, `N - Σc σt(c)` (the number of "unused" data) will be **large**.

    ◦ The denominator in the warm-up formula `max { max_c σt, N - Σc σt }` will therefore be dominated by `N - Σc σt(c)`.

    ◦ This ensures that the denominator starts large and **gradually decreases** as more unlabeled data gain high-confidence predictions and get "used" (`Σc σt(c)` increases).

    ◦ As a result, `βt(c)` for all classes will **gradually rise from zero**.

3. **Effect on thresholds**:

    ◦ Since `Tt(c) = βt(c) · τ`, a `βt(c)` that gradually rises from zero means that the flexible thresholds `Tt(c)` for all classes will also **gradually increase from zero**.


## 🧠 The Problem

At the **start of training**, the model is still guessing almost randomly.  
If we immediately trust these guesses to decide which unlabeled samples are "confident," then:

- It might **incorrectly push most samples into one class** just by chance.
    
- This would make the **class thresholds jump around** a lot.
    
- Some classes could **get very high thresholds too early**, so **almost no unlabeled data** would be used from those classes later.
    

This is called **confirmation bias** — the model wrongly "confirms" its early random guesses.

---

## 💡 The Warm-Up Idea

FlexMatch fixes this by **starting with very low thresholds** for all classes and **slowly raising them** as the model becomes more confident.

It does this by changing how it **normalizes** the number of confident samples in each class:

- Normally:
    
    βt(c)=σt(c)max⁡cσt\beta_t(c) = \frac{\sigma_t(c)}{\max_c \sigma_t}
- With warm-up:
    
    βt(c)=σt(c)max⁡(max⁡cσt,  N−∑cσt(c))\beta_t(c) = \frac{\sigma_t(c)}{\max(\max_c \sigma_t, \; N - \sum_c \sigma_t(c))}

Here:

- σt(c)\sigma_t(c) = number of unlabeled samples the model is confident about in class cc.
    
- NN = total number of unlabeled samples.
    
- N−∑cσt(c)N - \sum_c \sigma_t(c) = number of **unused (low-confidence) samples**.
    

---

## ⚙️ How It Works

- At the very beginning:
    
    - Almost all samples are low-confidence.
        
    - So ∑cσt(c)\sum_c \sigma_t(c) is tiny, and N−∑cσt(c)N - \sum_c \sigma_t(c) is **huge**.
        
    - This makes the denominator large → βt(c)\beta_t(c) becomes **very small** → thresholds are low.
        
- As training goes on:
    
    - The model becomes more confident, ∑cσt(c)\sum_c \sigma_t(c) grows.
        
    - N−∑cσt(c)N - \sum_c \sigma_t(c) gets smaller → the denominator shrinks.
        
    - βt(c)\beta_t(c) **increases**, raising the thresholds.
        

---

## 🎯 The Effect

- Early on, **almost all unlabeled data are accepted** because thresholds are tiny → more learning signals.
    
- Later, **thresholds tighten** as the model gets better, filtering out only low-quality predictions.
    
- This **avoids confirmation bias** from bad early guesses and **stabilizes learning**.
    



## 📌 Normally (without warm-up)

- The learning status of class ccc is measured as:
    
    βt(c)=σt(c)max⁡cσt\beta_t(c) = \frac{\sigma_t(c)}{\max_c \sigma_t}βt​(c)=maxc​σt​σt​(c)​
- This only depends on **σt(c)\sigma_t(c)σt​(c)**: how many high-confidence pseudo-labeled samples currently belong to class ccc.
    

So if class ccc has few confident samples, its βt(c)\beta_t(c)βt​(c) will be small compared to the most confident class.

---

## 📌 With warm-up (with unused data)

FlexMatch changes it to:

βt(c)=σt(c)max⁡(max⁡cσt,  N−∑cσt(c))\beta_t(c) = \frac{\sigma_t(c)}{\max \big( \max_c \sigma_t,\; N - \sum_c \sigma_t(c) \big)}βt​(c)=max(maxc​σt​,N−∑c​σt​(c))σt​(c)​

The key part is N−∑cσt(c)N - \sum_c \sigma_t(c)N−∑c​σt​(c), which is the **number of unlabeled samples that are still unused (not confidently predicted as any class)**.

This term affects **all classes equally** because it’s the same for every ccc.

So:

- In the **early stage**, N−∑cσt(c)N - \sum_c \sigma_t(c)N−∑c​σt​(c) is huge.
    
- This huge denominator makes **every βt(c)\beta_t(c)βt​(c) very small**, no matter how many confident samples a class has.
    
- So the “learning status” of each class is **scaled down globally**, as if the model is saying:
    
    > “We’re still unsure about almost all the data, so no class should be considered well-learned yet.”
    
- As training progresses, unused data shrinks (∑cσt(c)\sum_c \sigma_t(c)∑c​σt​(c) grows), and the denominator is gradually dominated by max⁡cσt\max_c \sigma_tmaxc​σt​ again.  
    Then βt(c)\beta_t(c)βt​(c) depends mainly on the relative number of confident samples per class.


Very simply:

- The flexible threshold for each class is
    
    Tt(c)=βt(c)⋅τT_t(c) = \beta_t(c) \cdot \tauTt​(c)=βt​(c)⋅τ
- At the start, because **βt(c)\beta_t(c)βt​(c) is made very small** by the large unused-data term,  
    → **Tt(c)T_t(c)Tt​(c) starts very low**.
    
- As training goes on, unused data decreases, βt(c)\beta_t(c)βt​(c) rises,  
    → **Tt(c)T_t(c)Tt​(c) gradually increases** toward the fixed threshold τ\tauτ.
    

✅ So the warm-up makes **initial thresholds very low** (easy to accept data) and **slowly raises them** as the model becomes more confident.


![[FlexMatch 2.pdf]]