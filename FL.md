
## 🧩 Your Current Structure

```latex
\section{Federated Learning}

\subsection{Overview}

\subsection{Core Challenges}

\subsection{Characterizing Data Heterogeneity}
  \subsubsection{Modeling Non-IID Data with the Dirichlet Distribution}

\subsection{Taxonomy of FL}
```

---

## ✅ Why this is the **ideal structure**

|Section|Purpose|Why it’s in the right place|
|---|---|---|
|**Overview**|Introduce what FL is, how it differs from centralized ML, and its motivation (privacy, edge computation).|Opens the chapter clearly.|
|**Core Challenges**|Explain _why_ FL is difficult — sets up motivation for all later methods (non-IID, system heterogeneity, privacy, comms).|Perfect second — readers now know _what we’re solving_.|
|**Characterizing Data Heterogeneity**|Zooms in on the main challenge (non-IID data) and explains it conceptually.|Natural follow-up to challenges — it isolates the main problem your thesis focuses on.|
|**Modeling Non-IID Data with the Dirichlet Distribution**|Provides the _formal/statistical_ way to represent the challenge in experiments.|This is the bridge from concept → math. Excellent subsubsection placement.|
|**Taxonomy of FL**|Describes _how FL methods differ_ (horizontal, vertical, cross-silo/device, optimization approaches like FedAvg/FedProx).|Closing section that broadens the reader’s view again, showing the whole landscape.|

So yes — **putting the algorithms / taxonomy last** works perfectly:  
the reader first understands _what the problem is_, _how we model it_, and _finally how the community solves it_.

---

## ✍️ Recommended elaboration per section (for smooth writing)

### 4.1 Overview

- Define FL (decentralized ML where clients keep data local).
    
- Briefly contrast with centralized ML.
    
- Mention the server–client communication cycle (local training + global aggregation).
    
- Note real-world motivations: data privacy laws (GDPR), edge computing, device diversity.
    

### 4.2 Core Challenges

- **Statistical heterogeneity** → non-IID data (main focus).
    
- **System heterogeneity** → different hardware, participation frequency.
    
- **Privacy constraints** → cannot share raw data.
    
- **Communication bottlenecks** → bandwidth, limited rounds.
    

_(End with a small transition like: “Among these challenges, data heterogeneity is particularly significant, as it directly affects model convergence and generalization performance. We explore this further in the next section.”)_

### 4.3 Characterizing Data Heterogeneity

- Define IID formally (identical and independent distributions).
    
- Explain non-IID: clients sample from different local data distributions ( P_k(x, y) \neq P(x, y) ).
    
- Mention forms of heterogeneity:
    
    - **Label distribution skew** (most common)
        
    - **Feature distribution skew**
        
    - **Quantity skew**
        
- Connect this to how it affects learning (client drift, biased updates).
    

### 4.3.1 Modeling Non-IID Data with the Dirichlet Distribution

- Describe Dirichlet mathematically:  
    [  
    \mathbf{p}_k \sim \text{Dir}(\alpha)  
    ]  
    where ( \mathbf{p}_k ) represents the per-class data proportion for client ( k ).
    
- Explain how α controls heterogeneity:
    
    - α → ∞ = IID
        
    - α → 0 = strongly non-IID
        
- State that this is the standard way to simulate non-IID partitions in FL experiments (FedAvg, FedMatch, CBAFed, etc.).
    
- Add a sentence connecting to your experiments (“In this thesis, α = 0.3 is used to simulate moderate non-IID conditions.”)
    

### 4.4 Taxonomy of FL

- Broaden again to the _global picture_:
    
    - **Horizontal FL** (same features, different samples)
        
    - **Vertical FL** (different features, same samples)
        
    - **Federated transfer learning**
        
    - **Cross-device vs. cross-silo**
        
- Conclude with **optimization taxonomy**:
    
    - **FedAvg** (simple averaging)
        
    - **FedProx, Scaffold, FedNova** (handle heterogeneity and client drift)
        
- Optional: add a summary figure showing this taxonomy as a tree.
    
