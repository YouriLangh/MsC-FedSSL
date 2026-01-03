

### **Chapter 3 – Methodology

---

#### **3.1 Problem Statement**

As before — you define the federated semi-supervised setup, data splits, and the training objective.

---

#### **3.2 Methodology Overview**

A short conceptual introduction explaining what FixFed is and how it integrates FedAvg and FixMatch.  
_(You could also add a figure here illustrating the client–server setup and workflow.)_

> Example:
> 
> > This section introduces our proposed **FixFed** framework, which unifies Federated Averaging (FedAvg) and FixMatch into a cohesive Federated Semi-Supervised Learning (FSSL) approach. The framework retains the simplicity of both algorithms while enabling detailed analysis of their interaction under non-IID and label-scarce conditions.

---

#### **3.3 Federated Learning Component (FedAvg)**

Keep your explanation here — including your note about weighting by unlabeled data size.  
That’s an appropriate and clear “minor deviation” to explain here, not in the experiments section.

---

#### **3.4 Local Training Component (FixMatch)**

Keep as is — this section should explain the FixMatch procedure and how you adapted it locally for the federated setup.

---

#### **3.5 Extensions**

Keep your two subsections (STM, Adaptive Thresholding).

---

#### **3.6 FixFed Framework**

_(This replaces “Full Algorithm”)_  
Here, you describe the **overall framework** — i.e., how everything ties together — and show **Algorithm 1 (FixFed)**.

> Example intro:
> 
> > Bringing together the components discussed above, Algorithm \ref{alg:fixfed} summarizes the complete FixFed framework. Each communication round involves local FixMatch training on participating clients followed by global model aggregation through FedAvg.

Then include your pseudocode here.

---

#### **3.7 Discussion (optional)**

Briefly reflect on:

- Why this design is minimal yet representative of common FSSL setups.
    
- How it allows controlled evaluation of key factors (non-IID data, label scarcity, etc.).
    
- How it differs from prior works (if you want to highlight originality subtly).
    

---

## 🧩 Why this version works

- **“Methodology”** feels like a top-level framing term, fitting both the formalization and the algorithmic description.
    
- **“FixFed Framework”** as a subsection reads much better than “Full Algorithm” — it signals conceptual integration rather than just code.
    
- Your **minor modifications** (e.g., weighting by unlabeled dataset, STM updates, adaptive thresholds) are all logically placed where they matter:
    
    - Modifications to FedAvg → in the FedAvg section.
        
    - Modifications to FixMatch → in the FixMatch or Extensions section.
        

So you won’t need to repeat them again in Experiments — there you just _state how they’re applied and evaluated_.

---

Would you like me to show you how the _first page_ of that chapter (intro + transitions up to the FedAvg section) could look rewritten in polished thesis style? It’ll make the new structure feel coherent and professional.