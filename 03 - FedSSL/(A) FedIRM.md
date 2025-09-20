#FedSSL 

> Not read, just summarized

**FedIRM** is a federated semi-supervised learning (FSSL) approach designed for situations where **only a few clients have labeled data** and **most clients have only unlabeled data** (like hospitals without annotation budgets).

Here’s how it works in simple terms and where it fits in the FL pipeline:

---

## 🧠 Core idea

Normally, unlabeled clients in FSSL just do **consistency regularization** (making predictions stable under perturbations), but without labels, they can **drift away from the actual task** and forget what they are supposed to learn.

FedIRM solves this by:

- **Extracting “disease relationship” information from labeled clients** (how classes relate to each other in feature space)
    
- **Forcing unlabeled clients to learn similar class relationships** while doing their own consistency-based training
    

This extra alignment keeps unlabeled clients focused on the task.

---

## ⚙️ How it fits in the FL process

|Step|What happens in FedIRM|
|---|---|
|**Local training at labeled clients**|Train normally with **cross-entropy loss** on labeled data. Also compute a **relation matrix** from features showing how classes relate.|
|**Local training at unlabeled clients**|Use **consistency regularization** on unlabeled data **+ an extra IRM loss** that forces their relation matrix to match the one from labeled clients.|
|**Aggregation at server**|The server does **normal FedAvg aggregation** (just averages local model weights). It also **sends the averaged relation matrix from labeled clients** to the unlabeled clients for their next round.|

**So, FedIRM only changes the local training step — the aggregation remains standard FedAvg.**

---

## 📌 Summary

- ✅ Alters **local training objectives**
    
    - Labeled: `cross-entropy`
        
    - Unlabeled: `consistency + inter-client relation matching`
        
- ✅ Adds knowledge sharing (relation matrix) from labeled → unlabeled clients
    
- ❌ Does **not** alter the aggregation step
    
