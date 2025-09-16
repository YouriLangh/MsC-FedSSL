#FL 

Federated Optimization has the following properties:
- Non-iid data
- unbalanced data (some users use more than others)
- Massively distributed
- Limited communication.
We only use first two


![[Federated Learning Original - Rounds till Accuracy.png]]

Batch-size = 10, more epochs (20) is best speed up in terms of communications rounds to reach a certain acc.

Their models use learning rate decay, they start at like 0.05-0.25 and decay with 0.99

Basically this paper says that you still need like 200+ rounds and calls this "few" rounds.

![[Federated_Learning_Original.pdf]]


###  Federated Learning (FL) — The General Framework

Federated Learning is **a general machine learning paradigm** where:

- Multiple **clients (or participants)** each hold their **own local data**.
    
- A **coordinating entity (often called a server)** orchestrates the learning process.
    
- The goal is to **train a global model collaboratively without sharing raw data**.
    

Key ideas:

- Clients **train locally** on their own data.
    
- Some form of **model update or knowledge (not data)** is sent to the server.
    
- The server **aggregates** those updates and distributes the new global model back.
    

🔑 Important:

- This is the **framework/paradigm** of FL.
    
- It **does not prescribe** _how_ updates are aggregated or even if there is a central server — just that there is some mechanism to combine local learning.
    

---

### ⚖️ FedAvg — A Specific Algorithm _Within_ Federated Learning

**Federated Averaging (FedAvg)** is the **canonical algorithm** introduced by McMahan et al. (2017) and is the most widely used approach to implement FL.

FedAvg adds **specific rules** to the general FL framework:

- The **server sends the current global model** to a sample of clients.
    
- Each client **runs several local SGD steps** on their local data.
    
- Clients send their **updated model weights (or gradients)** back to the server.
    
- The **server averages** these updates **weighted by the number of samples** each client used.
    

So:

- FedAvg uses a **central server** and **weight averaging** as the aggregation rule.
    
- But other FL algorithms can use **different aggregation rules** (e.g. FedProx, FedNova, Scaffold) or **no central server at all** (peer-to-peer FL, decentralized FL).
    

---

### 📌 Summary

|Concept|Description|Server?|Aggregation?|
|---|---|---|---|
|**Federated Learning**|General framework for collaborative training on decentralized data|Optional (usually yes)|Any method|
|**FedAvg**|Specific FL algorithm using local SGD + weight averaging|Yes (central)|Weighted average of model weights|

So to answer directly:

> is the framework in federated learning all about collaborative learning and a central server that aggregates the weights or is that a framework in FedAvg?

✅ **Collaborative learning on private data = FL framework**  
✅ **Central server + averaging weights = FedAvg (a method within FL)**

---

If you want, I can also give you a **diagram showing how FL in general differs from FedAvg specifically** — would you like me to do that?