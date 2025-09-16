#FL 

Not a big fan of the taxonomy, this paper can be used to investigate how other's tackle it. Though this paper does mention that SSL isa  future direction for FL!!
though the SSL is about "send your outputs, not your weights to server"

N also mentions how others divide their data for the project (Class 1 non-iid n so on.)

Some sources are highlighted if needed.


### 🟠 Class 1 non-IID (by label/class)

- **How it works:** Data is **first grouped by label** (e.g. all “0” digits together, all “1” digits together), and then **each client gets only one or a few specific classes**.
    
- **Effect:** Each client’s data is **very biased toward only certain classes**, and **class distributions are highly skewed** between clients.
    
- **Example:** MNIST digits are separated by their digit label, and each client only receives data from 1 or 2 digit classes.
    

✅ Creates **extreme heterogeneity** because clients see completely different classes.

---

### 🟠 Class 2 non-IID (partially mixed, partition-based)

- **How it works:** The data is **split into partitions or shards first**, and then **each client is given a few random shards**, where each shard mostly contains one class but possibly some mixture.
    
- **Effect:** Clients still have **uneven class distributions**, but **more mixed than Class 1**.
    
- **Example:** Data is sorted by class and divided into 20 partitions. Each client is randomly assigned 2 of those partitions.
    

✅ Creates **moderate heterogeneity**—clients may share some classes but with **different proportions**.

---

### ⚪ IID (Independent and Identically Distributed)

- **How it works:** The data is **shuffled randomly** and **evenly distributed** to all clients.
    
- **Effect:** Each client has **the same class distribution** as the overall dataset.
    
- **Example:** 60 000 MNIST samples are shuffled and evenly split into 100 clients, each getting 600 random samples.
    

✅ Creates **no heterogeneity**—every client has a **balanced and representative subset**.

---

### 📝 Summary Table

|Setting|Distribution Method|Class Balance per Client|Heterogeneity|
|---|---|---|---|
|**Class 1 non-IID**|Group by class → assign to clients|One/few classes per client|Very high|
|**Class 2 non-IID**|Partition into shards → assign|Skewed mixture of classes|Moderate|
|**IID**|Shuffle → distribute evenly|All classes evenly per client|None|

---

If you want, I can also **visualize these 3 distributions (Class 1 vs Class 2 vs IID) in a simple diagram** to make it clearer.  
Would you like me to do that?

![[A state-of-the-art survey on solving non-IID data in FL 1.pdf]]