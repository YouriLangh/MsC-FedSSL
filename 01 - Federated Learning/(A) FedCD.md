#FL #Non-IID

Main downside i see already is that each device required validation data which can be expensive in practice & extremely low labeled data scenarios.


Basically, have a model, this one gets averaged and updated, once you hit a cloning round, then certain clients will only update or mostly update a specific global models. So idea is that clients with similar data will update the same model --> specialized model for each client/data archetype.
> But how to choose which clone? When the server **clones a model**, the **weights are the same**, **but the scores are intentionally made different** on each device:
> New clone’s initial score = 1−(score of parent model)
> This forces devices to **not treat the clones as identical** from the start.



CIFAR10 Archetypes:

On the CIFAR-10 dataset, the researchers defined **archetypes** to represent different non-IID data distributions across devices, which are themselves IID subsets of the overall non-IID data. Two distinct setups for these archetypes were used in the experiments:

1. **Hierarchical Archetypes**:

    ◦ This setup involved **10 archetypes** nested within **two "meta-archetypes"**.

    ◦ One meta-archetype consisted of devices with data primarily associated with labels **0, 1, 2, 3, and 4**.

    ◦ The second meta-archetype included devices with data primarily associated with labels **5, 6, 7, 8, and 9**.

    ◦ Each of the 10 archetypes was represented by one of these labels. For instance, a device belonging to "meta-archetype 1" (labels 0-4) would only have access to training examples with those specific labels.

    ◦ The data distribution for devices within these archetypes was controlled by a **bias (b), which was set to uniform between 0.6 and 0.7**. This bias indicated the fraction of a device's local dataset composed of examples whose labels matched its specific archetype. The remaining fraction of data was equally distributed among the other labels within that device's meta-archetype. For example, a device with archetype 3, using 5k training images, would have `b * 5k` images with label 3, and `(1-b)/4 * 5k` images each for labels 0, 1, 2, and 4.

    ◦ There were 3 devices per archetype in this experiment.

2. **Hypergeometric Archetypes**:

    ◦ This setup explored more complex scenarios and featured **6 archetypes**.

    ◦ For these archetypes, each device sampled labeled training examples from a **hypergeometric distribution** over the 10 CIFAR-10 labels.

    ◦ The parameters used for the hypergeometric distribution were N = 110, n = 10, and K ∈ {5, 25, 45, 65, 85, 105}. These different K values corresponded to the distinct archetypes.

    ◦ The hypergeometric distribution allowed for varying "skewed probability distributions" across the archetypes. Some archetypes, like 0 and 5, had more skewed distributions (differing most from the global distribution), while "central archetypes," such as 2 and 3, had distributions more similar to the global distribution.

    ◦ There were 5 devices per archetype in this experiment.
    
 *For instance, in a next-word prediction application, devices belonging to users of a particular age group within a country could form an archetype, as they might share common vocabulary and usage patterns, making their data similar. Devices that belong to the same archetype will preferentially update the same global model.*

The **FedCD (Federated Cloning-and-Deletion) pipeline** is a novel approach designed to improve the performance of federated learning on non-IID data by dynamically grouping devices with similar data into these archetypes. It works as follows:

1. **Initialisation and Milestone Cloning:** The process begins with a single global model on a centralised server. At specific "milestone" rounds, all existing global models on the central server are **cloned and compressed**. This iterative cloning encourages the exploration of different model specialisations.

2. **Local Training and Scoring:** In each training round, a random subset of participating devices train their local models (specifically, those models for which they have a positive score) on their own local data for a set number of epochs. After training, each device **evaluates how well each model performs** on its local validation data. Based on this performance, the device assigns a "score" to each model. This score is typically a normalised average of the model's validation accuracy over the most recent rounds.

3. **Weighted Model Updates:** Devices then send their weight updates and these model scores to the central server. The server takes a **weighted average of the weight updates** from all devices for each global model, with the weights determined by the scores assigned by the devices to that particular model. This means models that perform well on a device's data have a greater influence on that model's global update.

4. **Re-deployment and Model Deletion:** The updated global models are then re-deployed to the appropriate edge devices. Crucially, FedCD incorporates a **deletion mechanism** to manage the number of models and ensure efficiency. Devices delete models that are performing poorly on their local data (e.g., if a model's score is significantly lower than the device's highest-performing model). Furthermore, if a model receives a score of zero from all devices, it is deleted from the central server. This aggressive deletion prevents an "explosion" in storage requirements and helps devices converge on a few high-performing models tailored to their data distribution.

5. **Specialisation and Convergence:** Through this iterative process of cloning, adaptive updating based on model scores, and deletion of poor performers, FedCD allows devices to **self-select into groups with similar data (archetypes)**. This leads to the development of **specialised models for each archetype**, resulting in faster convergence and higher accuracy compared to traditional methods like FedAvg, especially when dealing with non-IID data. The use of compression via quantization also helps manage the memory overhead of maintaining multiple models on devices.

### 🧠 What FedCD Does

FedCD is a **federated learning method for highly non-IID data**.  
Instead of training **one global model** like FedAvg, it lets **multiple specialized models emerge**, each serving a group of devices with similar data (called **archetypes**).

---

### ⚙ How It Works (Simple Steps)

1. **Start with one model** on the server.
    
2. **At certain rounds, clone it** → make multiple global models.
    
3. **Devices train and score all models** on their local (labeled) validation data.
    
4. **Server updates each model** using a **weighted average of updates**, where weights are the devices’ scores (better models for a device influence more).
    
5. **Delete weak models**: devices drop models that score badly, and the server deletes models that nobody scores.
    
6. Over time, **devices group themselves** around the model that works best for them → forming **archetype-specialized models**.


**Next training rounds:**

- Device A’s data fits better with model 1 → it gives model 1 a slightly higher score, so its update contributes more to model 1.
    
- Device B prefers model 2 → its updates push model 2 more.
    
- Models 3 and 4 get very few or no updates.

**Because updates are weighted by scores**, the models quickly **specialize toward the devices that favor them**.
Devices stop updating models that perform poorly on their validation set (score ≈ 0), so those models eventually get **deleted**.



![[FedCD. Improving Performance in non-IID Federated Learning.pdf]]