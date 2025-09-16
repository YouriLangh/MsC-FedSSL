#FL 

Basically it allows worse devices to also contribute by sending partial updates to the server. 
The second part of FedProx is that it doesnt simply optimize the loss function, rather it adds a proximal term which adds $\lambda / 2$ $||w-w^t||^2$ which somehow improves the statistical heterogeneity.


In FedProx, the **proximal term** is an addition to the local objective function that each device minimises during federated learning, particularly relevant for networks with non-identically distributed (non-IID) data.

In simple terms:

• **The Problem (Non-IID Data):** In federated learning, each device often holds data that is statistically different from other devices (non-IID). If devices update their models too aggressively based only on their unique local data, their individual models can "drift" significantly from what would be optimal for the overall global model. This drift can lead to unstable convergence or even divergence of the entire learning process.

• **The FedProx Solution (Proximal Term):** FedProx introduces a **proximal term** into the local objective function that each device minimises. This term is `µ/2 ||w - wt||^2`, where:

    ◦ `w` represents the local model that the device is currently trying to update.

    ◦ `wt` represents the global model received from the central server at the beginning of the current communication round.

    ◦ `µ` is a penalty constant that determines how strongly the local model is pulled back towards the global model.

Essentially, the proximal term acts like a **"gravitational pull"** or a **"regularizer"** for the local updates. It penalises devices if their local model (`w`) deviates too far from the current global model (`wt`).



![[FedProx.pdf]]