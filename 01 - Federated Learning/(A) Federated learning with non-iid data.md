
#FL 

this approach suggests sharing some data between all clients to improve performance

Contains alot of good info for training hyperparams. Basically says that the loss is due to model weight divergence.


At the initialization stage of FedAvg, the warm-up model trained on G and a random $\alpha$
portion of G are distributed to each client. The local model of each client is trained on the shared
data from G together with the private data from each client. The cloud then aggregates the local models from the clients to train a global model with FedAvg.


![[Federated learning with non-iid data.pdf]]