
- **FedMatch (2022)**: combines FixMatch with FedAvg
    
- **FedProto (2021)**: uses prototype-based classification and clustering
    
- **Fed-ET (2021)**: ensembles local models for teacher and distills global student
    
- **FedSTAR (2023)**: does semi-supervised training with server-side unlabeled data augmentation

## Strategy 2: _FedSSL with Teacher-Student and Role Reversal_

1. **Clients train** local **teacher** models with their few labeled samples + pseudo-labeled unlabeled data (e.g., FixMatch).
    
2. **Send teacher weights** to the server.
    
3. **Server aggregates** these into a global teacher model.
    
4. **Server uses weak data** to train a global **student** model (i.e., mimic the global teacher with public data).
    
5. **Send student back** to clients as initialization or regularizer (via distillation loss).
    
6. Repeat.
    

This alternates **knowledge distillation** between strong but sparse (client) and weak but dense (server) supervision.