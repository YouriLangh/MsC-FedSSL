
### **A. FixMatch**

- Rigid confidence threshold → **biased pseudo-labels**
    
- Confirmation bias → **error amplification**
    
- Low data utilization in early rounds
    

### **B. FedAvg**

- Statistical heterogeneity → **client drift**
    
- Many local epochs and much unlabeled data → **catastrophic forgetting**
    
- Inability to maintain consistent representations across clients
    

### **C. Their combination**

- A naïve FixMatch+FedAvg pipeline **cannot handle**:
    
    - non-IID distributions
        
    - label scarcity
        
    - inconsistent client updates
        

This leads directly to **poor representation learning** and **severe performance degradation**, especially under high heterogeneity.