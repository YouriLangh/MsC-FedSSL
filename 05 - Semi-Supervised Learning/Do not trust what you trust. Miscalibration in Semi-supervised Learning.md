#SEMI-SL 

add a simple penalty term to prevent bad pseudolabels 
Attempts to calibrate SSL methods

### low-density separation between classes" mean?

It means that:

- The **regions in the input space where different classes meet (i.e., class boundaries)** are assumed to **have few data points** (low density).
    
- Conversely, **clusters of data points** (high-density regions) are assumed to **belong to the same class**.


They want to also enforce that the winner class in pseudo-label generation has to have a specific margin with the follow-up contenders -- i think?
![[Do not trust what you trust. Miscalibration in Semi-supervised Learning.pdf]]