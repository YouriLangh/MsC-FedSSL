
#SEMI-SL
`Known as SSL`

Information from [video](https://www.youtube.com/watch?v=C3Lr6Waw66g) and [video](https://www.youtube.com/watch?v=eYgPJ_7BkEw)
**Wrapper Method:**
Base model is trained on labeled dataset
Use this model to predict labels for the unlabeled dataset (pseudo-labels). These pseudo-labels come with a probability of how confident model is in the label (85% for example)
High confidence pseudo-labels are then combined with original labeled dataset --> re-train multiple times with all the data 

**Unsupervised Pre-Processing:**
This uses an autoencoder (represent each data in a more compact/meaningful way)
Then it does this for unlabeled, it extracts these features, to train the supervised model more

**Clustering-Based:**
Similar data are likely to belong to same class

**Active Learning:**
Bring humans in the loop, samples with low confidence pseudolabels can be given to human annotators

### Video 2:
To train the actual model, the loss consists of supervised loss + param * unsupervised loss.
L = L$_s$ + $\lambda$ L$_u$

**Approach:**
Run the weakly augmented data point through the model, we get a prediction. If p(y) > threshold --> pseudo-label made.
Make a strongly augmented version (hue, light intensity, contrast) --> through model, this gives a prediction
Pretend pseudo-label as ground truth label and calculate loss with prediction & pseudo-label.

Can also do nth for weak augmentation to get a simpler pipeline.
Model learns to revert/ignore the strong augmentation.
--> Seems to help

Kind of augmentation matters. FixMatch= 95% accuracy with 250 labelled examples from CIFAR. 4 labels per class (40 labelled images) --> 88.61% accuracy

They employ consistency regularization$^1$ & confidence threshold
They just combined 1 with pseudo-labelling

*What is consistency regularization?*
"Model should output similar predictions when fed perturbed versions of the same images"

FixMatch mentions alot of hyperparameters that are essential (not insane decrease on performance), but they used momentum sgd rather than AdaM, learning rate using cos function and stuff.
--> 1 Bad hyperparam (1 order of magnitude incorrect) like weight decay can cause 10% decrease in performance --> nitpicking results