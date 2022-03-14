# From cat to dog

#### *A short and simple motivating example*

Suppose we have a sample of cats and dogs with information about two features: height and tail length. Based on these two features we have trained two black box classifiers to distinguish between cats and dogs: firstly, an artificial neural network with weight regularization and secondly, that same neural network but its Bayesian counterpart with [Laplace approximation](https://towardsdatascience.com/go-deep-but-also-go-bayesian-ab25efa6f7b) ([Figure 1](#fig-predictive) below). One individual cat – let’s call her Kitty 🐱 – is friends with a lot of cool dogs and wants to be part of that group. Let’s see how we can generate counterfactual paths for her.

![Figure 1: Classification for toy dataset of cats and dogs. The contour indicates confidence in predicted labels. Left: MLP with weight regularization. Right: That same MLP, but with Laplace approximation for posterior predictive.](www/predictive.png)

### From basic principles …

Counterfactual search happens in the feature space: we are interested in understanding how we need to change 🐱’s attributes in order to change the output of the black-box classifier. We will start with the first model, that relies on simple plugin estimates to produce its predictions. The model was pre-trained using Flux.jl and can be loaded as follows:

In order to make the Flux.jl model compatible with CounterfactualExplanations.jl we need to run the following:

Let `x̅` be the 2D-feature vector describing Kitty 🐱. Based on those features she is currently labelled as `y̅ = 0.0`. We have set the target label to `1.0` and the desired confidence in the prediction to `γ = 0.75`. Now we can use the generic counterfactual generator as follows:

The resulting counterfactual path is shown in [Figure 2](#fig-recourse-mlp) below. We can see that 🐱 travels through the feature space until she reaches a destination where the black-box model predicts that with a probability of \>75% she is actually a dog. Her counterfactual self is in the target class so the algorithmic recourse objective is satisfied. We have also gained an intuitive understanding of how the black-model arrives at its decisions: increasing height and decreasing tail length both raise the predicted probability that 🐱 is actually a dog.

![Figure 2: Classification for toy dataset of cats and dogs. The contour indicates confidence in predicted labels. Left: MLP with weight regularization. Right: That same MLP, but with Laplace approximation for posterior predictive.](www/recourse_mlp.gif)

### … towards realistic counterfactuals.

The generic search above yielded a counterfactual sample that is still quite distinct from all other individuals in the target class. While we successfully fooled the black-box model, a human might look at 🐱’s counterfactual self and get a little suspicious. One of the requirements for algorithmic recourse is that counterfactuals are realistic and unambigous. A straight-forward way to meet this requirement is to generate counterfactuals by implicitly minimizing predictive uncertainty. The plugin estimator underlying the black-box model above does not incorporate uncertainty: even in areas free of any data the neural network predicts labels with high confidence. Its Bayesian counterpart is much more conservative. By maximizing the predicted probability of the Bayesian model in [Figure 3](#fig-recourse-laplace) below, we implicitly minimize the predictive uncertainty around the counterfactual (Schut et al. 2021). This way we end up generating a counterfactual that looks more like the individuals in the target class and is therefore more realistic.

![Figure 3: Classification for toy dataset of cats and dogs. The contour indicates confidence in predicted labels. Left: MLP with weight regularization. Right: That same MLP, but with Laplace approximation for posterior predictive.](www/recourse_laplace.gif)

### References

Schut, Lisa, Oscar Key, Rory Mc Grath, Luca Costabello, Bogdan Sacaleanu, Yarin Gal, et al. 2021. “Generating Interpretable Counterfactual Explanations by Implicit Minimisation of Epistemic and Aleatoric Uncertainties.” In *International Conference on Artificial Intelligence and Statistics*, 1756–64. PMLR.
