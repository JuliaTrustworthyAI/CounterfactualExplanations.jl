
## Latent space search

Latent space search is implemented by recourse generators like REVISE and CLUE. Searching counterfactuals in a latent space has at least two benefits:

1. The latent space may be lower dimensional, therefore making gradient search computationally less expensive.
2. Traversing through the latent space implicitly ensures that we "sample the set of high probability paths of changes that are close to the original attributes".

Searching in a latent space can therefore be expected to improve the quality of counterfactuals generally and independent of additional constraints placed on the search objective. For example, it is plausible that the quality of counterfactuals returned when using the DICE objective (diverse counterfactuals) may additionally benefit from searching a latent space. 

The same narrative applied to counterfactual generation for Bayesian classifiers. As Schut et al. have shown, counterfactuals generated for Bayesian classifiers tend to be more realistic. But both generic and greedy search can be applied. Similarly, it could be interesting to see how DICE performs for Bayesian classifiers. 

How can we achieve this?

*Idea 💡*

- Counterfactual generators that search a latent space are not actually different from counterfactual generators that search the feaeture space. 
- That is to say that the actual counterfactual search objective does not change: for example, the objective function used in REVISE [@joshi2019towards] is equivalent to the one originally proposed by Wachter. 
- The only difference is that it is evaluated with respect to $\hat{x}=d(e(x))$ where $e:\mathcal{X}\mapsto\mathcal{Z}$ and $d:\mathcal{Z}\mapsto\mathcal{X}$.
- This means that any modifications of the counterfactual search objective are also applicable to search in the latent space. 
- Using this observation we can combine the benefits of searching in the latent space with benefits associated with different counterfactual search objectives. 

*Implementation*

- Express the counterfactual state variable as a function $f$ of $s_t$ where is the feature that will actually be perturbed. 
- For search in the feature space we just have $f(s_t)=s_t$ where $s_0=x$. For search in the latent space we have $f(s_t)=d(s_t)$ where $s_0=e(x)$.
    - Gradients can then just always be computed with respect to $s$.
    - Mutability constraints can also be implemented with respect to $s$, since $\Delta f(s)=0$ for $\Delta s=0$.
    - Domain constraints may be more difficult, but this can be parked for now. 