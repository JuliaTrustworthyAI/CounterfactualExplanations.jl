**Title**: Explaining black-box models through counterfactuals
**Type**: Talk (30 minutes)
**Track**: JuliaCon

## Abstract

We propose [`CounterfactualExplanations.jl`](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/): a package for explaining black-box models through counterfactuals. Counterfactual explanations are based around the simple idea of strategically perturbing model inputs to change model predictions. Our package is novel, very accessible and designed to be extensible. It can be used to explain custom predictive models including those developed and trained in other programming languages.

## Description

### The Need for Explainability ⬛

Machine learning models like deep neural networks have become so complex, opaque and underspecified in the data that they are generally considered as black boxes. Nonetheless, such models often play a key role in data-driven decision-making systems. This often creates the following problem: human operators in charge of such systems have to rely on them blindly, while those individuals subject to them generally have no way of challenging an undesirable outcome:

> “You cannot appeal to (algorithms). They do not listen. Nor do they bend.”
> — Cathy O'Neil in *Weapons of Math Destruction*, 2016

### Enter: Counterfactual Explanations 🔮

Counterfactual Explanations can help human stakeholders make sense of the systems they develop, use or endure: they explain how inputs into a system need to change for it to produce different decisions. Explainability benefits internal as well as external quality assurance. Explanations that involve realistic and actionable changes can be used for the purpose of algorithmic recourse (AR): they offer human stakeholders a way to not only understand the system's behaviour, but also react to it or adjust it. Counterfactual Explanations have certain advantages over related tools for explainable artificial intelligence (XAI) like surrogate eplainers (LIME and SHAP). These include:

- Full fidelity to the black-box model, since no proxy is involved. 
- No need for (reasonably) interpretable features as opposed to LIME and SHAP.
- Clear link to Causal Inference and Bayesian Machine Learning.
- Less susceptible to adversarial attacks than LIME and SHAP.
### Problem: Limited Availability in Julia Ecosystem 😔

Software development in the space of XAI has largely focused on various global methods and surrogate explainers with implementations available for both Python and R. In the Julia space we have only been able to identify one package that falls into the broader scope of XAI, namely [`ShapML.jl`](https://github.com/nredell/ShapML.jl). Support for Counterfactual Explanations has so far not been implemented in Julia. 
### Solution: `CounterfactualExplanations.jl` 🎉

Through this project we aim to close that gap and thereby contribute to broader community efforts towards explainable AI. Highlights of our new package include:

- **Simple and intuitive interface** to generate counterfactual explanations for differentiable classification models trained in Julia.
- **Detailed documentation** providing examples involving linear classifiers and deep learning models trained in [Flux](https://fluxml.ai/) for binary and multi-class prediction tasks.
- **Easily extensible** through custom models and counterfactual generators: a carefully designed package architecture allows for seamless extension through multiple dispatch (see [here](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/tutorials/models/)). 
- **Interoperability** with other popular programming languages as demonstrated through examples involving deep learning models trained in `Python` and `R` (see [here](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/tutorials/interop/)).

### Ambitions for the Package 🎯

Our goal is to provide a go-to place for counterfactual explanations in Julia. To this end, the following is a non-exhaustive list of features we would like to add in the future:

1. Additional counterfactual generators.
2. Additional predictive models.
3. More examples to be added to the documentation.
4. Native support for categorical features.
5. Support for regression models.

The package is designed to be extensible, which should facilitate contributions through the community.
### Further Resources 📚

For some additional colour you may find the following resources helpful:

- [Blog post](https://towardsdatascience.com/individual-recourse-for-black-box-models-5e9ed1e4b4cc) and [motivating example](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/cats_dogs/).
- [GitHub repo](https://github.com/pat-alt/CounterfactualExplanations.jl) and [package docs](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/).

## Notes

The package has been developed by Patrick during the first few months of his PhD in Trustworthy Artificial Intelligence at Delft University of Technology. Its first version has been registered on the General registry for a few weeks, but updates can be expected ahead of JuliaCon as Patrick continues to develop the package for his own research. Below is a list of links pointing to additional resources related to the package including the source code, its documentation, a draft proceedings paper and more:

- [GitHub repo](https://github.com/pat-alt/CounterfactualExplanations.jl)
- Docs: [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pat-alt.github.io/CounterfactualExplanations.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/CounterfactualExplanations.jl/dev)
- [Draft paper](https://github.com/pat-alt/CounterfactualExplanations.jl/blob/paper/paper/paper.pdf)
- [Poster]()
- [Preliminary slides](https://github.com/pat-alt/CounterfactualExplanations.jl/tree/dev/dev/presentation/juliacon.html)
- [Preliminary video presentation]()

We would love to present this work at JuliaCon 22 for the following reasons:

1. **We think that our package is a timely and valuable contribution to the Julia community**. Explainable AI tools are crucial for (re-)building trust in machine learning and counterfactual explanations are among the most promising approaches. To the best of our knowledge our package offers its first implementation in Julia.
2. Support for other programming languages also makes this package useful for the broader community working on explainable AI community. **Explaining models programmed in `Python` and `R` through a pure-Julia package is a striking example of Julia's unique support for language interoperability**. We hope that this may help draw attention to Julia from the broader programming community. 
3. **We are looking for challenge and support from the community**. JuliaCon 22 is a unique opportunity to create awareness. 

Regarding the third point it is worth mentioning that Patrick has only recently moved to Julia following years of programming in R, Python, C++ and MATLAB. While every effort has been made to follow Julian best practices and develop an accessible and extensible package, there is certainly scope for improvement and guidance from more experienced Julians. To this end, we are looking for constructive criticism and also contributors that can help us develop this package further in the future. 

Finally, just a quick note that for the session image we have chosen an animated GIF. Should this not work for you, we can send a static PNG instead. 

Thank you very much for your time and consideration!

