**Title**: Explaining Black-Box Models through Counterfactuals
**Type**: Talk (30 minutes)
**Track**: JuliaCon

## Abstract

We propose [`CounterfactualExplanations.jl`](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/): a package for explaining black-box models through counterfactuals. Counterfactual explanations are based on the simple idea of strategically perturbing model inputs to change model predictions. Our package is novel, very accessible and designed to be extensible. It can be used to explain custom predictive models including those developed and trained in other programming languages.

## Description

### The Need for Explainability ⬛

Machine learning models like deep neural networks have become so complex, opaque and underspecified in the data that they are generally considered as black boxes. Nonetheless, they often form the basis for data-driven decision-making systems. This creates the following problem: human operators in charge of such systems have to rely on them blindly, while those individuals subject to them generally have no way of challenging an undesirable outcome:

> “You cannot appeal to (algorithms). They do not listen. Nor do they bend.”
> — Cathy O'Neil in *Weapons of Math Destruction*, 2016

### Enter: Counterfactual Explanations 🔮

Counterfactual Explanations can help human stakeholders make sense of the systems they develop, use or endure: they explain how inputs into a system need to change for it to produce different decisions. Explainability benefits internal as well as external quality assurance. Explanations that involve realistic and actionable changes can be used for the purpose of algorithmic recourse (AR): they offer human stakeholders a way to not only understand the system's behaviour, but also strategically react to it. Counterfactual Explanations have certain advantages over related tools for explainable artificial intelligence (XAI) like surrogate eplainers (LIME and SHAP). These include:

- Full fidelity to the black-box model, since no proxy is involved. 
- Clear link to Causal Inference and Bayesian Machine Learning.
- No need for (reasonably) interpretable features.
- Less susceptible to adversarial attacks than LIME and SHAP.

### Problem: Limited Availability in Julia Ecosystem 😔

Software development in the space of XAI has largely focused on various global methods and surrogate explainers with implementations available for both Python and R. In the Julia space we have only been able to identify one package that falls into the broader scope of XAI, namely [`ShapML.jl`](https://github.com/nredell/ShapML.jl). Support for Counterfactual Explanations has so far not been implemented in Julia. 

### Solution: `CounterfactualExplanations.jl` 🎉

Through this project we aim to close that gap and thereby contribute to broader community efforts towards explainable AI. Highlights of our new package include:

- **Simple and intuitive interface** to generate counterfactual explanations for differentiable classification models trained in Julia.
- **Detailed documentation** involving illustrative example datasets, linear classifiers and deep learning models trained in [Flux](https://fluxml.ai/) and counterfactual generators for binary and multi-class prediction tasks.
- **Interoperability** with other popular programming languages as demonstrated through examples involving deep learning models trained in Python and R (see [here](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/tutorials/interop/)).
- **Seamless extensibility** through custom models and counterfactual generators (see [here](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/tutorials/models/)). 

### Ambitions for the Package 🎯

Our goal is to provide a go-to place for counterfactual explanations in Julia. To this end, the following is a non-exhaustive list of exciting feature developments we envision:

1. Additional counterfactual generators and predictive models.
2. Additional datasets for testing, evaluation and benchmarking.
3. Improved preprocessing including native support for categorical features.
4. Support for regression models.

The package is designed to be extensible, which should facilitate contributions through the community.
### Further Resources 📚

For some additional colour you may find the following resources helpful:

- [Blog post](https://towardsdatascience.com/individual-recourse-for-black-box-models-5e9ed1e4b4cc) and [motivating example](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/cats_dogs/).
- Package docs: [[stable]](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/stable), [[dev]](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/dev).
- [GitHub repo](https://github.com/juliatrustworthyai/CounterfactualExplanations.jl).

## Notes

The package has been developed by [presenter] during the first few months of their PhD in Trustworthy Artificial Intelligence at Delft University of Technology. It has already been registered on the General registry, but updates can be expected ahead of JuliaCon and beyond as [presenter] continues to develop the package for their own research. 

Below is a list of links pointing to additional resources related to the package:

- [Companion Paper](https://github.com/juliatrustworthyai/CounterfactualExplanations.jl/tree/paper) for JuliaCon proceedings (draft)
- [GitHub repo](https://github.com/juliatrustworthyai/CounterfactualExplanations.jl)
- Package docs: [[stable]](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/stable), [[dev]](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/dev)

We would love to present this work at JuliaCon 22 for the following reasons:

1. **We think that our package is a timely and valuable contribution to the Julia community**. Explainable AI tools are crucial for (re-)building trust in machine learning and counterfactual explanations are among the most promising approaches. To the best of our knowledge our package offers its first implementation in Julia.
2. Support for other programming languages also makes this package useful for the broader community working on explainable AI community. **Explaining models programmed in Python and R through a pure-Julia package is a striking example of Julia's unique support for language interoperability**. We hope that this may help draw attention to Julia from the broader programming community. 
3. **We are looking for challenge and support from the community**. JuliaCon 22 is a unique opportunity to create awareness. 

Regarding the third point it is worth mentioning that [presenter] has only recently moved to Julia following years of programming in R, Python, C++ and MATLAB. While every effort has been made to follow Julian best practices and develop an accessible and extensible package, there is certainly scope for improvement and guidance from more experienced Julians. To this end, we are looking for constructive criticism and also contributors that can help us develop this package further in the future. 

Finally, just a quick note that for the session image we have chosen an animated GIF. Should this not work for you, we can send a static PNG instead. 

Thank you very much for your time and consideration!

