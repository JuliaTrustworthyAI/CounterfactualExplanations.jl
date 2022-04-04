**Title**: Explaining black-box models through counterfactuals
**Type**: Talk (30 minutes)
**Track**: JuliaCon

### Abstract

We propose [`CounterfactualExplanations.jl`](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/): a package for explaining black-box models through counterfactuals. Counterfactual explanations are based around the simple idea of strategically perturbing model inputs to change model predictions. Our package is novel, very accessible and designed to be extensible. It can be used to explain custom predictive models including those developed and trained in other programming languages.

### Description

Machine learning models like deep neural networks have become so complex, opaque and underspecified in the data that they are generally considered as black boxes. Nonetheless, such models often play a key role in data-driven decision-making systems. This often creates the following problem: human operators in charge of such systems have to rely on them blindly, while those individuals subject to them generally have no way of challenging an undesirable outcome:

> “You cannot appeal to (algorithms). They do not listen. Nor do they bend.”
> — Cathy O'Neil in *Weapons of Math Destruction*, 2016

Counterfactual explanations can help programmers make sense of the systems they build: they explain how inputs into the system need to change for it to produce different decisions. Explanations that involve realistic and actionable changes can be used for the purpose of algorithmic recourse: they offer individuals subject to algorithms a way to turn a negative decision into a positive one. Through `CounterfactualExplanations.jl` we make these recent and promising approaches to explainable artificial intelligence (XAI) available to the Julia community.

### What we offer ⭐⭐⭐

At this stage the package highlights include:

- Native support for gradient-based counterfactual generation for differentiable, predictive classification models trained in Julia.
- The documentation includes detailed examples involving linear classifiers and deep learning models trained in [Flux](https://fluxml.ai/) for binary and multi-class prediction tasks.
- Support for custom models and counterfactual generators: a carefully designed package architecture allows for seamless extension through multiple dispatch. Guidance for this can be found [here](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/tutorials/models/). 
- The package can also be used to explain models trained in other programming languages with ease. Examples involving deep learning models trained in `Python` 🐍 and `R` can be found [here](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/tutorials/interop/).

To get started you may find the following links useful:

- [Blog post](https://towardsdatascience.com/individual-recourse-for-black-box-models-5e9ed1e4b4cc) and [motivating example](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/cats_dogs/).
- [GitHub repo](https://github.com/pat-alt/CounterfactualExplanations.jl) and [package docs](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/).

### Related work

Explainable AI is a relatively young field of research made up of a variety of subdomains, definitions and taxonomies. We will not cover all of them here, but it is worth mentioning some high-level concepts. The first broad distinction we want to make here is between **interpretable** and **explainable** AI. These terms are often used interchangeably, but this can cause confusion. We find the following distinction useful: interpretable AI involves models that are inherently interpretable and transparent such as general additive models (GAM), decision trees and rule-based models; explainable AI involves models that are not inherently interpretable, but require additional tools to be explained to humans. 

Some would argue that we best avoid the second category of models and instead focus solely on interpretable AI. While we agree that initial efforts should always be geared towards interpretable models, it is difficult to see people giving up on black box models altogether. For that reason, we expect the need for explainable AI to persist in the near future. The field can further be broadly divided into **global** and **local** explainability: the former is concerned with explaining the average behavior of a model, while the latter involves explanations for individual predictions. Counterfactual explanations (along with other popular tools like LIME and SHAP) fall into the category of local methods: they explain how individual predictions change in response to individual feature perturbations.  

Software development in the space of XAI has largely focused on various global methods and surrogate explainers with various implementations available for both Python and R. In the Julia space we have only been able to identify one package that falls into the broader scope of XAI, namely [`ShapML.jl`](https://github.com/nredell/ShapML.jl), which provides a fast implementation of SHAP. Arguably the current availability of tools for explaining black-box models in Julia is limited, but it appears that the community is invested in changing that. The team behind `MLJ.jl`, for example, is currently recruiting contributors for a project about interpretable and explainable AI. Through our work on counterfactual explanations we hope to contribute to these broader efforts. Through its unique transparency Julia naturally lends itself towards building greater trust in machine learning.

### Notes

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

