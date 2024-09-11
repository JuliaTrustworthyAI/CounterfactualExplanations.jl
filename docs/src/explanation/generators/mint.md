

``` @meta
CurrentModule = CounterfactualExplanations 
```

# MINT Generator

In this tutorial, we introduce the MINT generator, a counterfactual generator based on the Recourse through Minimal
Intervention (MINT) method proposed by Karimi, Schölkopf, and Valera (2021).

!!! note
    There is currently no custom type for this generator, because we anticipate changes to the API for composable generators. This tutorial explains how counterfactuals can nonetheless be generated consistently with the framework proposed by @karimi2021algorithmic

## Description

The MINT generator incorporates causal reasoning in algorithm recourse to achieve minimal interventions when generating a counterfactual explanation. In this sense, the main ideia is that just perturbating a black box model without taking into account the causal relations in the data can guide to misleading recommendations. Here we now shift to a perspective where every action/pertubation is an intervetion in the causal graph of the problem, thus the change is not made just in the intervened upon variable, but also in its childs in the causal structure. The generator utilizes a Structural Causal Model(SCM) to encode the variables in a way that causal effects are propagated and uses a generic gradient-based generator to create the search path, that is, any gradient-base generator (ECCo, REVISE, Watcher, …) can be used with the MNIT SCM encoder to generate counterfactual samples in latent space for minimal intervetions algorithm recourse.

The MNIT algorithm minimizes a loss function that combines the causal constraints of the SCM and the distance between the generated counterfactual and the original input. Since we want a gradient-based generator, we need to pass the constrained optimizaiton problem into an unconstrained one and we do this by using the Lagrangian. Initially, as defined in Karimi, Schölkopf, and Valera (2021), we aim to aim to find the minimal cost set of actions $A$ (in the form of structural interventions) that results in a counterfactual instance yielding the favorable output from $h$,

``` math
\begin{aligned}
A^* \in \arg\min_A \text{cost}(A; \mathbf{x}_F)\\
\textrm{s.t.} \quad & h(\mathbf{x}_{SCF}) \neq h(\mathbf{x}_F)\\
\end{aligned}
```

where $\mathbf{x}_F$ is the original input, $\mathbf{x}_{SCF}$ is the counterfactual instance, and $h$ is the black-box model. We use the $\mathbf{x}_{SCF}$ terminology because the counterfactual is derived from the SCM,

``` math
\begin{equation}

x_{SCF_i} = 
\begin{cases}
x_{F_i} + \delta_i, & \text{if } i \in I \\
x_{F_i} + f_i(\text{pa}_{SCF_i}) - f_i(\text{pa}_{F_i}), & \text{if } i \notin I  \; \; \text{,}
\end{cases} 

\end{equation}
```

where $I$ is the set of intervened upon variables, $f_i$ is the function that generates the value of the variable $i$ given its parents, and $\text{pa}_{SCF_i}$ and $\text{pa}_{F_i}$ are the parents of the variable $i$ in the counterfactual and original instance, respectively. This closed formula for the decision variable $\mathbf{x}_{SCF}$ is what makes possible to use a gradient-based generator, since the lagrangian is differentiable,

``` math
\begin{equation}
\mathcal{L_{\texttt{MINT}}}(\mathbf{x}_{SCF}) = \lambda \text{cost}(\mathbf{x}_{SCF}; \mathbf{x}_F) + \text{yloss}(\mathbf{x}_{SCF},y^*) \; \; \text{,}
\end{equation}
```

## Usage

## References

Karimi, Amir-Hossein, Bernhard Schölkopf, and Isabel Valera. 2021. “Algorithmic Recourse: From Counterfactual Explanations to Interventions.” In *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency*, 353–62.
