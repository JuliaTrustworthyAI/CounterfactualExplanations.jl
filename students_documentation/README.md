![](./pictures/logo.png)

## Description of project

In this project, we focus on contributing to CounterfactualExplanations.jl, a trustworthy artificial intelligence package for generating Counterfactual Explanations and Algorithmic Recourse for black-box algorithms in Julia. As contributors, we work on feature enhancements, including increasing the scope of predictive models that are compatible with the package. Our expected outcomes include familiarizing ourselves with the package, increasing our understanding of Counterfactual Explanations and Algorithmic Recourse, gaining relevant experience with FOSS development, and becoming familiar with Julia. This project provides us with an opportunity to get involved in Taija projects and contribute to the open-source community.

## Group members

| Name           | Email                          |
| -------------- | ------------------------------ |
| Lauri Keskul   | l.keskull@student.tudelft.nl   |
| Rauno Arike    | r.arike@student.tudelft.nl     |
| Vincent Pikand | v.pikand@student.tudelft.nl    |
| Simon Kasdorp  | s.a.kasdorp@student.tudelft.nl |
| Mariusz Kicior | m.a.kicior@student.tudelft.nl  |

## Repository structure

### Must Have Features

#### Generators

- Feature Tweak
  - Logic:
    - [feature_tweak.jl](..\src\generators\non_gradient_based\feature_tweak\feature_tweak.jl)
    <!-- - `src\generators\non_gradient_based\functions.jl` -->
    <!-- - `src\generators\non_gradient_based\generators.jl` -->
  - Tests: [counterfactuals.jl](..\test\generators\feature_tweak.jl)
  - Example: [summary.qmd](summary_notebook\summary.qmd)
  - Documentation: [feature_tweak.qmd](..\docs\src\explanation\generators\feature_tweak.qmd)

#### Added Datasets

- Statlog German credit dataset 
  - Logic: [german_credit.jl](..\src\data\tabular\german_credit.jl)
  - Tests: [tabular.jl](..\test\data\tabular.jl)
  - Example: [summary.qmd](summary_notebook\summary.qmd)

#### Model Compatibility

- PyTorch models
  - Core logic:
    - [pytorch_model](..\src\models\differentiable\python\pytorch_model.jl)
  - Utility logic:
    - [utils.jl](..\src\data_preprocessing\utils.jl)
    - [model_utils.jl](..\src\models\utils.jl)
  - Tests: [pytorch.jl](..\test\models\pytorch.jl)
  - Example: [summary.qmd](summary_notebook\summary.qmd)
  - Documentation: [model_catalogue.qmd](..\docs\src\tutorials\model_catalogue.qmd)
  - Summary of our efforts to make Python and R models compatible with the generators: [PyTorch_and_R_models_report.md](Python_and_R_models_report.md)

- R torch models
  - Core logic:
    - [rtorch_model.jl](..\src\models\differentiable\R\rtorch_model.jl)
    - [loss.jl](..\src\generators\gradient_based\loss.jl)
  - Utility logic (model loader function): [utils.jl](../src/models/utils.jl)
  - Example: [summary.qmd](summary_notebook\summary.qmd)
  - Summary of our efforts to make Python and R models compatible with the generators: [PyTorch_and_R_models_report.md](Python_and_R_models_report.md)

- DecisionTreeClassifier and RandomForestClassifier from the MLJ library (https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/) 
  - Core logic:
    - [tree.jl](..\src\models\nondifferentiable\mlj\tree.jl)
  - Utility logic: [utils.jl](..\src\data_preprocessing\utils.jl)
  - Tests: [models.jl](..\test\models\models.jl)
  - Example: [summary.qmd](summary_notebook\summary.qmd)
  - Documentation: [model_catalogue.qmd](..\docs\src\tutorials\model_catalogue.qmd)
  - Summary of our efforts to make MLJ models compatible with the generators: [MLJ_models_report.md](MLJ_model_report.md)

- EvoTreeClassifier from the MLJ library (https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/) 
  - Core logic: [evotree.jl](..\src\models\differentiable\other\evotree.jl)
  - Utility logic: [utils.jl](..\src\data_preprocessing\utils.jl)
  - Tests: [models.jl](..\test\models\models.jl)
  - Example: [summary.qmd](summary_notebook\summary.qmd)
  - Documentation: [model_catalogue.qmd](..\docs\src\tutorials\model_catalogue.qmd)
  - Summary of our efforts to make MLJ models compatible with the generators: [MLJ_models_report.md](MLJ_model_report.md)

#### Exports

### Should Have Features

#### Generators

#### Added Datasets

- CIFAR10
  - Logic: [cifar_10.jl](..\src\data\vision\cifar_10.jl)
  - Tests: [vision.jl](..\test\data\vision.jl)
  - Example: [vision.qmd](..\dev\artifacts\vision.qmd)

- UCI Adult Dataset
  - Logic: [adult.jl](..\src\data\tabular\adult.jl)
  - Tests: [tabular.jl](..\test\data\tabular.jl)
  - Example: [summary.qmd](summary_notebook\summary.qmd)

### Could Have Features

- Investigate using UMAP for visualizing high-dimensional data
  - Report: [visualization_report.md](visualization_report.md)

- Investigate informed choices for default penalty strength 
  - Report: [notebook.qmd](penaltystrength\notebook.qmd)