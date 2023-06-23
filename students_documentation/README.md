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
    - `src\generators\non_gradient_based\base.jl`
    - `src\generators\non_gradient_based\functions.jl`
    - `src\generators\non_gradient_based\generators.jl`
  - Tests: `test\counterfactuals.jl`
  - Example: `students_documentation\summary_notebook\summary.qmd`
  - Documentation: `docs\src\explanation\generators\feature_tweak.qmd`

#### Added Datasets

- Statlog German credit dataset 
  - Logic: `src\data\tabular.jl`
  - Tests: `test\data.jl`
  - Example: `students_documentation\summary_notebook\summary.qmd`

#### Model Compatibility

- PyTorch models
  - Core logic:
    - `src\models\pytorch_model.jl`
    - `src\generators\gradient_based\functions.jl`
  - Utility logic:
    - `src\data_preprocessing\utils.jl`
    - `src\models\model_utils.jl`
  - Tests: `test\models.jl`
  - Test utilities: `test\pytorch.jl`
  - Example: `students_documentation\summary_notebook\summary.qmd`
  - Documentation: `docs\src\tutorials\model_catalogue.qmd`
  - Summary of our efforts to make Python and R models compatible with the generators: `students_documentation\PyTorch_and_R_models_report.md`

- R torch models
  - Core logic:
    - `src\models\differentiable\R\rtorch_model.jl`
    - `src\generators\gradient_based\loss.jl`
  - Utility logic (model loader function): `src/models/utils.jl`
  - Example: `students_documentation\summary_notebook\summary.qmd`
  - Summary of our efforts to make Python and R models compatible with the generators: `students_documentation\PyTorch_and_R_models_report.md`

- DecisionTreeClassifier and RandomForestClassifier from the MLJ library (https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/) 
  - Core logic:
    - `src\models\nondifferentiable\mlj_tree.jl`
    - `src\models\nondifferentiable\nondifferentiable.jl`
  - Utility logic: `src\data_preprocessing\utils.jl`
  - Tests: `test\models.jl`
  - Example: `students_documentation\summary_notebook\summary.qmd`
  - Documentation: `docs\src\tutorials\model_catalogue.qmd`
  - Summary of our efforts to make MLJ models compatible with the generators: `students_documentation\MLJ_models_report.md`

- EvoTreeClassifier from the MLJ library (https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/) 
  - Core logic: `src\models\differentiable\evotree_model.jl`
  - Utility logic: `src\data_preprocessing\utils.jl`
  - Tests: `test\models.jl`
  - Example: `students_documentation\summary_notebook\summary.qmd`
  - Documentation: `docs\src\tutorials\model_catalogue.qmd`
  - Summary of our efforts to make MLJ models compatible with the generators: `students_documentation\MLJ_models_report.md`

#### Exports

### Should Have Features

#### Generators

#### Added Datasets

- CIFAR10 
  - Logic: `src\data\vision.jl`
  - Tests: `test\data.jl`
  - Example: `dev\artifacts\vision.qmd`