# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

*Note*: We try to adhere to these practices as of version [v1.1.1].

## Version [1.3.3] - 2024-09-30



## Version [1.3.2] - 2024-09-24

### Added 

- Added support for using a random forest as a surrogate model for the T-CREx generator. [#483]

### Changed

- Improved the T-CREx documentation further by bringing example even closer to the example in the paper. [#483]
- Include citation linking to ICML paper in T-CREx documentation and docstrings. [#480]

## Version [1.3.1] - 2024-09-24

### Changed

- Fixed a remaining bug in `NeuroTreeExt` extensions. [#475]

## Version [1.3.0] - 2024-09-16

### Changed

- Fixed bug in `NeuroTreeExt` extensions. [#475]

### Added

- Added basic support for the T-CREx counterfactual generator. [#473]
- Added docstrings for package extensions to documentation. [#475]

## Version [1.2.0] - 2024-09-10

### Added

- Added documentation for generating counterfactuals consistent with the MINT framework. [#467]
- Added tests for new evaluation metrics and JEM extension. [#471]
- Added support for gradient-based causal algorithm-recourse (MNIT) as described in Karimi et al. ([2020](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=umI56k0AAAAJ&sortby=pubdate&citation_for_view=umI56k0AAAAJ:kNdYIx-mwKoC)). This incorporates an input encoder that is based on a Structural Causal Model  [#457] 
- Added out-of-the-box support for training joint energy models (JEM). [#454]
- Added new evaluation metric to measure faithfulness of counterfactual explanations as in Altmeyer et al. ([2024](https://scholar.google.com/scholar?cluster=3697701546144846732&hl=en&as_sdt=0,5)). [#454]
- A tutorial in the documentation ("Explanation" section) explaining the faithfulness metric in detail. [#454]
- Added support for an energy constraint as in Altmeyer et al. ([2024](https://scholar.google.com/scholar?cluster=3697701546144846732&hl=en&as_sdt=0,5)). This is the first step towards adding functionality for ECCCo. [#387] 
  
### Changed

- The `fitresult` field of `Model` now takes a concrete `Fitresult` type, for which some basic methods have been defined. This mutable struct has a field called `other` that accepts a dictionary `Dict` that can be filled with additional objects. [#454]
- Regenerated pre-trained model artifacts. [#454]
- Updated the tutorial on "Handling Data". [#454]

### Removed

- Removed bug in `find_potential_neighbours` method. [#454]

## Version [1.1.6] - 2024-05-19

### Removed

- Removed the call to the `Iris` function in the test suite because of HTTPs issues. [#452]
- Removed the `mlj_models_catalogue` because it served no obvious purpose. In the future, we may instead add meta information to the `all_models_catalogue`. [#444]

### Added

- New general `Model` struct that wraps empty concrete types. This adds a more general interface that is still flexible enough by simply using multiple dispatch on the empty concrete types. [#444]
- A new `incompatible(::AbstractGenerator, ::AbstractCounterfactualExplanation)` function has been added to avoid running a counterfactual search if the generator is incompatible with any other specification (e.g. the model). [#444]

### Changed

- No longer exporting many of the deprecated functions. [#452]
- Updated pre-trained model artifacts. [#444]
- Some function signatures have been deprecated, e.g. `NeuroTreeModel` to `NeuroTree`, `LaplaceReduxModel` to `LaplaceNN`. [#444]
- Support for `DecisionTree.jl` models and the `FeatureTweakGenerator` have been moved to an extension (`DecisionTreeExt`). [#444]
- Updates to NeuroTreeModels extensions to incorporate breaking changes to package. [#444]
- No longer running alloc test on Windows. [#441]
- Slight change to doctests. [#447]

## Version [v1.1.5] - 2024-04-30

### Added 

- Unit tests: adds a simple performance benchmark to test that for a small problem, generating a counterfactual using the generic generator takes at most 4700 allocations. Only run on julia `v1.10` and higher. [#436]

### Changed

- The `find_potential_neighbours` is now only triggered if one of the penalties of the generator requires access to samples from the target domain. This improves scalability because calling the function can be computationally costly (forward-pass). [#436] 
- The target variable encodings are now handled more efficiently. Previously certain tasks were repeated, which was not necessary. [#436]

### Removed

- Removed the assertion checking that the model ever predicts the target value. While this assertion is useful, it is not essential. For large enough models and datasets, this forward pass can be very costly. [#436]
- Removed redundant `distance_from_targets` function. [#436]

## Version [v1.1.4] - 2024-04-25

### Changed

- Refactors the encodings and decodings such that it is now more streamlined. Instead of conditional statements, encodings are now dispatched on the type of a new unifying `data.input_encoder` field. [#432]
- Refactors the check for redundancy. This is now based on the convergence type and done right before the counterfactual search begins, if not redundant. [#432]

### Added

- Added additional unit tests. [#437]

## Version [v1.1.3] - 2024-04-17

### Added

- Adds a section on `Convergence` to the documentation, `Changelog.jl` functionality and a few doc tests. [#429]

### Changed

- Changes style of taking gradients for the counterfactual search from implicit to explicit. [#430]
- Removed all implicit imports. [#430]

### Removed 

- Removed CUDA.jl dependency, because redundant. [#430]
- Removed Parameters.jl dependency, because redundant. [#430]

## Version [v1.1.2] - 2024-04-16

### Changed

- Replaces the GIF in the README and introduction of docs for a static image. 

## Version [v1.1.1] - 2024-04-15

### Added

- Added tests for LaplaceRedux extension. Bumped upper compat bound for LaplaceRedux.jl. [#428]


<!-- Links generated by Changelog.jl -->

[#428]: https://github.com/juliatrustworthyai/CounterfactualExplanations.jl/issues/428
[#429]: https://github.com/juliatrustworthyai/CounterfactualExplanations.jl/issues/429
