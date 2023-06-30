# Client meeting 28.04

## Repository Questions (questions from the TA)

- Double check if our client needs the mirroring, or whether a ZIP file is enough, as ideally we are trying to avoid mirroring repositories (mainly to limit the size of repositories and amount of updates needed).
- Ideally, TA would like to use SSH key but at the moment he doesn't see any SSH URL in the client's repo. Can client check if they have an SSH key setup? TA can then generate an SSH key for their repo.
- Do we need all the branches or only the protected ones (from the client's repo)?
- Idea: just fork the repo and every week push our changes from GitLab to GitHub.

## Product Specifications

- Are there already existing products/technologies that do things similar to what you need? Investigate them; can you learn from them? Incorporate them? Why (not)?
  - There’s no good reference solution, Python reference: [https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/?badge=latest](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/?badge=latest)
  - Differences: counterfactual generator as Patrick’s priority.
  - Similarities: a couple of datasets, benchmarking, separate module, inspiration for different approaches (e.g., feature tweak), explore them all and decide which are the most interesting, the easiest to implement (if something is really hard to create with the current architecture add a note or create a discussion on GitHub).
- What is expected from us:
  - If we want to meet other experts, we can reach out to Martin Pawelczyk, author of CARLA

## Feasibility Study

- Given the constraints on time and resources, it appears that the project is feasible but may require adjustments depending on the progress made during each phase. The project has goals that are part of open-ended research, which means that their difficulty is hard to gauge. We have agreed with the client that in the case where one of our predefined goals turns out to be infeasible, we will consider the goal completed upon delivery of a written report explaining the reason for the goal not being possible to complete.
  - Regarding the availability of technologies, frameworks, and data, there are existing resources like the CARLA library and the ONNX.jl framework that can be utilized, although the latter is currently under reconstruction. The project can also benefit from incorporating similar approaches and datasets to improve its functionality.

We have agreed to be flexible with the frameworks we use, as their effectiveness is difficult to predict.

## Risk Analysis

- Client Availability: While Patrick may have limited availability during the initial weeks, he has assured that he will be more involved after May 17th. Regular meetings can be scheduled to ensure project progress.
- Hardware and Data Requirements: The project does not require anything besides access to the current GitHub repository of Taija, which has been granted.
- Legal Issues: We are operating under the MIT license, and every team member is responsible for making sure the work they commit does not violate any laws or have copyright issues.
- Possible blockages due to insufficient prior knowledge:
  - Testing question -> Ask Antony, he has strong experience in testing AI systems.
  - General question -> As the project requires strong domain knowledge, we can expect to become blocked. We have discussed this with our client Patrick, and he will be fully available from May 17th onwards, where he will solely focus on our project and answer questions we have about the blockages we will have been facing.

## Requirements

- What’s the meaning of the ‘PERMANENT’ mark on issues?
  - It means there's no perfect solution and it can be always extended.
- What does it mean to ‘add more benchmark datasets’? Where are we supposed to add them? What kind of datasets should we add?
  - Datasets - enough synthetic data
  - Statlog (German credit data) dataset from the UCI ML repo
  - Adult Data Set from the UCI ML repo
  - JuliaML / MLDatasets.jl - add FashionMNIST, CIFAR10 - 1 or 2 is a must-have, more is a could
  - Boston housing
  - Also, 1 or 2 real-world datasets are a must-have
- Sort out exports
  - Just remove unnecessary exports. When a user comes to use the package, which of the methods should be exported to be readily available in their Julia session. 
- What is ONNX.jl? How are we supposed to use it? In the readme of the mentioned framework, it is said that the library is in the process of total reconstruction and that ‘When possible, functions from NNlib or standard library are used, but no conversion to Flux is implemented yet’. Does that mean we cannot use it?
  - Explore this, see whether it’s really not implemented, and if we get stuck on the issue and find out it can’t be implemented, that’s also considered as having completed the issue.
- When it comes to ‘Interface to <: Supervised MLJ models’ issue, are we supposed to build a ‘translation’ layer for MLJ models so that we can differentiate with regard to their features?
  - logits and probs methods on our side, predict method on the MLJ side - how can we wrap the predict call to use it on our side? To what extent is the predict call differentiable?
- ‘[Generator] MINT #105’ issue is mentioned in both ‘Getting started’ and ‘Beyond deep learning’ issues, so does that mean we’re not supposed to try to implement all generators in the ‘Getting started’ issue? If not, what is the order in which we should try to implement the generators - which ones are the most, which ones least important?
  - PROBE definitely fits the current framework well and should be prioritized. Has been shown to outperform ROAR, so ROAR becomes kind of redundant once PROBE has been implemented.
  - Feature tweak and growing spheres - seem easy to implement
  - MINT is challenging
- What does this mean to ‘improve dealing with count data’? How are we supposed to treat it?
  - Count data - the outcome variable is a count
- ‘Improve visualization for multivariate cases’ - are we supposed to find our own way of improving this? Are there any ‘expected’ solutions for this problem?
  - Multivariate data - currently compressed to 2D using PCA and then visualized with a chart generator from the package. Could look at other compression techniques, e.g., UMAP. There are 2 master’s students who have ideas.
- ‘[PERMANENT] Performance profiling’ - what are we supposed to do?
  - Optional, can think about this in the end
- For the issues that are marked as optional, does the client have any expectations about us getting some specific amount of them done or should we move on to them as the very last thing when everything else has been implemented?
  - We can tackle them after everything required is implemented.

## Frontend

- Documentation + we can make an interactive notebook for users to interact with the code.