## Generating artifacts

To ensure that artifacts are compatible with legacy Julia versions, we generate and serialize them on LTS (`v1.6`). 

### Quarto notebooks

We use Quarto notebooks for this purpose. To run the contained code cells in the VSCode Julia REPL with version `v1.6` activated, simply run `juliaup default 1.6`, then reload the VSCode window using the command palette (see [here](https://discourse.julialang.org/t/how-do-set-julia-version-in-vscode-when-using-juliaup/105619/10?u=pat-alt) for details)