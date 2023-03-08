# Fine-Mapping
![License: Apache-2.0](https://img.shields.io/github/license/gil2rok/fine-mapping)

Perform fine-mapping that identifies which genetic variants (SNPS) cause a complex, phenotypic trait. Do so by extending a probablistic regression model (SuSiE) to smartly incoprorate functional annotations.

**Note: this repo is under active development and is not yet fully functional**

## Overview

TODO

## Navigating the Repo

1. The directory [code](/code) contains all my code for this project.
2. The directory [texts](/texts) contains relevant papers, summaries, or work of others.
3. The directory [lab_ntbk](/lab_ntbk) contains my notes on experiments, extensions, implentation progress, and debugging.# HERACLES
![License: Apache-2.0](https://img.shields.io/github/license/gil2rok/heracles)

HERACLES algorithm reconstructs CRISPR-Cas9 phylogenetic lineages with hyperbolic embeddings.

**Note: this repo is under active development and is not yet fully functional**

## Overview

Insert CRISPR-Cas9 casettes (i.e. cellular "barcodes") into a cell and sequence its descendants to reconstruct the evolutionary cell lineage. Using a CRISPR-specific continous time Markov chain, model the mutations that accumulate on these casettes. Then construct a function to quanitify the likelihood of a particular evolutionary tree, in both tree topology and branch lengths. Approixmate the tree-metric by embedding points in hyperbolic space. Lastly, optimize over the hyperbolic embeddings using Riemannian stochastic gradient descent.

## Navigating the Repo

1. The directory [code](/code/) contains all my code for this project.
2. The directory [writeups](/writeups/) contains presentations I have made for this project
3. The directory [texts](/texts/) contains relevant papers, summaries, or work of others. For example, an excellent summary of the theory behind this project can be found [here](/texts/sitara-writeup.pdf).