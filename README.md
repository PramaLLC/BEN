# BEN: Background Erase Network

[![arXiv](https://img.shields.io/badge/arXiv-2501.06230-b31b1b.svg)](https://arxiv.org/abs/2501.06230)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-BEN-blue)](https://huggingface.co/PramaLLC/BEN)

## Overview
BEN (Background Erase Network) introduces a novel approach to foreground segmentation through its innovative Confidence Guided Matting (CGM) pipeline. The architecture employs a refiner network that targets and processes pixels where the base model exhibits lower confidence levels, resulting in more precise and reliable matting results.

This repository provides the official evaluation for our model, as detailed in our research paper: [BEN: Background Erase Network](https://arxiv.org/abs/2501.06230).



## BENV2 Access
BEN2 base model is now publicly available. Trained on DIS5k and our 22K proprietary segmentation dataset, it delivers superior commercial performance in hair matting, 4K processing, object segmentation, and edge refinement.
- 🤗 [PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2)


## Model Access
The base model is publicly available and free to use for commercial use on HuggingFace:
- 🤗 [PramaLLC/BEN](https://huggingface.co/PramaLLC/BEN)
