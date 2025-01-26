# BEN: Background Erase Network

[![arXiv](https://img.shields.io/badge/arXiv-2501.06230-b31b1b.svg)](https://arxiv.org/abs/2501.06230)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-BEN-blue)](https://huggingface.co/PramaLLC/BEN)
[![Website](https://img.shields.io/badge/Website-backgrounderase.net-104233)](https://backgrounderase.net)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ben-using-confidence-guided-matting-for/dichotomous-image-segmentation-on-dis-vd)](https://paperswithcode.com/sota/dichotomous-image-segmentation-on-dis-vd?p=ben-using-confidence-guided-matting-for)

## Overview
BEN (Background Erase Network) introduces a novel approach to foreground segmentation through its innovative Confidence Guided Matting (CGM) pipeline. The architecture employs a refiner network that targets and processes pixels where the base model exhibits lower confidence levels, resulting in more precise and reliable matting results.

This repository provides the official evaluation for our model, as detailed in our research paper: [BEN: Background Erase Network](https://arxiv.org/abs/2501.06230).



## BEN2 Access
BEN2 is now publicly available, trained on DIS5k and our 22K proprietary segmentation dataset. Our enhanced model delivers superior performance in hair matting, 4K processing, object segmentation, and edge refinement. Access the base model on Huggingface, try the full model through our free web demo or integrate BEN2 into your project with our API:
- 🤗 [PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2)
- 🌐 [backgrounderase.net](https://backgrounderase.net)

## Model Access
The base model is publicly available and free to use for commercial use on HuggingFace:
- 🤗 [PramaLLC/BEN](https://huggingface.co/PramaLLC/BEN)
