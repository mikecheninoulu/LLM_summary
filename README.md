# LLM_summary
It's a summarization of the recent LLM for vision understanding

Shoutout to myself for following up on the trendy techs even though I'm super busy after my PhD graduation.

# Classical models

## Large language models

**[1] LLaMA: Open and Efficient Foundation Language Models**
- intro: Meta AI
- point: Open and Efficient Foundation Language Models
- paper: [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)
- project: https://ai.meta.com/llama/

**[2] GPT-3: Language Models are Few-Shot Learners**
- intro: OPENAI, GPT3, 3.5,4
- point: no need to introduce
- paper: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
- project: https://github.com/openai/gpt-3

**[3] Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\% ChatGPT Quality**
- intro: Vicuna, descendants of the Meta LLaMA model 
- point: Strong baseline
- paper: [https://lmsys.org/blog/2023-03-30-vicuna/](https://lmsys.org/blog/2023-03-30-vicuna/)
- project: https://github.com/lm-sys/FastChat

**Others**
- compare different models: https://sapling.ai/llm/llama2-vs-vicuna

## Large vision models

**[1] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
- intro: ViT
- point: Backbone for the vision learning
- paper: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- project: https://github.com/google-research/vision_transformer


**Others**
- Qformer: see BLIP2
- ResNet

## Vision-Language frameworks

**[1] CLIP: Learning Transferable Visual Models From Natural Language Supervision**
- intro: ICML 2021, OpenAI
- point: milestone of vision-language model
- paper: [https://arxiv.org/pdf/2103.00020.pdf](https://arxiv.org/pdf/2103.00020.pdf)
- project: https://github.com/OpenAI/CLIP
- demo: https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb

**[2] BLIP, BLIP2**
- intro: the SOTA models now
- point: Qformer, qeury-based generation, Junnan Li.
- paper: [arxiv.org/abs/2301.12597](arxiv.org/abs/2301.12597)
- project: https://github.com/salesforce/BLIP
- huggingface: https://link.zhihu.com/?target=https%3A//huggingface.co/spaces/Salesforce/BLIP2
- zhihu: https://zhuanlan.zhihu.com/p/606364639


**Others**
- SLIP
- ALBEF
- ALIGN: Scaling Up Visual and Vision-Language
- InstructBLIP
- X-InstructBLIP
- llava https://llava-vl.github.io/


# LLM for video understanding

## Descrption

**[1] VideoChat: Chat-Centric Video Understanding**
- intro: OpenGVLab
- point: long video understanding, many variants, chat groups
- paper: [https://arxiv.org/pdf/2307.16449v2.pdf](https://arxiv.org/pdf/2307.16449v2.pdf)
- project: https://github.com/OpenGVLab/Ask-Anything/tree/main
- 2023 May

**[2] Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**
- intro: Alibaba group 
- point: process video inputs, not for long video reasoning
- paper: [https://arxiv.org/pdf/2306.02858.pdf](https://arxiv.org/pdf/2306.02858.pdf)
- project: https://github.com/DAMO-NLP-SG/Video-LLaMA
- 2023 Jun

**[3] MovieChat: From Dense Token to Sparse Memory for Long Video Understanding**
- intro: Microsoft Research Asia
- point: long video understanding, no reasoning
- paper: [https://arxiv.org/pdf/2305.06355.pdf](https://arxiv.org/pdf/2305.06355.pdf)
- project: https://github.com/rese1f/MovieChat
- 2023 August

## Classification

**[1] Frozen CLIP Models are Efficient Video Learners**
- intro: ECCV 2023

**[2] Fine-tuned CLIP Models are Efficient Video Learners**
- intro: CVPR 2023

# Library
**[1] LAVIS - A Library for Language-Vision Intelligence**
- intro: Junnan Li
- link: https://github.com/salesforce/LAVIS

**[2] ASK anything**
- intro: OpenGVLab
- link: [https://github.com/OpenGVLab/Ask-Anything](https://github.com/OpenGVLab/Ask-Anything)

  
# Self-supervised learning framework
**[1] Moco: Momentum Contrast for Unsupervised Visual Representation Learning**
- intro: FAIR, the first contrastive learning for large models from FAIR
- point: moco v1 baseline, moco v2 adds tricks from simCLR, moco v3 adds Transformer with stability, 
- link: https://github.com/salesforce/LAVIS

**[2] SimCLR: A Simple Framework for Contrastive Learning of Visual Representations**
- intro: Google brain
- point: two tricks, data augmentation, projection head.
- blog: https://sthalles.github.io/simple-self-supervised-learning/


**[3] Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning**
- intro: Google brain， BYOL
- point: non constrative learning
- code: https://github.com/sthalles/PyTorch-BYOL

- others


## Ideas
- generation CLIP-QAGAN
https://medium.com/nightcafe-creator/vqgan-clip-tutorial-a411402cf3ad
- emotion EXPBLIP
https://github.com/Yujianyuan/Exp-BLIP


## Implementation tutorial
**[1] AMD GPU**
- Youtube link: https://www.youtube.com/watch?v=UtcaO3zTCKQ
- https://www.reddit.com/r/LocalLLaMA/comments/13sthxx/using_amd_gpus/


| Updated: 2023/12/19|
| :---------: |
