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

**[1] Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**
- intro: Alibaba group 
- point: process video inputs, not for long video reasoning
- paper: [https://arxiv.org/pdf/2306.02858.pdf](https://arxiv.org/pdf/2306.02858.pdf)
- project: https://github.com/DAMO-NLP-SG/Video-LLaMA


**[2] VideoChat : Chat-Centric Video Understanding**
- intro: OpenGVLab
- point: long video understanding, many variants, chat groups
- paper: [https://arxiv.org/pdf/2305.06355.pdf](https://arxiv.org/pdf/2305.06355.pdf)
- project: https://github.com/OpenGVLab/Ask-Anything/tree/main


# Library
**[1] LAVIS - A Library for Language-Vision Intelligence**
- intro: Junnan Li
- link: https://github.com/salesforce/LAVIS

**[2] ASK anything**
- intro: OpenGVLab
- link: [https://github.com/salesforce/LAVIS](https://github.com/OpenGVLab/Ask-Anything)


## Ideas
CLIP-QAGAN
https://medium.com/nightcafe-creator/vqgan-clip-tutorial-a411402cf3ad
EXPBLIP
https://github.com/Yujianyuan/Exp-BLIP


## Implementation tutorial
**[1] AMD GPU**
- Youtube link: https://www.youtube.com/watch?v=UtcaO3zTCKQ
- https://www.reddit.com/r/LocalLLaMA/comments/13sthxx/using_amd_gpus/


| Updated: 2023/12/19|
| :---------: |