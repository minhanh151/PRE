# Vision-Language Prompt Learning with Reparameterization Encoder

This repo contains the codebase of a research project focused on adapting vision-language models like [CLIP](https://arxiv.org/abs/2103.00020) to downstream datasets via *prompt learning*:

* [PRE: Vision-Language Prompt Learning with Reparameterization Encoder], 2023.

## Highlights
We introduce Prompt Learning with Reparameterization Encoder (PRE) - a simple and efficient method that enhances the generalization ability of the learnable prompt to unseen classes while maintaining the capacity to learn Base classes. Instead of directly optimizing the prompts, PRE employs a prompt encoder to reparameterize the input prompt embeddings, enhancing the exploration of task-specific knowledge from few-shot samples. Extensive evaluation shows that PRE is an efficient method, i.e., achieves better performance within good training time.

<img width="922" alt="Screenshot 2023-08-23 at 13 34 15" src="https://github.com/minhanh151/PRE/assets/55950352/2f614791-9739-4487-843e-3d2d0a7f49a6">

</br>
</br>

| Methods | Prompts | Base | New | H | Training-time|
|---------|---------|------|------|---|-------------|
| CLIP | hand-crafted | 68.81 | 74.43 | 71.42 | -|    
| CoOp | textual | 83.32 | 66.92 | 73.34 | 6ms/image|
| ProGrad | textual | 82.96 | 70.30 | 75.58 | 22ms/image|
| CoCoOp | textual+visual | 80.89 | 70.99 | 74.47 | 160ms/image|
| PRE | textual | 82.14 | 71.88 | 76.27 | 6.3ms/image|

## How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `PRE/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to Run

Click a paper below to see the detailed instructions on how to run the code on CoOp and CoCoOp to reproduce the results.

* [Learning to Prompt for Vision-Language Models](COOP.md)
* [Conditional Prompt Learning for Vision-Language Models](COCOOP.md)

Follow [PRE.md](PRE.md) to see the detailed instructions on how to run the PRE method to reproduce the results.

## Models and Results

- The pre-trained weights of CoOp (both M=16 & M=4) on ImageNet based on RN50, RN101, ViT-B/16 and ViT-B/32 can be downloaded altogether via this [link](https://drive.google.com/file/d/18ypxfd82RR0pizc5MM1ZWDYDk4j0BtPF/view?usp=sharing). The weights can be used to reproduce the results in Table 1 of CoOp's paper (i.e., the results on ImageNet and its four variants with domain shift). To load the weights and run the evaluation code, you will need to specify `--model-dir` and `--load-epoch` (see this [script](https://github.com/KaiyangZhou/CoOp/blob/main/scripts/eval.sh) for example).
- The raw numerical results can be found at this [google drive link](https://docs.google.com/spreadsheets/d/12_kaFdD0nct9aUIrDoreY0qDunQ9q9tv/edit?usp=sharing&ouid=100312610418109826457&rtpof=true&sd=true).


```
