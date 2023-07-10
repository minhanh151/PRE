import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

from itertools import cycle
from copy import deepcopy
# from transformers import AdamW
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from sklearn.metrics import matthews_corrcoef, f1_score

_tokenizer = _Tokenizer()

class ResMLP(torch.nn.Module):
    def __init__(self,
                 bottleneck_size,
                 module_type='MLP1',
                 emb_dimension=512,
                 nonlinearity='relu', # activation function
                 layer_norm=True,
                 dropout=0.0,
                 residual=True,
                 ):
        """MLP class for soft prompt re-parameterization. MLP can have a Residual connection.
        Args:
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used.
                Currently supports 1-layer and 2-layer MLPs, and simple transformer layer ('MLP1'/'MLP2'/'transformer').
                Defaults to 'MLP1'.
            emb_dimension (int, optional): Dimension of T5 model embeddings. Defaults to 512 (T5-small embedding dimension).
            residual (bool, optional): Whether to use residual connection in MLP. Defaults to True.
        """
        super().__init__()
        assert module_type in ['MLP1', 'MLP2', 'transformer', 'LSTM', 'LSTM1', 'LSTM2']
        assert nonlinearity in ['relu', 'tanh', 'sigm']

        self.module_type = module_type

        if module_type not in ['LSTM', 'LSTM1', 'LSTM2', 'transformer']:
            layers = [nn.Linear(emb_dimension, bottleneck_size)]

            if nonlinearity=='relu':
                layers.append(nn.ReLU())
            elif nonlinearity=='tanh':
                layers.append(nn.Tanh())
            elif nonlinearity=='sigm':
                layers.append(nn.Sigmoid())

            layers.append(nn.Linear(bottleneck_size, emb_dimension))

            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
            if layer_norm:
                layers.append(nn.LayerNorm(emb_dimension))

            if module_type=='MLP2':
                layers = layers + layers # repeat twice
            self.module = torch.nn.Sequential(*layers)

        elif module_type in ['LSTM1', 'LSTM2', 'LSTM']:
            self.lstm_head = torch.nn.LSTM(input_size=emb_dimension,
                                           hidden_size=emb_dimension // 2,
                                           num_layers=1 if module_type=='LSTM1' else 2,
                                           dropout=0.05,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(emb_dimension, emb_dimension),
                                          nn.ReLU(),
                                          nn.Linear(emb_dimension, emb_dimension))


        elif module_type=='transformer':
            device = 'cpu'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs):
        if self.module_type=='LSTM':
            output_embeds = self.mlp_head(self.lstm_head(inputs)[0]).squeeze()
        elif self.module_type in ['LSTM1', 'LSTM2']:
            output_embeds = self.lstm_head(inputs)[0].squeeze()
            if self.residual:
                output_embeds += inputs
            return output_embeds

        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)

# Create MLP for prompt tuning
def get_MLP(self, prefix_MLP, bottleneck_size, layer_norm, mlp_dropout, separate_mlps=False):
    if prefix_MLP == 'None':
        self.prefix_MLP = None
    else:
        print('Using MLP reparametrization with bottleneck = ', bottleneck_size)
        N = self.model.encoder.embed_tokens.weight.shape[1]

        if separate_mlps: # separate MLP for each prompt token
            print('Creating a dictionary of MLPs')
            m = self.prefix_len
            self.prefix_MLP = {}
            for i in range(m):
                self.prefix_MLP[i] = ResMLP(bottleneck_size=bottleneck_size,
                                        module_type=prefix_MLP,
                                        dropout=mlp_dropout,
                                        emb_dimension=N,
                                        nonlinearity='relu', # activation function
                                        layer_norm=layer_norm,
                                        #residual=True
                                        residual=self.residual,
                                        )
            for i in list(self.prefix_MLP):
                self.prefix_MLP[i].to(self.device)

        else: # 1 shared MLP
            self.prefix_MLP = ResMLP(bottleneck_size=bottleneck_size,
                                        module_type=prefix_MLP,
                                        dropout=mlp_dropout,
                                        emb_dimension=N,
                                        nonlinearity='relu', # activation function
                                        layer_norm=layer_norm,
                                        #residual=True
                                        residual=self.residual,
                                        )
            self.prefix_MLP.to(self.device)
    #if self.prefix_MLP!=None:
    #    self.prefix_MLP.to(self.device)


# Initialize new task prompt from random vocab. tokens
def init_new_prompt(self, prompt_len, random_init=False):
    if self.prefix_path==None:
        model = self.model
        N = model.encoder.embed_tokens.weight.shape[0]
        prompt_weigths = []

        # init from random uniform
        if random_init:
            r1, r2 = -0.5, 0.5
            x = (r1 - r2) * torch.rand(prompt_len, N) + r2
            prompt_weigths = x.numpy()
        else: # init from random tokens
            for i in range(prompt_len):
                with torch.no_grad():
                    j = np.random.randint(N) # random token
                    w = deepcopy(model.encoder.embed_tokens.weight[j].detach().cpu().numpy())
                    prompt_weigths.append(w)
        prompt_weigths = np.array(prompt_weigths)

    else: # initializing from existing path
        prompt_weigths = np.load(self.prefix_path)
    return prompt_weigths

# passing each prompt token through a separate MLP
def pass_separate_mlps(self):
    x = self.model.prompt
    out = []
    for i in range(self.prefix_len):
        h = self.prefix_MLP[i](x[i:i+1])
        out.append(h)
    out = torch.concat(out)
    return out





def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, 
                 bottleneck_size=64, # bottleneck size in case of using MLP reparametrization
                 mlp_dropout=0.0,
                 prefix_MLP='MLP1',
                 layer_norm=False,
                 residual=True
                 ):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.ResidualPrompting.N_CTX
        ctx_init = cfg.TRAINER.ResidualPrompting.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.ResidualPrompting.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.prefix_MLP = ResMLP(bottleneck_size=bottleneck_size,
                                         module_type=prefix_MLP,
                                         dropout=mlp_dropout,
                                         emb_dimension=ctx_dim,
                                         nonlinearity='relu', # activation function
                                         layer_norm=layer_norm,
                                         residual=residual,
                                         )

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.ResidualPrompting.CLASS_TOKEN_POSITION

    def forward(self):
        # ctx = self.ctx
        ctx = self.prefix_MLP(self.ctx)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )      

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class ResidualPrompting(TrainerX):
    """Residual Prompting).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ResidualPrompting.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.ResidualPrompting.PREC == "fp32" or cfg.TRAINER.ResidualPrompting.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.ResidualPrompting.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.ResidualPrompting.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    # Create optimizer
    def get_optimizer(self, lr, weight_decay): # task is used for MLP

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.prompt_learner.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },

            {
                "params": [p for n, p in self.model.prompt_learner.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
        ]

        # if self.model.prompt_learner.prefix_MLP!=None:
        #     weight_decay_mlp = weight_decay
        #     mlp_lr = lr

        #     # if self.separate_mlps==False: # shared MLP
        #     #     all_mlps = [self.prefix_MLP]
        #     # else: # separate MLP for each token
        #     #     all_mlps = [self.prefix_MLP[i] for i in list(self.prefix_MLP)]
        #     all_mlps = [self.model.prompt_learner.prefix_MLP]
        #     print(all_mlps)

        #     for mlp in all_mlps:
        #         optimizer_grouped_parameters.append({
        #             "params": [p for n, p in mlp.named_parameters()],# if not any(nd in n for nd in no_decay)],
        #             "weight_decay": weight_decay_mlp,
        #             "lr": mlp_lr,
        #         })
        # print(optimizer_grouped_parameters)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, eps=1e-8)
        return optimizer
