[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HH5Fey_mTiz29l9dGmHGqZqdzwLpLrxj?usp=sharing)
[![Huggingface Space](https://img.shields.io/badge/🤗-Huggingface%20Space-cyan.svg)](https://huggingface.co/spaces/pcuenq/paella)

# Paella
Conditional text-to-image generation has seen countless recent improvements in terms of quality, diversity and fidelity. Nevertheless, most state-of-the-art models require numerous inference steps to produce faithful generations, resulting in performance bottlenecks for end-user applications. In this paper we introduce Paella, a novel text-to-image model requiring less than 10 steps to sample high-fidelity images, using a speed-optimized architecture allowing to sample a single image in less than 500 ms, while having 573M parameters. The model operates on a compressed & quantized latent space, it is conditioned on CLIP embeddings and uses an improved sampling function over previous works. Aside from text-conditional image generation, our model is able to do latent space interpolation and image manipulations such as inpainting, outpainting, and structural editing.
<br>
<br>
![collage](https://user-images.githubusercontent.com/61938694/231021615-38df0a0a-d97e-4f7a-99d9-99952357b4b1.png)

## Update 12.04
Since the paper-release we worked intensively to bring Paella to a similar level as other 
state-of-the-art models. With this release we are coming a step closer to that goal. However, our main intention is not
to make the greatest text-to-image model out there (at least for now), it is to bring text-to-image models closer
to people outside the field on a technical level. For example, a lot of models have codebases with many thousand lines 
of code, that make it very hard for people to dive into the code and easily understand it. And that is the contribution
we are the proudest of with Paella. The training and sampling code for Paella is minimalistic and can be understood in 
a few minutes, making further extensions, quick tests, idea testing etc. extremely fast. For instance, the entire
sampling code can be written in just **12 lines** of code.


Please find all details about the model and how it was trained in our [preprint paper on arxiv](https://arxiv.org/pdf/2211.07292.pdf).
<hr>

## Code
We especially want to highlight the minimalistic amount of code that is necessary to run & train Paella. 
The training & sampling code can fit in under 140 lines of code. We hope to the field of generative AI and especially 
text-to-image more accessible to more people this way. In order to just understand the basic logic you can take a look 
at the [main folder](https://github.com/dome272/Paella/tree/main/src). For a more advanced training script, 
including mixed precision, distributed training, better logging and all conditioning models you can take a look at the 
[distributed folder](https://github.com/dome272/Paella/tree/main/src_distributed).

## Models
| Model           | Download                                             | Parameters      | Conditioning                       |
|-----------------|------------------------------------------------------|-----------------|------------------------------------|
| Paella v3       | [Huggingface](https://huggingface.co/dome272/Paella) | 1B (+1B prior)  | ByT5-XL, CLIP-H-Text, CLIP-H-Image |

## Sampling
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1geY_Z8m8dyjrky6uwiMepwySTWkVYl1j?usp=sharing)

For sampling you can just take a look at the [sampling.ipynb](https://github.com/dome272/Paella/blob/main/paella_inference.ipynb) notebook. :sunglasses:

## Train your own Paella
The main file for training will be [paella.py](https://github.com/dome272/Paella/blob/main/paella.py). You can adjust all [hyperparameters](https://github.com/dome272/Paella/blob/main/paella.py#L322) to your own needs. During training we use webdataset, but you are free to replace that with your own custom dataloader. Just change the line on 119 in [paella.py](https://github.com/dome272/Paella/blob/main/paella.py#L119) to point to your own dataloader. Make sure it returns a tuple of ```(images, captions)``` where ```images``` is a ```torch.Tensor``` of shape ```batch_size x channels x height x width``` and captions is a ```List``` of length ```batch_size```. Now decide if you want to finetune Paella or start a new training from scratch:
### From Scratch
```
python3 paella.py
```
### Finetune
If you want to finetune you first need to download the [latest checkpoint and it's optimizer state](https://drive.google.com/drive/folders/1ADAV-WPhMKGnm2w0bTO4HKhv6yoHB0Co), set the [finetune hyperparameter](https://github.com/dome272/Paella/blob/main/paella.py#L249) to ```True``` and create a folder ```models/<RUN_NAME>``` and move both checkpoints to this folder. After that you can also just run:
```
python3 paella.py
```

### License
The model code and weights are released under the [MIT license](https://github.com/dome272/Paella/blob/main/LICENSE).