[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HH5Fey_mTiz29l9dGmHGqZqdzwLpLrxj?usp=sharing)
[![Huggingface Space](https://img.shields.io/badge/🤗-Huggingface%20Space-cyan.svg)](https://huggingface.co/spaces/pcuenq/paella)

# Paella
Conditional text-to-image generation has seen countless recent improvements in terms of quality, diversity and fidelity. Nevertheless, most state-of-the-art models require numerous inference steps to produce faithful generations, resulting in performance bottlenecks for end-user applications. In this paper we introduce Paella, a novel text-to-image model requiring less than 10 steps to sample high-fidelity images, using a speed-optimized architecture allowing to sample a single image in less than 500 ms, while having 573M parameters. The model operates on a compressed & quantized latent space, it is conditioned on CLIP embeddings and uses an improved sampling function over previous works. Aside from text-conditional image generation, our model is able to do latent space interpolation and image manipulations such as inpainting, outpainting, and structural editing.
<br>
<br>
![cover-figure](https://user-images.githubusercontent.com/117442814/201474789-a192f6ab-9626-4402-a3ec-81b8f3fd436c.png)

Please find all details about the model and how it was trained in our [preprint paper on arxiv](https://arxiv.org/pdf/2211.07292.pdf).
<hr>

## Code
We especially want to highlight the minimalistic amount of code that is necessary to run & train Paella. The entire code including training, sampling, architecture and utilities can fit in approx. 400 lines of code. We hope to make this method more accessible to more people this way. In order to just understand the basic logic you can take a look at [paella_minimal.py](https://github.com/dome272/Paella/blob/main/paella_minimal.py).

## Sampling
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HH5Fey_mTiz29l9dGmHGqZqdzwLpLrxj?usp=sharing)

For sampling you can just take a look at the [sampling.ipynb](https://github.com/delicious-tasty/Paella/blob/main/paella_sampling.ipynb) notebook. :sunglasses:

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
