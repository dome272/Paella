[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1geY_Z8m8dyjrky6uwiMepwySTWkVYl1j?usp=sharing)
[![LAION Blog Post](https://user-images.githubusercontent.com/61938694/232235929-94dacf4a-b3f6-4359-901b-500781f55c12.png)](https://laion.ai/blog/paella/)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/pcuenq/paella)

# Abstract
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


Please find all details about the model and how it was trained in our [paper on arxiv](https://arxiv.org/abs/2211.07292v2).
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

For sampling, you can just take a look at the [sampling.ipynb](https://github.com/dome272/Paella/blob/main/paella_inference.ipynb) notebook. :sunglasses: <br>
**Note**: Since we condition on ByT5-XL, CLIP-H-Text, CLIP-H-Image, sampling with the model takes at least 30GB of RAM,
unfortunately. We are hoping to use smaller conditioning models in the future.

## Train your own Paella
Depending on how you want to train Paella, we provided code for running it on a 
[single-GPU](https://github.com/dome272/Paella/tree/main/src) or for 
[multiple-GPU / multi-node training](https://github.com/dome272/Paella/tree/main/src_distributed).
The main file for training is [train.py](https://github.com/dome272/Paella/blob/main/src/train.py). You can adjust all 
[hyperparameters](https://github.com/dome272/Paella/blob/main/src/train.py#L10) to your own needs. 
In the distributed training code we provided a [webdataset](https://github.com/webdataset/webdataset/) dataloader,
whereas in the single-GPU code you have to set your [own dataloader](https://github.com/dome272/Paella/blob/main/src/utils.py#L19).
Make sure it returns a tuple of ```(images, captions)``` where ```images``` is a ```torch.Tensor``` of shape 
```batch_size x channels x height x width``` and captions is a ```List``` of length ```batch_size```. To start the
training you can just run ```python3 paella.py``` for the single-GPU case and for the multi-GPU case we provided a
[slurm](https://slurm.schedmd.com/documentation.html) script for launching the training you can find 
[here](https://github.com/dome272/Paella/blob/main/src_distributed/run/run.sh).


### License
The model code and weights are released under the [MIT license](https://github.com/dome272/Paella/blob/main/LICENSE).
