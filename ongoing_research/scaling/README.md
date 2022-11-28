# Paella
This is our open-source place where we put our latest runs / ideas on scaling / improving Paella. We invite everybody to
contribute or post ideas on ways to improve the method. 


## Train your own Paella
The main file for training will be [paella_h.py](https://github.com/dome272/Paella/blob/main/paella.py). You can adjust all [hyperparameters](https://github.com/dome272/Paella/blob/main/paella.py#L322) to your own needs. During training we use webdataset and we use SLURM to run jobs.

## Ideas
- [ ] Improve noising function. Currently the noising process either keeps a token or randomly replaces the token with another token from the codebook. An idea is to noise tokens based on their similarity to other tokens according to noising level. (E.g. if the noise-level is very low, tokens will be replaced by very similar other tokens and when the noising-level is high, the tokens will be replaced by very different tokens from the codebook.)
- [ ] Initialize learned token embeddings with the embeddings from the VQGAN
- [ ] Make the model much deeper and do not increase width when trying to increase model size
- [ ] Use Stable-Diffusion backbone.
- [ ] Replace Modulated LayerNorms with Cross-Attention for injecting conditions.
- [ ] Use T5 for conditioning

### License
The model code and weights are released under the [MIT license](https://github.com/dome272/Paella/blob/main/LICENSE).
