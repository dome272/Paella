# Paella Goes Video
### Note: This is work in progress

## Idea
We are building upon the amazing [Phenaki](https://openreview.net/pdf?id=vOEXS39nOF) paper in that we again make use of a two-stage approach which compresses videos spatially and temporally. After that we learn Paella in the latent space just as in the original text-to-image approach, by just extending the 2D convolutions to 3D.

## First Stage Results
We trained a convolutional 3D VQGAN with a spatial compression of f8 and temporal compression of f2. Videos of **(10+1)x128x128** are encoded to a latent size of **(5+1)x16x16**. cViViT proposes to use a separate stem to encode the first frame. In our early experiments we saw that this stem would not receive a lot gradients and thus evolve very slowly, while the rest of the frames looked much better. As a result, we only use a single stem for all frames at once. To still enable image only training in the second stage, we learn an additional frame and prepend it to the start of the sequence, such that when downsampling temporally by 2, the learned and first frame would be encoded into one and the model could learn to ignore the learned embedding and only encode the information from the first frame. We trained the model (43M parameters) for 100k steps, with a batch size of 64 on 8 A100 for 1 day.
