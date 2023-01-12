# Attention is All you Need (Transformers)

- __Resources__:
    1. https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#The-Transformer-architecture
    2. https://peterbloem.nl/blog/transformers
    3. https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/transformer_from_scratch
    

- Transformer Encoder/Decoder Unit Dtails
    - __Operations__: Linear, Layer Norm, Activation, Tensor Multiply/Add, Softmax

- Types of data that transformers can process:
    - It can process anothing.
        - It can process various details at the same time.

- Good in processing sequences of data.

- The data goes through two blocks
    1. Self Attention
    2. Feed Forward Neural Network
    - The encoder and the decoder are made of a sequence of these blocks.

> The goal is to learn the attention weights.

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

- On images, let's consider the digit 3 from the MNIST,
    - The images are divided into patches, and let's consider one of the patch.
        - __Queries__ : Thinks it's part of digit 3 or 5.
        - __Key__ : Thinks it's part of 3, 5, 6, or 8.
        - __Values__ : It can see everything and it knows it's part of 3.

## Self-Attention

- Q, K, V are weight vectors which are multiplied with the inputs, to give rise to $q_i$, $k_i$ and $v_i$.

$$q_i = W^Qx_i^T;k_i = W^Kx_i^T;v_i = W^Vx_i^T$$

Now this can be easily vectorized by:

$$Q=X\cdot W^Q; K=X\cdot W^K; V=X\cdot W^V$$

- Now, the dot product of Q and K is computed.

$$S = Q\cdot K^T$$

## Multi Head Attention
- In most cases, the systems are highly non-linear, so we would need more attention layers and weights.
- In such scenarios we have multiple heads.
- If we have 8 heads, then we would have $<W^Q_1, W^K_1, W^V_1>, ..., <W^Q_8, W^K_8, W^V_8>$
- With multiple heads, we have multiple output features. How do we fuse them?
    - We concatenate them together and multiply them with the output weights tensor and heth the final output.

$$Z = cat(Z_1, ..., Z_8)\cdot W^O$$




## Adding Position Info to Inputs
- The problem with transformers in the lack inductive bias or the information of the sequence.
- Without the positional information, the transformer is clueless about which part is the noun, verb, etc.
    - So we add something called the __Positional Encodings__.

$$\hat{X}_i = x_i^T + p_i^T$$

### Positional Embeddings

$$PE_{(pos, 2i)} = \sin(\frac{pos}{10000^{\frac{2i}{dk}}}), \text{dim = 2i is even}$$

$$PE_{(pos, 2i+1)} = \cos(\frac{pos}{10000^{\frac{2i}{dk}}}), \text{dim = 2i+1 is odd}$$

- $pos = 0,1,...,n_{pos-1}$
- $dim = 0,1,...,n_{dim-1}$

- Other positional encodings can be used such as learnable parameters


## Other Improvements
- Layer Norm is feature dimension is dependant only spatial dimension and features and thus, computation is not dependant on the batch size, unlike Batch Norm.
- The concept of "Residual Connections" were also added to the network.
- There is an MLP layer to process the attributes. $$MLP(X) = max(0, XW_1+b_1)W_2 + b_2$$  and the activation function can be ReLU or GELU.


