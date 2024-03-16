## Word2vec

The key insight behind the word2vec is that *you shall know a word by the company it keeps*. Concretely, consider a 'center' word $c$ surrounded before and after by a context of a certain length. We term words in this contextual window "outside word". For example, in the following figure, the context window length is $2$, the center word is "banking", and the outside words are "turning", "into", "crises" and "as". 

<img src = "C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240228153213259.png" width = 700>

Skip-gram word2vec aims to learn the probability distribution $P(O|C)$. Specifically, given a center word $c$ and the outside word $o$, we want to predict $P(O=o|C=c)$. We model this probability by taking the softmax function over a series of vector dot-products: 
$$
P(O=o|C=c) = \frac{\exp(u_o^{T}v_c)}{\sum_{w\in\text{Vocab}} \exp(u_w^Tv_c)}
$$
For each word, we learn feature vectors $u$ and $v$, where $u_o$ is the outside vector representing the outside word $o$ and $v_c$ is the center vector representing the center word $c$. 

The loss function and the cost over a minibatch is given by: 
$$
J(v_c,o) = -\log P(O = o | C=c)\\
J(v_c, w_{t-m}...w_{t+m}) = \sum_{-m\leq j\leq m\\\ \ \ \ j\not=0} J(v_c,w_{t+j})
$$
Now we shall consider the Negative Sampling loss, which is an alternative to the Naive Softmax loss. Assume that $K$ negative examples are drawn from the vocabulary. We shall refer to them as $w_1, w_2...w_k$. The Negative Sampling loss is given by: 
$$
J_{neg}(v_c,o) = -\log (\sigma (u^{T}_ov_c)) - \sum_{i=1}^k \log(1-\sigma(u_{w_i}^Tv_c))
$$

## Neural Transition-Based Dependency Parsing

A dependency parser analyzes the grammatical structure of a sentence, establishing the relationship between head word, and words which modify those head. 

Here we implement the **transition-based parser**, which incrementally builds up a parser one step at a time. At every timestep it maintains a partial parser, which is represented as follows: 

- A **stack** of words that are currently being processed. 
- A **buffer** of words yet to be processed. 
- A list of **dependencies** predicted by the parser. 

Initially, the stack only contains ROOT, the dependency list is empty, and the buffer contains the words of the sentence in order. At each timestep, the parser applied a **transition** to the partial parser until its buffer is empty and the stack size is $1$. The following three transitions can be applied: 

- SHIFT: remove the first word in the buffer and push it into the stack. 
- LEFT-ARC: add a *first_word -> second_word* dependency to the dependency list and remove the *second_word* from the stack. 
- RIGHT-ARC: add a *second_word -> first_word* dependency to the dependency list and remove the *first_word* from the stack. 

The dependency list corresponds to the edge set of a tree, demonstrating the structure of target sentence. 

On each step, your parser will decide among the three transitions using a neural network classifier. Look at the architecture of the model: 

![image-20240228140823496](C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240228140823496.png)

First, the model extracts a feature vector representing the current state. This feature vector contains a list of token(words, POS tags, arc labels). They can be represented by a list of integers $ w = [w_1, w_2...w_m]$ where $m$ is the number of the features and each $0\leq w_i<|V|$ is the index of a token in the vocabulary. Then the input layer looks up an embedding for each word and concatenate them into a single input vector: 

$$\bold x = [\bold E_{w_1}, \bold E_{w_2}...\bold E_{w_m}]$$

where $\bold E$ is an embedding matrix with each row $\bold E_{w_i}$ as the vector of the word $w_i$. Then we compute our prediction as: 
$$
\bold h = \text{ReLU}(\bold {xW_1} + \bold b_1) \\
\bold l = \bold{hU} + \bold b_2 \\ 
\bold {\hat y} = \text{softmax}(\bold l)
$$
We will train the model to minimize the cross-entropy loss: 

$$J(\theta) = -\sum_{i=1}^3 y_i \log(\hat y_i)$$

## Neural Machine Translation with RNNs

In this assignment, we will implement a sequence-to-sequence(Seq2Seq) network with attention, to build a NMT system. The training procedure uses a Bidirectional LSTM Encoder and a Unidirectional LSTM Decoder. 

![image-20240301162543426](C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240301162543426.png)

**Encoder**

Given a sentence in the source language, we look up the **word embeddings** in the embedding matrix, yielding $x_1, x_2...x_m(x_i\in \mathbb{R}^{e\times 1})$, where $m$ is the length of the source sentence and $e$ is the embedding size. 

Then, we apply **1D-convolution** to the embeddings, while maintaining their shapes. 1D-convolution is as much same as 2D-convolution, despite that the operation is based on two sequence. 

Next, we feed the convolutional layer output into the **bidirectional encoder**, yielding hidden state and cell state for both the forward and backward LSTM. The forward and backward versions are concatenated to obtain the hidden state $ h^{enc}_i$ and cell states $ c_i^{enc}$
$$
h_i^{enc} = [\overset{\rightarrow}{ h_i^{enc}} ; \overset{\leftarrow}{ h_i^{enc}}] \ \ \ \ \ \ \ h_{i}^{enc}\in\mathbb{R}^{2h\times 1}\\
 c_i^{enc} = [\overset{\rightarrow}{ c_i^{enc}} ; \overset{\leftarrow}{ c_i^{enc}}] \ \ \ \ \ \ \ c_{i}^{enc}\in\mathbb{R}^{2h\times 1}
$$
where $ h_{i}^{enc}, c_{i}^{enc}\in \mathbb{R}^{2h\times 1}$ and $h$ is the size of unidirectional hidden state. 

Finally, we initialize the decoder's first hidden state $ h^{dec}_0$ and cell state $ c_{0}^{dec}$ with a linear projection of the encoder's final hidden state and final cell state: 
$$
h_0^{dec} = \text{W}_h[\overset{\rightarrow}{h_m^{enc}} ; \overset{\leftarrow}{ h_1^{enc}}] \ \ \ \ \ \ \ h_{0}^{dec}\in\mathbb{R}^{h\times 1}\\
c_0^{dec} = \text{W}_c[\overset{\rightarrow}{c_m^{enc}} ; \overset{\leftarrow}{ c_1^{enc}}] \ \ \ \ \ \ \ c_{0}^{dec}\in\mathbb{R}^{h\times 1}
$$
**Decoder**

When decoder is initialized, we must now feed it a target sentence. On the $t^{th}$ step,  we look up the embedding for the $t^{th}$ word $y_t\in \mathbb{R}^{e\times 1}$. We then concatenate $y_t$ with the **combined-output vector** $o_{t-1}$ from the previous time step to produce $\overline{y_t}\in \mathbb{R}^{(e+h)\times 1}$. Specifically, $o_t$ is an zero vector.

In the next stage, we feed $\overline{y_t}$ into the decoder LSTM:

$h_t^{dec}, c_t^{dec} = \text{ Decoder}(\overline{y_t}, h_{t-1}^{dec}, c_{t-1}^{dec})$

After we get the result from the LSTM cell, we use $h_{t}^{dec}$ to compute the attention weight over $\bold h^{enc}$: 
$$
e_{t,i} = h_{t}^{dec}\text{W}_{att}h^{enc}_i \\
\alpha_t = \text{softmax}(e_t) \\ 
a_t = \sum_{i=1}^{len}\alpha_{t,i}h_{i}^{enc}
$$
We now concatenate the decoder hidden state $h_{t}^{dec}$ with the attention output at and pass this through a linear layer, tanh, and dropout to attain the combined-output vector $o_t$. 
$$
u_t = [h_t^{dec}; a_t] \ \ \ \ \ \ \ u_t\in \mathbb{R}^{3h\times 1}\\ 
v_t = W_u u_t \ \ \ \ \ \ \ \ \ \ \ \ \ W_u\in\mathbb{R}^{h\times 3h}\\
o_t= \text{dropout}(\tanh (v_t))
$$
Then we use the combined output vector $o_t$ to produce a probability distribution over the vocabulary and finally compute the cross-entropy loss: 
$$
P_t = \text{softmax}(W_{vocab}o_t) \ \ \ \ \ \ W_{vocab}\in \mathbb R^{V\times h} \\ 
J(\theta) = \text{CrossEntropy}(P, y)
$$