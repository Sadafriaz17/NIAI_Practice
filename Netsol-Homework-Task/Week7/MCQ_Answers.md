# Week 7 – MCQs: ANN, CNN, LSTM and GRU (answer key)

| Q | Answer | Why |
|---|--------|-----|
| 1 | **B** | A weight controls the strength of the connection between two neurons. |
| 2 | **C** | A bias is added to the weighted sum before the activation function. |
| 3 | **A** | Sigmoid squashes the output to (0, 1), i.e. a probability for binary classification. |
| 4 | **B** | Backpropagation computes the gradients that the optimizer uses to update the parameters. |
| 5 | **A** | A filter/kernel learns to detect local patterns such as edges or textures. |
| 6 | **B** | Pooling reduces the spatial size of the feature maps (less computation, some translation invariance). |
| 7 | **A** | Local connections + shared filter weights = far fewer parameters and translation-equivariant features. |
| 8 | **A** | Padding adds values (usually zeros) around the border to control the output size (e.g. "same" convolution). |
| 9 | **B** | Vanishing/exploding gradients during backpropagation through time. |
| 10 | **A** | The forget gate decides which parts of the cell state are kept or discarded. |
| 11 | **B** | The cell state is the memory line that carries information across time steps. |
| 12 | **A** | The input gate controls which new information is written into the cell state. |
| 13 | **B** | A GRU merges the cell state and hidden state into a single state (and has 2 gates instead of 3). |
| 14 | **A** | The update gate balances how much of the past state is kept vs. how much new candidate state is used. |
| 15 | **A** | The reset gate decides how much of the previous hidden state contributes to the candidate state. |

**Compact key:** 1-B, 2-C, 3-A, 4-B, 5-A, 6-B, 7-A, 8-A, 9-B, 10-A, 11-B, 12-A, 13-B, 14-A, 15-A
