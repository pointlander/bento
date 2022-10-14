# Simplified Transformer
## Model
This model is a feedforwd network with [ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation functions. The input is image blocks tagged with a learned vector. Awareness is added by computing the average and variance of the output of each layer and then feeding that into the next layer. Self awareness is the sum of the system state mixed back into the system state.

## Results
Below is the accuracy of the model for the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset for different versions of the model.

| Correct | Total | Change                              |
| ------- | ----- | ----------------------------------- |
| 8474    | 10000 | without awareness                   |
| 8924    | 10000 | with average awareness              |
| 8989    | 10000 | with average and variance awareness |
| 9102    | 10000 | with dropout                        |
| 8825    | 10000 | with tags on middle layer           |
| 9082    | 10000 | with awareness on the input         |
| 9155    | 10000 | with relu on first layer            |
| 9325    | 10000 | removed TanH from output layer      |
| 9599    | 10000 | three layers                        |

## Citations
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [An Attention Free Transformer](https://arxiv.org/abs/2105.14103)
* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)