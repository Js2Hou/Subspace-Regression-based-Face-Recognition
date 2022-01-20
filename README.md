# Subspace Regression based Face Recognition

## Introduction

This project implement some classic subspace regression for face recognition algorithms, including **LRC [1], RRC, SRC [2], CRC [3], Euler RRC, Euler SRC [4], Euler CRC**. RRC , Euler RRC and Euler CRC are proposed by myself. These methods can be divided into two categories, LRC-based and CRC-based. The former includes LRC, RRC, Euler RRC; the latter includes CRC, SRC, Euler CRC, Euler SRC. The project adopts sklearn style api design, which can be directly trained and tested by calling the fit method and score method of the model.

## Structure

```python
cProject/
|-- data/  # Store dataset used in experiments.
|
|-- model/
|   |-- _base.py
|   |-- complex_pca.py  # include complex pca and euler pca [5]
|   |-- subspace_regression.py  # implements subspace regression classification models here
|
|-- plot/  # Draw figures and tables to visualize experimental results.
|
|-- results/
|
|-- utils.py
|-- dataset.py  # Load and import dataset.
|-- model_evaluator.py  # Evaluate the performance of different models on different datasets.
|-- main.py
|-- README
```

## Examples

All models are implemented in subspace_regression.py. You can simply call the models in this class to complete a face recognition task. A simple example is shown below.

```python
import numpy as np

from dataset import AR
from model.subspace_regression import LRC

train_xs, train_ys, test_xs, test_ys = AR.exp1(mode=2)
model = LRC()
model.fit(train_xs, train_ys)
acc = model.score(test_xs, test_ys)
print(f'The accuracy of {model.__class__.__name__} on AR datest is: {acc}.\n')
```

```python
# Output
>> The accuracy of LRC on AR datest is: 0.7271.
```



## Acknowledgments

Thanks to the authors of the papers cited in this project.

## References

[1] Naseem I, Togneri R, Bennamoun M. Linear regression for face recognition. IEEE Trans Pattern Anal Mach Intell. 2010;32(11):2106-12.

[2] Wright J, Yang AY, Ganesh A, Sastry SS, Ma Y. Robust face recognition via sparse representation. PAMI. 2008;31(2):210-27.

[3] Zhang L, Yang M, Feng X, editors. Sparse representation or collaborative representation: Which helps face recognition? 2011 International conference on computer vision; 2011: IEEE.

[4] Liu Y, Gao Q, Han J, Wang S, editors. Euler sparse representation for image classification. Proceedings of the AAAI Conference on Artificial Intelligence; 2018.

[5] Liwicki S, Tzimiropoulos G, Zafeiriou S, Pantic M. Euler Principal Component Analysis. International Journal of Computer Vision. 2012;101(3):498-518.

