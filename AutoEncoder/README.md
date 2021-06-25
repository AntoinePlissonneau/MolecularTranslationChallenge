# AutoEncoder

Basic knowledge of PyTorch, convolutional networks is assumed.

Questions, suggestions, or corrections can be posted as issues.

I'm using `PyTorch 1.7.1` in `Python 3.8.5`.

---

# Contents

[***Objective***](https://github.com/AntoinePlissonneau/MolecularTranslationChallenge/tree/main/AutoEncoder#objective)

[***Inference***](https://github.com/AntoinePlissonneau/MolecularTranslationChallenge/tree/main/AutoEncoder#inference)

# Objective

**AutoEncoder based on Unet architecture whose goal is to remove the disturbances on the images of molecules, to give color to the molecules, and to reconstruct the missing bonds.**

This AutoEncoder facilitates the work of the YOLO algorithm and the attention mechanism developed in parallel.

Here are some captions generated on _test_ images not seen during training or validation:

---

![](https://github.com/AntoinePlissonneau/MolecularTranslationChallenge/blob/main/AutoEncoder/img/AE_1.png "RES 0")

---

![](https://github.com/AntoinePlissonneau/MolecularTranslationChallenge/blob/main/AutoEncoder/img/AE_2.png "RES 2")

---

There are more examples at the [end of the tutorial](https://github.com/AntoinePlissonneau/MolecularTranslationChallenge/tree/main/AutoEncoder#some-more-examples).

---

# Inference

### Some more examples

---
