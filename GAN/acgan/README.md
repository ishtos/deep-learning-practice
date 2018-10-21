<img width="300" src="">

## requirements
- python 3.6.6
- GPU recommended
```
pip install -r requiremenrs.txt
```

## train
trained weights by our experiments are available (so you can skip this step), but also you can train your own model.
```
python dcgan.py --epochs=<int> --batch_size=<int> --latent_size=<int> --mnist=<0 or 1>
```
options

- epochs: default 500
- batch_size: default 100 
- latent_size: default 100 
- mnist: default 0
    - 0: mnist
    - 1: fashion_mnist
    
trained weights of both generator and discriminator, and generated images, are saved at log/ on every epoch.

## test
open test.ipynb.
