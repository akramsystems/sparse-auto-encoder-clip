# Sparse Autoencoder on ImageNet

## Set up Environment with Anaconda

### Create Environment
```bash
conda create -n sae-clip     python=3.10
pip install -r requirements.txt
```

### Activate Environment

```bash
conda activate sae-clip
```

## Train Sparse Autoencoder on ImageNet

```bash
python -m src.train.train_sae
```

## Evaluate Sparse Autoencoder on ImageNet

```bash
python -m src.train.evaluate_sae
```

## Visualize Sparse Autoencoder on ImageNet

```bash
python -m src.train.visualize_sae
```