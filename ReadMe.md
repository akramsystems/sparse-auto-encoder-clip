# Sparse Autoencoder on ImageNet

![Streamlit App Screencast](streamlit-app-screencast.gif)

## Set up Environment with Anaconda

### Create Environment
```bash
conda create -n sae-clip     python=3.10
conda activate sae-clip
pip install -r requirements.txt
```


## Train Sparse Autoencoder on Clip ImageNet
This generates the model file sae_epoch.pt file

```bash
python -m src.train.train_sae
```

## Dashboard

### Download Sparse Autoencoder Trained Model

[Sparse Autoencoder Trained Model](https://drive.google.com/file/d/1xMGPoUgSBCY8LmsHxtl9gge5FILrfYQl/view?usp=drive_link)

Download the file and put it in the root directory of the project

### Precompute features (clip features)

```bash
python -m src.train.precompute_clip_features
```

### Find Top Activating Features

```bash
python -m src.evaluation.neuron_activation
```

### Run Dashboard
```bash
python -m streamlit run app.py
```
