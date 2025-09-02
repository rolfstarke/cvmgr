# onemanstreasure
a repo for training computer vision models in jupyter

```
bash git clone https://github.com/rolfstarke/onemanstreasure.git
```

make sure the dataset folder for label-studio is set correctly as env variable in the environment.yml
conda env create -f environment.yml
conda activate onemanstreasure
# label-studio

```
pip install label-studio label-studio-converter
label-studio start
```


# cvat
https://docs.docker.com/desktop/setup/install/linux/ubuntu/
https://docs.docker.com/engine/install/linux-postinstall/
download chrome 

conda env create -f environment.yml
conda install ipykernel
conda activate onemanstreasure
python -m ipykernel install --user --name onemanstreasure --display-name "Python (onemanstreasure)"
restart vscode