# onemanstreasure
a repo for training computer vision models and merging datasets

```
bash git clone https://github.com/rolfstarke/onemanstreasure.git
```

conda env create -f environment.yml
conda activate cvmgr

to update the existing env
conda env update --file environment.yml --prune

# fiftyone cli

List all FiftyOne datasets
```
fiftyone datasets list
```
```
fiftyone datasets delete <dataset_name>
```





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

automessage a commit

# issues

when fiftyone doesnt list datasets, reinstalling and deleting .fiftyone and fiftyone in home seemed to work :/