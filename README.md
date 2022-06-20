<h1 align="center">UCluster for the LHC Olympics 2020</h1>

> UCluster code used for [Mariana Vivas' project](https://github.com/marianaiv/benchtools). For the original repository go to [this link](https://github.com/lovaslin/GAN-AE_LHCOlympics).

# Instructions to use the code
First, clone the repository and enter in the folder:
```
git clone https://github.com/marianaiv/UCluster.git
cd UCluster
```
Then, create a virtual enviroment from the eviroment.yml file using conda and activate it..
```
conda env create -f environment.yml
conda activate UCluster
```
Donwload the datasets and save them in a `data` folder.
- [R&D dataset](https://zenodo.org/record/2629073#.XjOiE2PQhEa)
- [Black Boxes and labels](https://zenodo.org/record/4536624)

Pre-process the data. For the R&D dataset:
```
cd scripts
python prepare_data_unsup.py --dir ./../data --RD
```
For the BB1 dataset:
```
cd scripts
python prepare_data_unsup.py --dir ./../data --boxn 1 --BBk
```

Train the model with the R&D dataset:
```
python train_kmeans_seg.py --log_dir RnD --n_clusters=2 --RD
```
where,
* --n_clusters: Number of clusters to create
* --RD: Expect a data set containing also the true labels. Only used to assess the performance during training.

Make predictions for a subset of the R&D dataset.
```
python evaluate_kmeans_seg.py --log_dir RnD --name UCluster-RnD --n_clusters=2 --RD --full_train
```
Make predictions for the BB1 dataset with labels.
```
python evaluate_kmeans_seg.py --log_dir BB1 --name UCluster-BB1 --n_clusters=2 --box 1 --BBk --full_train
```

# License

MIT License
