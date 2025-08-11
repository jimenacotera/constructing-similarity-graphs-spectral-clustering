# Constructing Similarity Graph for Spectral Clustering

This code is set up to reproduce the results obtained in my dissertation project _Constructing Similarity Graphs for Spectral Clustering_, submitted in fulfillment of the requirements for the degree of MSc in Artificial Intelligence. This project was supervised by Dr. Peter Macgregor. 


## How to Run Experiments

First run the following from the main directory to install the dependencies of this project: 

```bash
pip install -r requirements.txt
```

Then run the following to execute all experiments.

```bash
python runAllExperiments.py
```

## System Requirements

Please note that systems will less than 32GB of RAM will most likely be unable to run the full set of experiments. 


## Running only one experiment
The following command can be used in order to run just one experiment. 

```bash
python experiments.py {dataset} {similaritygrahp}{bsds_image_id}
```

The possible options for ``{dataset}`` are: 
- bsds
- mnist


The value of ``{similaritygraph}`` can be any of the following: 
- fcn-rbf-{variance}
- fcn-lpl-{variance}
- fcn-inv
- knn{valueofk}
- sparsifier-clus-{gamma}
- sparsifier-spec-{kernelval}-{epsilon}
