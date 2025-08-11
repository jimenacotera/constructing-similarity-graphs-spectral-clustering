import yaml
import os
import subprocess
import pandas as pd


with open('experiment_configurations.yaml') as f: 
    configs = yaml.safe_load(f)


results = pd.DataFrame()

for dataset, experiments in configs['datasets'].items(): 
    print(f"--------------------RUNNING EXPERIMENTS WITH DATASET {dataset}---------------")
    # print(experiments)
    # data = experiment['data']
    data = dataset
    for experiment in experiments:
        experiment_name = experiment['name']
        sim_graph = experiment['sim_graph']

        print("[INFO] Running experiment " + experiment_name + " with dataset: " + data )
        subprocess.run(["python3", "experiments.py", data, sim_graph])

        # run the evaluation 
        if data == "bsds":
            subprocess.run(["python3", "analyseBSDSExperiments.py", experiment_name ])

        # append evaluation to results with the name of the experiment and the configsss


# Save results to a csv with the experiment date