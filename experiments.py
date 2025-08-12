"""
Run experiments with the new spectral clustering approach.
"""
import multiprocessing
import time
import argparse
import scipy as sp
import scipy.sparse
import scipy.io
import numpy
from multiprocessing import Process, Queue
import os
import os.path
import math
import sgtl
import sgtl.random
import sgtl.clustering
import pysc.objfunc
import pysc.sc
import pysc.datasets
import pysc.evaluation
import pysc.objfunc
from pysc.sclogging import logger
import pandas as pd

import similaritygraphs.graph



def basic_experiment_sub_process(dataset, num_clusters, num_eigenvalues: int, q):
    # logger.info(f"Starting clustering: {dataset} with {num_eigenvalues} eigenvalues.")
    print(f"Starting clustering: {dataset} with {num_eigenvalues} eigenvalues and num_clusters = {num_clusters}.")
    start_time = time.time()
    found_clusters = sgtl.clustering.spectral_clustering(dataset.graph, num_clusters=num_clusters,
                                                         num_eigenvectors=num_eigenvalues)
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Finished clustering: {dataset} with {num_eigenvalues} eigenvalues.")
    this_rand_score = pysc.evaluation.adjusted_rand_index(dataset.gt_labels, found_clusters)
    this_mutual_info = pysc.evaluation.mutual_information(dataset.gt_labels, found_clusters)
    # this_conductance = pysc.objfunc.KWayExpansion.apply(dataset.graph, found_clusters)
    # q.put((num_eigenvalues, this_rand_score, this_mutual_info, this_conductance, total_time))
    q.put((num_eigenvalues, this_rand_score, this_mutual_info, None, total_time))


def basic_experiment(dataset, num_clusters):
    """
    Run a basic experiment with a given dataset, in which we do spectral clustering with a variety of numbers of
    eigenvectors, and compare the resulting clustering.

    :param dataset: A `pysc.datasets.Dataset` object.
    :param num_clusters: The number of clusters.
    """
    logger.info(f"Running basic experiment with {dataset.__class__.__name__}.")

    # Start all of the sub-processes to do the clustering with different numbers of eigenvalues
    rand_scores = {}
    mutual_info = {}
    conductances = {}
    times = {}
    q = Queue()
    processes = []

    num_eigenvalues = num_clusters

    p = Process(target=basic_experiment_sub_process, args=(dataset, num_clusters, num_eigenvalues, q))
    p.start()
    processes.append(p)

    logger.info(f"All sub-processes started for {dataset}.")

    for p in processes:
        p.join()

    logger.info(f"All sub-processes finished for {dataset}.")

    rand_score = 0
    # Get all of the data from the subprocesses
    while not q.empty():
        #print("[DEBUG] in basic_experiment loop - segmentation ran one time at least")
        num_vectors, this_rand_sc, this_mut_info, this_conductance, this_time = q.get()
        rand_score = this_rand_sc
        mutual_info[num_vectors] = this_mut_info
        # conductances[num_vectors] = this_conductance
        times[num_vectors] = this_time

    # return rand_scores, mutual_info, conductances, times
    return rand_score, mutual_info, None, times
    # return this_rand_sc, mutual_info, conductances, times

def basic_experiment_no_multiprocessing(dataset, num_clusters):
    '''
    Run a basic experiment as above without multiprocessing
    '''
    num_eigenvalues = num_clusters

    # Start the clustering
     # this_mutual_info = pysc.evaluation.mutual_information(dataset.gt_labels, found_clusters)
    # this_conductance = pysc.objfunc.KWayExpansion.apply(dataset.graph, found_clusters)
    # q.put((num_eigenvalues, this_rand_score, this_mutual_info, this_conductance, total_time))
    # q.put((num_eigenvalues, this_rand_score, this_mutual_info, None, total_time))
    # num_vectors = num_eigenvalues

    # if bsds_dataset.graph_type.startswith("sparsifier-clus"):
    #     # Cluster preserving sparsifier
    #     print("Getting the laplacian for cluster preserving")
    #     laplacian_matrix = bsds_dataset.graph.normalised_laplacian().to_scipy()
    # else:
    #     laplacian_matrix = bsds_dataset.graph.normalised_laplacian_matrix()

    
    if dataset.graph_type.startswith("sparsifier-clus"):
        # Cluster preserving sparsifier
        all_segmentations = []
        #print("Getting the laplacian for cluster preserving")
        laplacian_matrix = dataset.graph.normalised_laplacian().to_scipy()
        _, eigvecs = scipy.sparse.linalg.eigsh(laplacian_matrix, num_eigenvalues, which='SM')

        #print(f"Segmenting MNIST into {num_clusters} segments with {num_eigenvalues} eigenvectors.")
        found_clusters = pysc.sc.sc_precomputed_eigenvectors(eigvecs, num_clusters=num_clusters, num_eigenvectors=num_eigenvalues)
        all_segmentations.append(found_clusters)
        this_rand_score = pysc.evaluation.adjusted_rand_index(dataset.gt_labels, found_clusters)

        #print("ARI ", this_rand_score)
        

    else: 
        found_clusters = sgtl.clustering.spectral_clustering(dataset.graph, num_clusters=num_clusters,
                                                         num_eigenvectors=num_eigenvalues)
        this_rand_score = pysc.evaluation.adjusted_rand_index(dataset.gt_labels, found_clusters)
   


    return this_rand_score


############################################
# Experiments on MNIST and USPS
############################################
def run_mnist_experiment(graph_type):
    """
    Run experiments on the MNIST dataset.
    """
    num_clusters_mnist = 10

    # Parse graph type argument to creat ap
    # k = None
    # if graph_type[:3] == "knn":
    #     k = int(graph_type[3:])
    # elif 
    # print(graph_type)

    # experiment_stats = []
    # Run experiment
    downsample = 14
    dataset =  pysc.datasets.MnistDataset(downsample=downsample, graph_type=graph_type)
    #print("[DEBUG] the graph type in experiments.py is: " , type(dataset.graph))

    start = time.perf_counter()
    # this_rand_scores, this_mut_info, this_conductances, this_times = basic_experiment(dataset=dataset, num_clusters=num_clusters_mnist)
    this_rand_scores = basic_experiment_no_multiprocessing(dataset=dataset, num_clusters=num_clusters_mnist)
    duration = time.perf_counter() - start
    # q.put((d, nn, this_rand_scores, this_mut_info, this_conductances, this_times))

    graph_size = dataset.getGraphSize()
    if graph_type[:3] == "knn": 
        avg_deg = int(graph_type[3:])
    else : 
        avg_deg = dataset.getAverageDegree()
        
    #print("[DEBUG] avg deg", avg_deg)
    # Write results to csv file
    experiment_stats = {'dataset': 'MNIST'
                        , 'ARI': this_rand_scores
                        , 'numClusters': num_clusters_mnist
                        , 'downsample': 0 if downsample is None else downsample
                        , 'graphSize': graph_size
                        , 'averageDegree': avg_deg
                        , 'graphType': graph_type
                        , 'duration': duration }
    
    output_filename = output_filename = "results/mnist/results.csv"

    #print("[DEBUG] ARI score", this_rand_scores )

    # Read existing CSV 
    if os.path.exists(output_filename):
        df = pd.read_csv(output_filename)
    else:
        df = pd.DataFrame(columns=experiment_stats.keys())
    # chack if graphType in existing csv
    # mask = df['graphType'] == experiment_stats['graphType']
        
    mask = (
        (df['graphType'] == experiment_stats['graphType']) &
        (df['downsample'] == experiment_stats['downsample'])
    )
    
    if mask.any():
        for col, val in experiment_stats.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([experiment_stats])], ignore_index=True)


    # stats_df = pd.DataFrame(experiment_stats)
    df.to_csv(output_filename, index=False)



def usps_experiment_instance(d, nn, q):
    this_rand_scores, this_mut_info, this_conductances, this_times = basic_experiment(
        pysc.datasets.UspsDataset(k=nn, downsample=d), 10)
    q.put((d, nn, this_rand_scores, this_mut_info, this_conductances, this_times))

def run_usps_experiment():
    """
    Run experiments on the USPS dataset.
    """

    # We will construct the 3-NN graph
    k = 3

    # Kick off the experiment ina sub-process
    q = Queue()
    p = Process(target=usps_experiment_instance, args=(None, k, q))
    p.start()
    p.join()

    # Write out the results
    with open("results/usps/results.csv", 'w') as fout:
        fout.write("k, d, eigenvectors, rand\n")

        while not q.empty():
            downsample, k, rand_scores, _ ,_, _ = q.get()
            for i in range(2, 11):
                fout.write(f"{k}, {downsample}, {i}, {rand_scores[i]}\n")


############################################
# Experiments on synthetic data
############################################
class SBMJobRunner(Process):

    def __init__(self, k, n, prob_p, prob_q, queue, num_runs=1, use_grid=False, **kwargs):
        super(SBMJobRunner, self).__init__(**kwargs)
        self.k, self.n, self.prob_p, self.prob_q = k, n, prob_p, prob_q
        self.queue = queue
        self.num_runs = num_runs
        self.use_grid = use_grid
        self.d = self.k
        if use_grid:
            self.k = self.d * self.d

    def run(self) -> None:
        # We will run the whole experiment self.num_runs times, and take the average results.
        total_conductances = [0] * self.k
        total_rand_scores = [0] * self.k

        for run_no in range(self.num_runs):
            if self.use_grid:
                dataset = pysc.datasets.SBMGridDataset(d=self.d, n=self.n, p=self.prob_p, q=self.prob_q)
            else:
                dataset = pysc.datasets.SbmCycleDataset(k=self.k, n=self.n, p=self.prob_p, q=self.prob_q)
            logger.info(f"Starting experiment: {dataset}, run number {run_no}")

            # Pre-compute the eigenvectors of the graph
            laplacian_matrix = dataset.graph.normalised_laplacian_matrix()
            _, eigvecs = scipy.sparse.linalg.eigsh(laplacian_matrix, self.k+1, which='SM')

            for num_eigenvalues in range(1, self.k + 1):
                found_clusters = pysc.sc.sc_precomputed_eigenvectors(eigvecs, self.k, num_eigenvalues)
                logger.info(f"Finished clustering: {dataset}, run number {run_no} with {num_eigenvalues} eigenvectors.")
                total_conductances[num_eigenvalues - 1] += pysc.objfunc.KWayExpansion.apply(dataset.graph, found_clusters)
                total_rand_scores[num_eigenvalues - 1] += pysc.evaluation.rand_index(dataset.gt_labels, found_clusters)

            logger.info(f"Finished experiment: {dataset}, run number {run_no}")

        # Get the average, and submit to the queue
        for num_eigenvalues in range(1, self.k + 1):
            self.queue.put((self.k, self.n, self.prob_p, self.prob_q, num_eigenvalues,
                            total_conductances[num_eigenvalues - 1] / self.num_runs,
                            total_rand_scores[num_eigenvalues - 1] / self.num_runs))

        # let the calling code know that we're done
        self.queue.put(None)


def run_sbm_experiment(n, k, prob_p, use_grid=False):
    logger.info(f"Running experiment with SBM data.")

    # For each set of SBM parameters, run 10 times.
    num_runs = 10

    # Start all of the sub-processes to do the clustering with different numbers of eigenvalues
    processes = []
    if use_grid:
        results_filename = "results/sbm/grid_results.csv"
    else:
        results_filename = "results/sbm/cycle_results.csv"
    with open(results_filename, 'w') as fout:
        fout.write("k, n, p, q, poverq, eigenvectors, conductance, rand\n")
        fout.flush()
        for prob_q in numpy.linspace(prob_p / 10, prob_p, num=10):
            q = Queue()
            p = SBMJobRunner(k, n, prob_p, prob_q, q, num_runs=num_runs, use_grid=use_grid)
            p.start()
            processes.append(p)

            # Keep at most 20 sub-processes
            if len(processes) >= 20:
                for p in processes:
                    # Save all of the data to the output file
                    while True:
                        process_result = p.queue.get()

                        if process_result is None:
                            break
                        else:
                            k, n, prob_p, prob_q, num_vectors, this_conductance, this_rand_score = process_result
                            fout.write(f"{k}, {n}, {prob_p}, {prob_q}, {prob_p/prob_q}, {num_vectors}, {this_conductance}, {this_rand_score}\n")
                            fout.flush()
                    p.join()
                processes = []

        logger.info(f"All sub-processes started for sbm experiments.")

        for p in processes:
            # Save all of the data to the output file
            while True:
                process_result = p.queue.get()

                if process_result is None:
                    break
                else:
                    k, n, prob_p, prob_q, num_vectors, this_conductance, this_rand_score = process_result
                    fout.write(f"{k}, {n}, {prob_p}, {prob_q}, {prob_p/prob_q}, {num_vectors}, {this_conductance}, {this_rand_score}\n")
                    fout.flush()
            p.join()

    logger.info(f"All sub-processes finished for sbm experiments.")


############################################
# Experiments on BSDS
############################################
def segment_bsds_image(bsds_dataset, num_segments, num_eigenvectors_l):
    """
    Given a loaded bsds dataset, find a segmentation into the given number of segments.

    :param bsds_dataset: the already loaded bsds_dataset
    :param num_segments: the number of segments to find
    :param num_eigenvectors_l: a list with different numbers of eigenvectors to use to find the segmentation
    :return: a list of segmentations
    """
    all_segmentations = []

    #print("Computing eigenvectors ...")
    # First, compute all of the eigenvectors up front 
    if bsds_dataset.graph_type.startswith("sparsifier-clus"):
        # Cluster preserving sparsifier
        #print("Getting the laplacian for cluster preserving")
        laplacian_matrix = bsds_dataset.graph.normalised_laplacian().to_scipy()
    else:
        laplacian_matrix = bsds_dataset.graph.normalised_laplacian_matrix()
    _, eigvecs = scipy.sparse.linalg.eigsh(laplacian_matrix, max(num_eigenvectors_l), which='SM')
    # _, eigvecs = scipy.sparse.linalg.eigsh(laplacian_matrix, num_eigenvectors_l, which='SM')

    #print(f"Segmenting {bsds_dataset} into {num_segments} segments with {num_eigenvectors_l} eigenvectors.")
    found_clusters = pysc.sc.sc_precomputed_eigenvectors(eigvecs, num_segments, num_eigenvectors_l[0])
    all_segmentations.append(found_clusters)

    # for num_eigenvectors in num_eigenvectors_l:
    #     # debug
    #     print(f"Segmenting {bsds_dataset} into {num_segments} segments with {num_eigenvectors} eigenvectors.")
    #     found_clusters = pysc.sc.sc_precomputed_eigenvectors(eigvecs, num_segments, num_eigenvectors)
    #     all_segmentations.append(found_clusters)

    # print("found this num of segmentations: " , len(all_segmentations))
    return all_segmentations


def save_bsds_segmentations(bsds_dataset, segmentations, eigenvectors_l, filename, upscale=True):
    """
    Save the given segmentation of the given dataset to the given filename.
    Save in matlab file format so the analysis can be done with matlab tools.

    :param bsds_dataset: the bsds image in question, as a dataset object
    :param segmentations: a list of segmentations
    :param eigenvectors_l: a list of the number of eigenvectors used for each segmentation
    :param filename: the filename to save the segmentation to
    :param upscale: whether to scale the segmentation back up to the original size
    :return:
    """
    seg_cell = numpy.empty((len(segmentations, )), dtype=object)
    eigs_cell = numpy.empty((len(segmentations, )), dtype=object)

    for seg_i, segmentation in enumerate(segmentations):
        # Construct the labels rather than the list of clusters.
        pixel_labels = [0] * bsds_dataset.num_data_points
        for i, segment in enumerate(segmentation):
            for pixel in segment:
                pixel_labels[pixel] = i

        # Construct the labelled image with the downsampled dimensions
        labelled_image = numpy.array(pixel_labels, dtype="int32")
        # labelled_image = numpy.reshape(labelled_image, bsds_dataset.downsampled_image_dimensions) + 1
        if bsds_dataset.graph_type.startswith("sparsifier-spec"): #TODO this is so i can downsample clus
            labelled_image = numpy.reshape(labelled_image, bsds_dataset.original_image_dimensions) + 1
            seg_cell[seg_i] = labelled_image
        else:
            labelled_image = numpy.reshape(labelled_image, bsds_dataset.downsampled_image_dimensions) + 1

            # Scale up the segmentation by taking the appropriate tensor product
            labelled_image_upsample =\
                numpy.kron(labelled_image, numpy.ones((bsds_dataset.downsample_factor, bsds_dataset.downsample_factor)))
            labelled_image_upsample = labelled_image_upsample[:bsds_dataset.original_image_dimensions[0],
                                                            :bsds_dataset.original_image_dimensions[1]]

            seg_cell[seg_i] = labelled_image_upsample if upscale else labelled_image
       
        eigs_cell[seg_i] = eigenvectors_l[seg_i]

    # Save the labelled image to the given file
    data_to_save = {'segs': seg_cell, 'eigs': eigs_cell}
    sp.io.savemat(filename, data_to_save)


def get_bsd_num_cluster(gt_filename) -> int:
    """
    Given the path to the filename containing the ground truth clustering for an element of the bsds dataset, get the
    number of clusters that we should try to find.

    :param gt_filename:
    :return:
    """
    # Take the minimum number of clusters from the ground truth segmentations.
    gt_data = sp.io.loadmat(gt_filename)
    num_ground_truth_segmentations = gt_data["groundTruth"].shape[1]

    nums_segments = []
    for i in range(num_ground_truth_segmentations):
        this_segmentation = gt_data["groundTruth"][0, i][0][0][0]
        this_num_segments = numpy.max(this_segmentation)
        nums_segments.append(this_num_segments)

    # Return the median number of segments, and at least 2.
    return max(2, int(numpy.median(nums_segments)))


# def run_bsds_experiment( graph_type, hyperparam_0, image_id=None):
def run_bsds_experiment(graph_type, image_id=None):
    """
    Run experiments on the BSDS dataset.
    :image_files: a list of the BSDS image files to experiment with
    :return:
    """
    if image_id is None:
        # If no image file is provided, then run the experiment on all image files in the test data.
        ground_truth_directory = "data/bsds/BSR/BSDS500/data/groundTruth/test/"
        images_directory = "data/bsds/BSR/BSDS500/data/images/test/"
        output_directory = "results/bsds/segs/"
        #image_files = os.listdir(images_directory)
        # avoid segmenting on metadata and hidden files
        image_files = [f for f in os.listdir(images_directory) if not f.startswith('.') and f.endswith('.jpg')]
        # #debug
        # print(image_files)
    else:
        # If an image filename is provided, then work out whether it is in the test or training data
        image_filename = image_id + '.jpg'
        images_directory = "data/bsds/BSR/BSDS500/data/images/test/"
        if image_filename in os.listdir(images_directory):
            ground_truth_directory = "data/bsds/BSR/BSDS500/data/groundTruth/test/"
            output_directory = "results/bsds/segs/"
            image_files = [image_filename]
        else:
            images_directory = "data/bsds/BSR/BSDS500/data/images/train/"
            ground_truth_directory = "data/bsds/BSR/BSDS500/data/groundTruth/train/"
            output_directory = "results/bsds/segs/"
            image_files = [image_filename]

            if image_filename not in os.listdir(images_directory):
                # If the target file is not in the training directory, then it's a lost cause.
                raise Exception("BSDS image ID not found.")
    
    experiment_stats = []
    
    for i, file in enumerate(image_files):
        
        id = file.split(".")[0]
        #print(f"Running BSDS experiment with image {file}. (Image {i+1}/{len(image_files)})")
        k = get_bsd_num_cluster(os.path.join(ground_truth_directory, f"{id}.mat"))
        
        start = time.perf_counter()


        num_eigenvectors_l = [k]

        # Check if dataset for sparse graph or not
        if graph_type[:3] == "spa":
            dataset = pysc.datasets.BSDSDatasetSparsifier(id, graph_type=graph_type, data_directory=images_directory)
        else: 
            # dataset = pysc.datasets.BSDSDataset(id, blur_variance=0, graph_type=graph_type, hyperparam_0=hyperparam_0, data_directory=images_directory)
            dataset = pysc.datasets.BSDSDataset(id, blur_variance=0, graph_type=graph_type, data_directory=images_directory)

        
        if graph_type[:3] == "knn": 
            avg_degree = int(graph_type[3:])
        else : 
            avg_degree = dataset.getAverageDegree()
        # #print("[DEBUG]avg degree: " , avg_degree)
        #print("about to segment image")
        segmentations = segment_bsds_image(dataset, k, num_eigenvectors_l)

        # Record segmentation time and graph size
        duration = time.perf_counter() - start
        size = dataset.getGraphSize()
        # avg_degree = dataset.getAverageDegree()
        # print(avg_degree)
        experiment_stats.append({'image': id
                                 , 'duration': duration
                                 , 'graphSize': size
                                 , 'averageDegree': avg_degree
                                 , 'graphType': dataset.graph_type})

        # Save the downscaled image
        output_filename = f"results/bsds/downsamples/{dataset.img_idx}.jpg"
        # dataset.save_downsampled_image(output_filename)

        output_filename = f"results/bsds/segs/{dataset.img_idx}.mat"
        save_bsds_segmentations(dataset, segmentations, num_eigenvectors_l, output_filename)
        # # Save the upscaled segmentations
        # for i, num_eigenvectors in enumerate(num_eigenvectors_l):
        #     output_filename = f"results/bsds/segs/{dataset.img_idx}.mat"
        #     save_bsds_segmentations(dataset, segmentations, num_eigenvectors_l, output_filename)


        output_filename = f"results/bsds/downsampled_segs/{dataset.img_idx}.mat"
        save_bsds_segmentations(dataset, segmentations, num_eigenvectors_l, output_filename, upscale=False)

        # Save the downscaled segmentation
        # for i, num_eigenvectors in enumerate(num_eigenvectors_l):
        #     output_filename = f"results/bsds/downsampled_segs/{dataset.img_idx}.mat"
        #     save_bsds_segmentations(dataset, segmentations, num_eigenvectors_l, output_filename, upscale=False)

        #break #TODO debugging

    # Save image runtimes to csv
    runtimes_df = pd.DataFrame(experiment_stats)
    output_filename = "results/bsds/csv_results/experimentStats.csv"
    runtimes_df.to_csv(output_filename, index = False)

#### Input parsing

def parse_args():
    parser = argparse.ArgumentParser(description='Run the experiments.')
    parser.add_argument('experiment', type=str, choices=['cycle', 'grid', 'mnist', 'usps', 'bsds'],
                        help="which experiment to perform")
    parser.add_argument('graph_type', type=str, help = "Type of similarity graph")
    # parser.add_argument('hyperparam_0', type=str, help = "Config type for specified similarity graph")
    parser.add_argument('bsds_image', type=str, nargs='?', help="(optional) the BSDS ID of a single BSDS image file to segment")
    # parser.add_argument('bsds_image', type=str, help="(optional) the BSDS ID of a single BSDS image file to segment")
   
    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiment == 'cycle':
        run_sbm_experiment(1000, 10, 0.01)
    elif args.experiment == 'grid':
        run_sbm_experiment(1000, 4, 0.01, use_grid=True)
    elif args.experiment == 'mnist':
        run_mnist_experiment(graph_type = args.graph_type)
    # elif args.experiment == 'usps':
    #     run_usps_experiment()
    elif args.experiment == 'bsds':
        if args.bsds_image is None:
            logger.warning("\nThe BSDS experiment is very resource-intensive. We recommend running on a compute server.")
            logger.info("Waiting 10 seconds before starting the experiment...")
            time.sleep(10)
            # run_bsds_experiment()
            # run_bsds_experiment(graph_type=args.graph_type, hyperparam_0 = args.hyperparam_0)
            run_bsds_experiment(graph_type=args.graph_type)
        else:
            # run_bsds_experiment(image_id=args.bsds_image, graph_type=args.graph_type, hyperparam_0=args.hyperparam_0)
            run_bsds_experiment(image_id=args.bsds_image, graph_type=args.graph_type)
    else: 
        raise ValueError("Invalid dataset")

if __name__ == "__main__":
    main()
