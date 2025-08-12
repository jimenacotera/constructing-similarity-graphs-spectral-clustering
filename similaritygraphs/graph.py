'''
This implementation is based on the rbf_graph implementation in the SGTL python package. 
More information can be found here: https://sgtl.readthedocs.io/en/latest/getting-started.html
'''

import math
from typing import List

import scipy
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
import pandas as pd
import random
from multiprocessing import Pool,cpu_count
from functools import reduce
from operator import add

class Graph: 
    def __init__(self, adj_mat):
        """
        Initialise the graph with an adjacency matrix.

        :param adj_mat: A sparse scipy matrix.
        """
        # The graph is represented by the sparse adjacency matrix. We store the adjacency matrix in two sparse formats.
        # We can assume that there are no non-zero entries in the stored adjacency matrix.
        self.adj_mat = adj_mat.tocsr()
        self.adj_mat.eliminate_zeros()
        self.lil_adj_mat = adj_mat.tolil()

        # For convenience, and to speed up operations on the graph, we precompute the degrees of the vertices in the
        # graph.
        self.degrees = adj_mat.sum(axis=0).tolist()[0]
        self.inv_degrees = list(map(lambda x: 1 / x if x != 0 else 0, self.degrees))
        self.sqrt_degrees = list(map(math.sqrt, self.degrees))
        self.inv_sqrt_degrees = list(map(lambda x: 1 / x if x > 0 else 0, self.sqrt_degrees))
        
    def __add__(self, other):
        """
        Adding two graphs requires that they have the same number of vertices. the sum of the graphs is simply the graph
        constructed by adding their adjacency matrices together.

        You can also just add a sparse matrix to the graph directly.
        """
        if isinstance(other, scipy.sparse.spmatrix):
            if other.shape[0] != self.number_of_vertices():
                raise AssertionError("Graphs must have equal number of vertices.")
            # return Graph(self.adjacency_matrix() + other)
            return Graph(self.adjacency_matrix() + other)

        if other.number_of_vertices() != self.number_of_vertices():
            raise AssertionError("Graphs must have equal number of vertices.")

        # return Graph(self.adjacency_matrix() + other.adjacency_matrix())
        return Graph(self.adjacency_matrix() + other.adjacency_matrix())

    def number_of_vertices(self) -> int:
        """The number of vertices in the graph."""
        return self.adjacency_matrix().shape[0]
    

    def total_volume(self) -> float:
        """The total volume of the graph."""
        return sum(self.degrees)

    def inverse_sqrt_degree_matrix(self):
        """Construct the square root of the inverse of the diagonal degree matrix of the graph."""
        return scipy.sparse.spdiags(self.inv_sqrt_degrees, [0], self.number_of_vertices(), self.number_of_vertices(),
                                    format="csr")

    def adjacency_matrix(self):
        """
        Return the Adjacency matrix of the graph.
        """
        return self.adj_mat
    
    def laplacian_matrix(self):
        """
        Construct the Laplacian matrix of the graph. The Laplacian matrix is defined to be

        .. math::
           L = D - A

        where D is the diagonal degree matrix and A is the adjacency matrix of the graph.
        """
        return self.degree_matrix() - self.adjacency_matrix()
    
    def normalised_laplacian_matrix(self):
        """
        Construct the normalised Laplacian matrix of the graph. The normalised Laplacian matrix is defined to be

        .. math::
            \\mathcal{L} = D^{-1/2} L D^{-1/2} =  I - D^{-1/2} A D^{-1/2}

        where I is the identity matrix and D is the diagonal degree matrix of the graph.
        """
        return self.inverse_sqrt_degree_matrix() @ self.laplacian_matrix() @ self.inverse_sqrt_degree_matrix()


    def degree_matrix(self):
        """
        Construct the diagonal degree matrix of the graph.
        """
        return scipy.sparse.spdiags(
            self.degrees, [0], self.number_of_vertices(), self.number_of_vertices(), format="csr")
    
    def average_degree(self):
        """
        Get the average degree of the matrix
        """
        return sum(self.degrees) / len(self.degrees)




#############################################
### FULLY CONNECTED SIMILARITY GRAPH AND KERNELS

def fullyConnected(data, kernelName, threshold=0.1):
    """
    :param data: a sparse matrix with dimension :math:`(n, d)` containing the raw data
    :param kernelName: kernel description
    :param threshold: the threshold under which to ignore the weights of an edge. Set to 0 to keep all edges.
    :return: an `sgtl.Graph` object
    """
    # kernel, hyperParam, max_distance = parseKernelName(kernelName, threshold)
    kernel, hyperParam = parseKernelName(kernelName)
    # Get the maximum distance which corresponds to the threshold specified.pa
    if threshold <= 0:
        # Handle the case when threshold is equal to 0 - need to create a fully connected graph.
        max_distance = float('inf')
    elif kernel == inverse_euclidean: 
        max_distance = math.sqrt(-2 * 10 * math.log(threshold))
    else:
        # max_distance = math.sqrt(-2 * variance * math.log(threshold))
        max_distance = math.sqrt(-2 * hyperParam * math.log(threshold))

    # Create the nearest neighbours for each vertex using sklearn 
    # Neighbours closer than max_distance from given threshold
    distances, neighbours = NearestNeighbors(radius=max_distance).fit(data).radius_neighbors(data)

    #print("[DEBUG] in graph.fullyConnected - data dimensions are ", data.shape)
    # Construct the adjacency matrix of the graph iteratively
    adj_mat = scipy.sparse.lil_matrix((data.shape[0], data.shape[0]), dtype=np.float32)
    for vertex in range(data.shape[0]):
        # Get the neighbours of this vertex
        for i, neighbour in enumerate(neighbours[vertex]):
            if neighbour != vertex:
                distance = distances[vertex][i]
                # weight = math.exp(- (distance**2) / (2 * variance))
                weight = kernel(distance, hyperParam)
                adj_mat[vertex, neighbour] = weight
                adj_mat[neighbour, vertex] = weight

    return Graph(adj_mat)



def parseKernelName(kernelName):
    """
    Helper to get kernel name and hyperparameters
    """
    #print(kernelName)
    if kernelName[:3] == "rbf":
        return rbf, float(kernelName[4:])
    elif kernelName[:3] == "lpl":
        return laplacian, int(kernelName[4:])
    elif kernelName[:3] == "inv":
        return inverse_euclidean, kernelName[4:]
    else:
        print("Wrong kernel name used")
        return None

def rbf(distance, hyperparameter ):
    """
    hyperparam <- variance
    """
    return math.exp(- (distance**2) / (2 * hyperparameter))

def laplacian(distance,hyperparameter):
    """
    hyperparam <- variance
    """
    return math.exp(- (distance) / (math.sqrt(hyperparameter)))

def inverse_euclidean(distance,hyperparameter):
    """
    hyperparam <- "power-epsilon"
    """
    power=int(hyperparameter.split("-")[0])
    # print("power ", power)
    epsilon=float(hyperparameter.split("-")[1])
    # print("epsilon " , epsilon)
    return 1 / (epsilon + distance**power)


#############################################
### SPARSIFIERS


def spectralSparsifier(data, graph_type):
    """
    Following Algorithm 1 by Kent Quanrud as described in
    "Spectral Sparsification of Metrics and Kernels"

    Epsilon Sparsifier
    data: (n,d) dimension sparse matrix
    """

    # Parse graph type : error-epsilon
    err , eps = graph_type.split('-')
    err = float(err)
    eps = float(eps)

    # # Epsilon Sparsifier
    # eps = 7
    # # Variance for kernel
    # err = 0.001
    # d is the dimensions of each pixel
    d = data.shape[1]
    #n = num of pixels
    n = data.shape[0]

    # Random vector where each coordinate is independently 
    # distributed as a standard Gaussian
    # Using newer numpy random generator package 
    vector = np.random.default_rng().standard_normal(d)
    # Compute embedding of data on vector
    y = data.dot(vector)
    permutation = np.argsort(y) 
    # Rank each embedding in vector a
    # Putting embeddings in dataframe to leverage pd.df.rank()
    # df = pd.DataFrame({'y': y})
    # # Compute the rankings breaking ties by the first encountered ranking
    # df['rank'] = (df['y'].rank(method='first') -1).astype(int)


    # Probability distribution of length of difference of projections

     #TODO do i need to make this the actual prob funciton instead of normalising by the sum of the array
    lengths = np.arange(1,n) #1 to n-1
    raw_probs = (n-lengths) / lengths #np.array
    prob_dist = raw_probs / raw_probs.sum()
    # print(prob_dist)
    # Repeat (n logn eps^-2) times
    times = int(n * math.log(n) * (eps**(-2)))

    sampler = np.random.default_rng()
    ls = sampler.choice(a = lengths, p = prob_dist, size=times)

    # empty adjacency matrix
    adj_mat = scipy.sparse.lil_matrix((data.shape[0], data.shape[0]), dtype=np.float32)
    
    #print("[spectral sparsifier] starting sampling")
    #print("[DEBUG] n", n)
    #print("[DEBUG] times", times)
    #print("[DEBUG] eps", eps)

    # counter = 0
    for time in range(times):
        l = ls[time]
        # print(i)
        # 1. Sample edge (i,j) with prob proportional to inv rank difference
        # to do so, sample an interval length with the probability distribution above
        # l = sampler.choice(a = lengths, p = prob_dist)
        # print("sampled lenght: ", l)
        # then sample any rank interval with that length with even probability
        rank_i = random.randint(1, n-l-1)
        rank_j = rank_i + l
        # Get the corresponding datapoints
        i = permutation[rank_i]
        j = permutation[rank_j]
        # create an edge between them 
        # If edge already exists, add weight to it
        weight = spectralKernel(data[i], data[j], err) * l
        adj_mat[i,j] += weight
        adj_mat[j,i] += weight
        # counter += 2


    #print("[spectral sparsifier] Done with spectral sparsifier...")
    
    return Graph(adj_mat)



def spectralSparsifier_multiprocessing(data, graph_type):
    """
    Following Algorithm 1 by Kent Quanrud as described in
    "Spectral Sparsification of Metrics and Kernels"

    Epsilon Sparsifier
    data: (n,d) dimension sparse matrix
    """
    max_cpus = 16
    # Parse graph type : error-epsilon
    err , eps = graph_type.split('-')
    err = float(err)
    eps = float(eps)

    # d is the dimensions of each pixel
    d = data.shape[1]
    #n = num of pixels
    n = data.shape[0]


    # Probability distribution of length of difference of projections
    lengths = np.arange(1,n) #1 to n-1
    raw_probs = (n-lengths) / lengths #np.array
    prob_dist = raw_probs / raw_probs.sum()

    total_iters = int(math.log(n) * (eps**-2))
    #print("[DEBUG] total iters", total_iters)
    num_cpus = min(max_cpus, cpu_count() or 1)
    #print("[DEBUG] num cpus is: ", num_cpus)
    base, rem = divmod(total_iters,num_cpus)
    iters_per_cpu = [base + (1 if i<rem else 0) for i in range(num_cpus)]
    #print("[DEBUG] iters per CPU", iters_per_cpu)

    vector = np.random.default_rng().standard_normal(d)

        # Compute embedding of data on vector
    y = data.dot(vector)
    # Rank each embedding in vector a
    # Putting embeddings in dataframe to leverage pd.df.rank()
    df = pd.DataFrame({'y': y})
    # Compute the rankings breaking ties by the first encountered ranking
    df['rank'] = (df['y'].rank(method='first') -1).astype(int)

    pool_args = [(data, eps, err, lengths, prob_dist, n,d, vector, df) for its in iters_per_cpu]

    with Pool(processes=num_cpus) as pool:
        results = pool.map(spectralSparsifier_multiprocessing_helper, pool_args, chunksize=100)
    
    # join the adjacency matrices 
    adj_mat = scipy.sparse.lil_matrix((data.shape[0], data.shape[0]), dtype=np.float32)

    csr_mats = [matrix.tocsr() for matrix in results]
    sum_csr = reduce(add, csr_mats)
    adj_mat = sum_csr.tolil()

    return Graph(adj_mat)





    


    

# def spectralSparsifier_multiprocessing_helper(data, eps, err, lengths, prob_dist, n, d):
def spectralSparsifier_multiprocessing_helper(args):
    '''
    Sample n times
    '''
    data, eps, err, lengths, prob_dist, n, d, vector, df = args
    # Random vector where each coordinate is independently 
    # distributed as a standard Gaussian
    # Using newer numpy random generator package 
    



    # empty adjacency matrix
    adj_mat = scipy.sparse.lil_matrix((data.shape[0], data.shape[0]), dtype=np.float32)
    # Repeat (n logn eps^-2) times
    #print("[spectral sparsifier] starting sampling")
    # times = int(n * math.log(n) * (eps**(-2)))
    times = n
    #print("[DEBUG] n", n)
    #print("[DEBUG] times", times)
    #print("[DEBUG] eps", eps)

    sampler = np.random.default_rng()

    counter = 0
    for time in range(times):
        # print(i)
        # 1. Sample edge (i,j) with prob proportional to inv rank difference
        # to do so, sample an interval length with the probability distribution above
        l = sampler.choice(a = lengths, p = prob_dist)
        # print("sampled lenght: ", l)
        # then sample any rank interval with that length with even probability
        rank_i = random.randint(1, n-l-1)
        rank_j = rank_i + l
        # Get the corresponding datapoints
        i = df.index[df['rank'] == rank_i][0]
        # i = rank_dict[rank_i] 
        # j = rank_dict[rank_j]
        j = df.index[df['rank'] == rank_j][0]
        # create an edge between them 
        # If edge already exists, add weight to it
        weight = spectralKernel(data[i], data[j], err) * l
        adj_mat[i,j] += weight
        adj_mat[j,i] += weight
        counter += 2

        # Debugging
        # if(time%10000 == 0):
        #     print(time)
        # / Debugging

    #print("[spectral sparsifier] Done with spectral sparsifier...")
    #print("[DEBUG] should have # edges before orthog:", counter)
    # return Graph(adj_mat)
    return adj_mat


def spectralKernel(x_i, x_j, err):
    '''
    Inverse euclidean distance

    x_i: (1,d) matrix??
    '''
    # using euclidean distance
    dist = np.linalg.norm(x_i - x_j)
    p = 1
    return 1 / (err + np.abs(dist)**p)
