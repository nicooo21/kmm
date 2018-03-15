"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *
import sys

######################################################################
# helper functions
######################################################################

# keeps track of Cluster Update
class ClusterUpdate(object):
    
    # inits with initial array of cluster positions, assigns new clusters with each initial point
    def __init__(self, clusterInit, k):
       
        self.clusters = []
        print 'initializing centers...'
        for c in range(0, k):
            self.clusters.append(Cluster(clusterInit[c]))
        
    
    def add_to_clusters(self, points):
        # finds cluster that point is closest to
        # for each point
        for p in points:
            # initialize minimum distance
            minDistance = 10000000000
            # set current cluster to none
            curCluster = -1
            # for each of the clusters

            numCluster = 0
            for cluster in self.clusters:
                print p.attrs
                # compute distance to each of the cluster centroids
                print numCluster
                print cluster.center.attrs
                curDistance = cluster.center.distance(p)
                print curDistance
                # if there is currently no cluster or curDistance is less than minDistance
                if(curDistance < minDistance):
                    minDistance = curDistance
                    print "minDistance: %.3f" % minDistance
                    curCluster = numCluster

                numCluster += 1
            print "cluster: %d" % curCluster
            print minDistance
            print "======"
            self.clusters[curCluster].points.append(p)
        print len(self.clusters[0].points)
        print len(self.clusters[1].points)
        print len(self.clusters[2].points)


def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets
    
    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """
    
    n,d = X.shape
    
    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])
    
    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """
    
    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = []
    if average == 'random':
        centroids = clusters.centroids()
    if average == 'cheat':
        centroids = clusters.medoids()
    
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.
    
    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed
    
    Returns
    --------------------
        points -- list of Points, dataset
    """

    np.random.seed(seed)
    
    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
    
    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))
    
    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part c: implement (hint: use np.random.choice)
    
    # Uses np.random, which first:
    # first parameter, a, being points array
    # second parameter, size, being k
    # third parameter, replace, being false, to avoid replacement
    # fourth parameter, none

    return np.random.choice(points, k, False)

    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!
    
    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part f: implement
    clusterList = []

    for x in range(0, 3):
        clusterList.append(Cluster(None))
    
    for p in points:
        clusterList[p.label - 1].points.append(p)
    
    initial_points = []

    for x in range(0, 3):
        initial_points.append(clusterList[x].medoid())

    return initial_points
    ### ========== TODO : END ========== ###


def kMeans(points, k, init='random', plot=False) :
    
    """
    Cluster points into k clusters using variations of k-means algorithm.
    
    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable: 
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm
    
    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """
    
    ### ========== TODO : START ========== ###
    # part c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).

    # Initialize ClusterSet object
    # Create previous Cluster Set to be able to check when clustering no longer changes
    curClusterSet = ClusterSet()
    prevClusterSet = ClusterSet()

    # check which initialization parameter was used

    cluster_init = []

    # for using centroids
    if init == 'random':
        cluster_init = random_init(points, k)

    # keeps track of iteration number
    iterations = 1

    # basically a do while loop
    while True:
        # creates update that initializes centers of clusters

            
        cUpdate = ClusterUpdate(cluster_init, k)

        # adds each point to closest
        cUpdate.add_to_clusters(points)

        # sets current cluster set to clusters of ClusterUpdate
        curClusterSet.members = cUpdate.clusters
        
        # checks if it is centroid or metoid
        if init == 'random':
            cluster_init = curClusterSet.centroids()
        
        print cluster_init[0]
        print cluster_init[1]
        print cluster_init[2]

        # plots iteration
        

        

        if curClusterSet.equivalent(prevClusterSet) is False:
            prevClusterSet.members = curClusterSet.members
            if plot is True:
                title = 'Iteration: %d' % iterations
                plot_clusters(curClusterSet, title, init)
            iterations += 1
        else:
            break

    return curClusterSet
    ### ========== TODO : END ========== ###


def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part e: implement
    curClusterSet = ClusterSet()
    prevClusterSet = ClusterSet()

    # check which initialization parameter was used

    cluster_init = []

    # for using centroids
    if init == 'cheat':
        cluster_init = cheat_init(points)

    # keeps track of iteration number
    iterations = 1

    # basically a do while loop
    while True:
        # creates update that initializes centers of clusters

            
        cUpdate = ClusterUpdate(cluster_init, k)

        # adds each point to closest
        cUpdate.add_to_clusters(points)

        # sets current cluster set to clusters of ClusterUpdate
        curClusterSet.members = cUpdate.clusters
        
        # checks if it is centroid or metoid
        if init == 'cheat':
            cluster_init = curClusterSet.medoids()

        

        if curClusterSet.equivalent(prevClusterSet) is False:
            prevClusterSet.members = curClusterSet.members

            # plots iteration
            if plot is True:
                title = 'Iteration: %d' % iterations
                plot_clusters(curClusterSet, title, init)

            iterations += 1
        else:
            break

    return curClusterSet
    ### ========== TODO : END ========== ###



######################################################################
# main
######################################################################

def main() :
    
    ### ========== TODO : START ========== ###
    # part d, part e, part f: cluster toy dataset
    np.random.seed(1234)
    
    # use generate_points to get points

    points = generate_points_2d(20)

    kMeans(points, 3, 'random', True)

    kMedoids(points, 3, init='cheat', plot=True)

    ### ========== TODO : END ========== ###
    
    


if __name__ == "__main__" :
    main()
