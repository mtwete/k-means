"""
Matthew Twete

Implementation of the k-means algorithm on a cluster dataset.
"""

#Import needed libraries
import numpy as np
import random
import matplotlib.pyplot as plt

#Import the data
data = np.genfromtxt('cluster_dataset.txt')


#K-means algorithm class, allowing you to pass in clustering data and set the value
#of k and how many runs you want the algorithm to do. 
class kmeans:
    #K-means algorithm class constructor. The arguments are:
    #data is the data to be clustered
    #k is the number of clusters to separate the data into
    #r is the number of times to run the algorithm
    def __init__(self, Data, k, r):
        #Clustering data
        self.data = Data
        #Number of clusters
        self.k = k
        #Number of runs
        self.r = r
        #Data structure to hold the cluster centroids for each run
        self.centroids = np.ones((r,k,data.shape[1]))
        #Data structure to hold the cluster centroids at every iteration for each run
        self.prevcentroids = [[] for _ in range(self.r)]
        #Array to hold the final sum squared error for each run
        self.final_error = np.zeros(r)
        #Data structure to hold the data points in each cluster for plotting
        self.clusters = [[] for _ in range(self.k)]
    
    
    #Function to calculate the distance of each data point from each cluster centroid. 
    #The only arguments is:
    #means, array containing the centroids of each cluster
    def calc_dist(self,means):
        #Set up data structure to hold the distances and calculate the distance of each point from the first cluster centroid
        self.dist = np.sum((data-means[0][:])**2,axis=1).reshape(-1,1)
        #Calculate the data point distances from the rest of the centroids 
        for l in range(1,self.k):
            self.dist = np.append(self.dist,np.sum((data-means[l][:])**2,axis=1).reshape(-1,1),axis=1)
    
    
    #Function to re-calculate the cluster centroids based on data points in that cluster.
    #The only arguments is:
    #i, an integer representing which run the algorithm is on, zero indexed
    def recalc_centroids(self,i):
        #Get the closest cluster of each data point
        closest = self.dist.argmin(axis=1)
        #Loop over the clusters
        for l in range(self.k):
            #Get the data points that are in the current cluster
            currentClust = np.where(closest==l,1,0)
            #Calculate the new centroid
            self.centroids[i][l] = np.sum(currentClust.reshape(-1,1)*self.data,axis=0)/np.sum(currentClust)


    #Function to run the algorithm, it will handle all the calcualations and display the results
    def run(self):
        #Run the algorithm r times
        for i in range(self.r):
            #Pick initial centroids at random from the data
            for j in range(self.k):
                self.centroids[i][j] = random.choice(self.data)
            #Set up data structure to hold the previous iteration's centroids, to be used in the stopping condition
            prevcent = np.zeros(self.centroids[i].shape)
            #Loop until the previous iteration centroids are the same as the current centroids
            while(np.sum(self.centroids[i]-prevcent) != 0):
                #Save the centroids so they can be used to plot the algorithm at different iterations
                self.prevcentroids[i].append(np.copy(self.centroids[i])) 
                #Save the previous iteration's centroids
                prevcent = np.copy(self.centroids[i])
                #Recalculate the centroids
                self.calc_dist(self.centroids[i])
                self.recalc_centroids(i)

        #Calculate the sum squared error for each run
        self.calc_errors()
        #Plot the results from the best run
        self.plot_results()
        #Print the sum squared error of the best run
        print("Sum Squared Error of the best k-means model with " + str(self.k) + " clusters: " + str(np.amin(self.final_error)))
    
    
    #Function to calculate the sum squared error for given centroid values, the function will
    #return the sum squared error of the data for the given cluster centroids. The only arguments is:
    #means, array containing the centroids of each cluster
    def sum_sq_er(self,means):
        #Calculate the distance of each data point from each cluster
        self.calc_dist(means)
        #Variable to hold the sum squared error
        sse = 0
        #Get the closest cluster of each data point
        closest = self.dist.argmin(axis=1)
        #Loop over the clusters
        for l in range(self.k):
            #For the current cluster, get the data points in the cluster
            currentClust = np.where(closest==l,1,0)
            #Subtract the positions of the centroid from each data point
            delta = data-means[l][:]
            #Pick out only the data points in that cluster
            delta = currentClust.reshape(-1,1)*delta
            #Add the L2 squared distance of each of the data points in the cluster
            for i in range(self.data.shape[0]):
                sse += np.linalg.norm(delta[i])**2
        return sse
    
    
    #Function to separate the data into the clusters that they are closest to.
    #The only arguments is:
    #means, array containing the centroids of each cluster
    def partition_clusters(self,means):
        #Calculate the distance of each data point from each cluster
        self.calc_dist(means)
        #Get the closest cluster of each data point
        closest = self.dist.argmin(axis=1)
        #Loop over the clusters
        for l in range(self.k):
            #For the current cluster, get the data points in the cluster
            currentClust = np.where(closest==l,1,0)
            #Set all data points not in the cluster to 0
            clusterPoints = currentClust.reshape(-1,1)*data
            #Pick out only the data points in the cluster (ie the ones not equal to 0)
            clusterPoints = clusterPoints[~np.all(clusterPoints == 0,axis=1)]
            #Store those data points
            self.clusters[l] = clusterPoints
    
    
    #Function to calculate the sum squared error for each run of the algorithm.
    def calc_errors(self):
        #Loop over the final centroids of each run and calculate the sum squared error
        for i in range(self.r):
            self.final_error[i] = self.sum_sq_er(self.centroids[i])
    
    
    #Function to plot the results of the algorithm. It will plot the cluster centroids and data points
    #in those clusters from the best run of the algorithm. It will plot the initial centroids and clusters,
    #the centroids and clusters from the iteration in the middle of the run and then the final centroids and 
    #clusters.
    def plot_results(self):
        #Set up plots
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #Get the index of the best run of the algorithm
        bestrun = np.argmin(self.final_error)
        #Get the initial centroids from the best run of the algorithm, and separate the data into the clusters
        self.best_centroid = self.prevcentroids[bestrun][0]
        self.partition_clusters(self.best_centroid)
        #Loop over each cluster (except the last) and plot the centroid and data points in that cluster.
        #This is done so that each cluster's data points have a different color. 
        for i in range(self.k-1):
            ax1.scatter(self.clusters[i][:,0],self.clusters[i][:,1], s = np.full(len(self.clusters[i]),3),label = "Cluster " + str(i+1) + " data")
            ax1.scatter(self.best_centroid[i][0],self.best_centroid[i][1],marker='v',c=1)
        #Plot the centroid and data of the last cluster, the reason this is done separately is simply to make the 
        #plot legend formatted more neatly, since centroids are all plotted as the same shape and color and I want the 
        #legend to only have one row for the centroid points
        ax1.scatter(self.clusters[self.k-1][:,0],self.clusters[self.k-1][:,1], s = np.full(len(self.clusters[self.k-1]),3),label = "Cluster " + str(self.k) + " data")
        ax1.scatter(self.best_centroid[self.k-1][0],self.best_centroid[self.k-1][1],marker='v',c=1,label = "Cluster centroids")
        #Add a legend, title, axis labels and show the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('X-value of data')
        plt.ylabel('Y-value of data')
        plt.title("Plot of best k-means model with " + str(self.k) + " clusters on iteration 1")
        plt.show()
        
        #Set up plots
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #Get the centroids from the middle most iteration of the best run of the algorithm, and separate the data into the clusters
        self.best_centroid = self.prevcentroids[bestrun][int(len(self.prevcentroids[bestrun])/2)]
        self.partition_clusters(self.best_centroid)
        #Loop over each cluster (except the last) and plot the centroid and data points in that cluster.
        #This is done so that each cluster's data points have a different color. 
        for i in range(self.k-1):
            ax1.scatter(self.clusters[i][:,0],self.clusters[i][:,1], s = np.full(len(self.clusters[i]),3),label = "Cluster " + str(i+1) + " data")
            ax1.scatter(self.best_centroid[i][0],self.best_centroid[i][1],marker='v',c=1)
        #Plot the centroid and data of the last cluster, the reason this is done separately is simply to make the 
        #plot legend formatted more neatly, since centroids are all plotted as the same shape and color and I want the 
        #legend to only have one row for the centroid points
        ax1.scatter(self.clusters[self.k-1][:,0],self.clusters[self.k-1][:,1],s = np.full(len(self.clusters[self.k-1]),3),label = "Cluster " + str(self.k) + " data")
        ax1.scatter(self.best_centroid[self.k-1][0],self.best_centroid[self.k-1][1],marker='v',c=1,label = "Cluster centroids")
        #Add a legend, title, axis labels and show the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('X-value of data')
        plt.ylabel('Y-value of data')
        plt.title("Plot of best k-means model with " + str(self.k) + " clusters on iteration " + str(int(len(self.prevcentroids[bestrun])/2)))
        plt.show()
        
        
        #Set up plots
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #Get the final centroids from the best run of the algorithm, and separate the data into the clusters
        self.best_centroid = self.centroids[bestrun]
        self.partition_clusters(self.best_centroid)
        #Loop over each cluster (except the last) and plot the centroid and data points in that cluster.
        #This is done so that each cluster's data points hava a different color.
        for i in range(self.k-1):
            ax1.scatter(self.clusters[i][:,0],self.clusters[i][:,1],s = np.full(len(self.clusters[i]),3),label = "Cluster " + str(i+1) + " data")
            ax1.scatter(self.best_centroid[i][0],self.best_centroid[i][1],marker='v',c=1)
        #Plot the centroid and data of the last cluster, the reason this is done separately is simply to make the 
        #plot legend formatted more neatly, since centroids are all plotted as the same shape and color and I want the 
        #legend to only have one row for the centroid points
        ax1.scatter(self.clusters[self.k-1][:,0],self.clusters[self.k-1][:,1],s = np.full(len(self.clusters[self.k-1]),3),label = "Cluster " + str(self.k) + " data")
        ax1.scatter(self.best_centroid[self.k-1][0],self.best_centroid[self.k-1][1],marker='v',c=1,label = "Cluster centroids")
        #Add a legend, title, axis labels and show the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('X-value of data')
        plt.ylabel('Y-value of data')
        plt.title("Plot of best k-means model with " + str(self.k) + " clusters after final iteration")
        plt.show()

    
                
                
#Run the k-means algorithm with 3, 5 and 8 clusters for 10 runs each
    
k = kmeans(data,3,10)
k.run()      

k = kmeans(data,5,10)
k.run()

k = kmeans(data,8,10)
k.run()