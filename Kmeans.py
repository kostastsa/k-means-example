import numpy as np
import matplotlib.pyplot as plt


m=2000
class datapoint:

    def __init__(self,mu,cov,ind):
        self.index = ind
        self.pos = np.random.multivariate_normal(mu,cov,1)


    def __str__(self):
        return 'index={},pos={}'.format(self.index,self.pos)

def init_centroids(m,nclust):
    choice=np.random.choice(m,nclust,replace=False)
    return choice

def assign_cluster(data_points, cluster_centroids):
    """assigns cluster to each data point

    Arguments:
            clust_assignments {list} -- [list of the indices of the cluster to which the point with that label corresponds]
            data_points {list} -- [data points]
            cluster_centroids {np array} -- [positions of cluster centroids]
    """
    k = len(cluster_centroids)
    clust_assignments=[]
    for i in range(m):
        dist={cen:np.linalg.norm(data_points[i]-cluster_centroids[cen]) for cen in range(k)}
        clust_assignments.append(min(dist, key=dist.get))
    return clust_assignments

def segregate_clusters(clust_assignments,data_points,m):
        """It splits the data_point list into k different lists point_in_cluster[1..k]
        each one containing only the points in the corresponding cluster

        Arguments:
                clust_assignments {[type]} -- [description]
                data_points {[type]} -- [description]
        """
        k=2
        points_in_cluster=[]
        for j in range(k):
                points_in_cluster.append([data_points[i].index for i in range(m) if clust_assignments[i]==j])
        return points_in_cluster


def move_centroid(points_in_cluster,k):
        new_centroids = []
        for c in range(k):
            clust0 = points_in_cluster[c]
            s = len(clust0)
            new_centroids.append(sum(clust0)/s)
        return new_centroids


def similarities(real_ind, clust_ind):
    clust1=clust_ind[1]
    simils=[]
    for data_clust in real_ind:
        count_sim=0
        for i in data_clust:
            if i in clust1: count_sim+=1
        simils.append(count_sim)
    return simils


def misclassification_error(d,sigma):
    nclust = 2
    npclust = 1000
    m=nclust*npclust
    mu1 = np.array([-d/2,0])
    mu2 = np.array([+d/2,0])
    cov=[[sigma,0],[0,sigma]]
    data1 = [datapoint(mu1,cov,i) for i in range(npclust)]
    data2 = [datapoint(mu2,cov,npclust+i) for i in range(npclust)]
    data = data1 + data2
    data1_pos = [p.pos for p in data1]
    data2_pos = [p.pos for p in data2]
    data_pos = data1_pos+data2_pos
    k = nclust
    i_label = [int(init_centroids(m,nclust)[0]),int(init_centroids(m,nclust)[1])]
    centroids = np.array([data_pos[i_label[0]],data_pos[i_label[1]]])
    dt=1
    while dt>0.001:
        old_centroids = centroids
        assignments = assign_cluster(data_pos, centroids)
        sep_lists_ind = segregate_clusters(assignments, data, m)
        sep_lists=[]
        for clust in sep_lists_ind:
            sep_lists.append([data[i].pos for i in clust])
        centroids = move_centroid(sep_lists,k)
        dt=np.linalg.norm(np.array(centroids)-np.array(old_centroids))


    c1_ind = [p.index for p in data1]
    c2_ind = [p.index for p in data2]
    data_ind=[c1_ind,c2_ind]
    sim = similarities(data_ind, sep_lists_ind)
    error = 1 - max(sim)/npclust
    return error


sigma=0.05
errors=[]
sigmas=[]
for t in range(100):
   e = misclassification_error(0.5,sigma*t)
   errors.append(e)
   sigmas.append(sigma*t)
plt.plot(sigmas,errors)
plt.show()

#print(misclassification_error(0.1))



# f, (ax1, ax2) = plt.subplots(1, 2)
# data=np.array([data1_pos,data2_pos])
# for i in range(k):
#     datap1 = np.transpose(data[i])
#     ax1.scatter(datap1[0],datap1[1],s=1)
#     datum = np.array(sep_lists[i]).transpose()
#     ax2.scatter(datum[0],datum[1],s=1)
# plt.show()



