import numpy as np
import matplotlib.pyplot as plt


#mean_table =  np.transpose([np.random.uniform(0,1,nclust),np.random.uniform(0,1,nclust)])
#cov_table = [[[0.005,a],[a,0.005]] for a in [0*np.random.uniform(0,1) for i in range(nclust)]]
#data = [np.random.multivariate_normal(mean_table[i], cov_table[i], npclust) for i in range(nclust)]

class datapoint:
    __ind=0
    def __init__(self,mu,cov):
        self.index = datapoint.__ind
        self.pos = np.random.multivariate_normal(mu,cov,1)
        datapoint.__ind+=1

    def __str__(self):
        print(self.index,self.pos)






# data_join=[]
# for i in range(nclust):
#     for j in range(len(data[i])):
#         data_join.append(list(data[i][j]))

# data_join = np.array(data_join).transpose()

## K - means starts here
# Initialization of cluster centroids

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
        points_in_cluster=[]
        for j in range(k):
                points_in_cluster.append([data_points[i].index for i in range(m) if clust_assignments[i]==j])
        return points_in_cluster


def move_centroid(points_in_cluster,k):
        new_centroids = []
        for c in range(k):
            clust = points_in_cluster[c]
            s = len(clust)
            new_centroids.append(sum(clust)/s)
        return new_centroids


def misclasification(init_data, sep_lists):
    return


nclust = 2
npclust = 10000
d=0.51

mu1 = np.array([-d/2,0])
mu2 = np.array([+d/2,0])
cov=[[0.05,0],[0,0.05]]
data1 = [datapoint(mu1,cov) for i in range(npclust)]
data2 = [datapoint(mu2,cov) for i in range(npclust)]
data=data1+data2
data1_pos = [p.pos for p in data1]
data2_pos = [p.pos for p in data2]
data_pos = data1_pos+data2_pos
m = len(data_pos)


k=nclust
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
    plt.show()
    plt.close()
    centroids = move_centroid(sep_lists,k)
    dt=np.linalg.norm(np.array(centroids)-np.array(old_centroids))


c1_ind = [p.index for p in data1]
c2_ind = [p.index for p in data2]
count=0
for p in sep_lists_ind[1]:
    if p in c1_ind:
        count+=1
print('count=',count)
print(len(sep_lists_ind[1]))

f, (ax1, ax2) = plt.subplots(1, 2)
data=np.array([data1_pos,data2_pos])
for i in range(k):
    datap1 = np.transpose(data[i])
    ax1.scatter(datap1[0],datap1[1],s=1)
    datum = np.array(sep_lists[i]).transpose()
    ax2.scatter(datum[0],datum[1],s=1)
plt.show()





