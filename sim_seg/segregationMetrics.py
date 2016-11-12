import numpy as np
import pandas as pd
from numpy import float
from scipy.spatial.distance import cdist


class Metrics(object):
    def __init__(self):
        self.attributeMatrix = np.matrix([])    # attributes matrix full size - all columns
        self.location = []                      # x and y coordinates from file (2D lists)
        self.pop = []                           # groups to be analysed [:,4:n] (2D lists)
        self.pop_sum = []                       # sum of population groups from pop (1d array)
        self.locality = []                      # local population intensity for groups
        self.n_location = 0                     # length of list (n lines) (attributeMatrix.shape[0])
        self.n_group = 0                        # number of groups (attributeMatrix.shape[1] - 4)
        self.costMatrix = []                    # scipy cdist distance matrix
        self.cord = []                          # empty, readcoord function, remove it?
    # TODO remove sum from file and update index for pop, pop_sum...

    # def readCordinates(self, filepath):
    #     self.cord = np.asmatrix(pd.read_csv(filepath))
    #     return self.cord

    def readAttributesFile(self, filePath):
        """
        This function reads the csv file and populate the class attributes. Data has to be exactly in the
        following format or results will be wrong:
        area id,  x_coord, y_coord , sum, attribute 1, attributes 2, attributes 3, attribute n...

        :param filePath: path with file to be read
        :return: attribute Matrix [n,n]
        """
        self.attributeMatrix = np.asmatrix(pd.read_csv(filePath))
        n = self.attributeMatrix.shape[1]
        self.location = self.attributeMatrix[:, 1:3]
        self.location = self.location.astype('float')
        self.pop = self.attributeMatrix[:, 4:n].astype('int')
        self.pop[np.where(self.pop < 0)[0], np.where(self.pop < 0)[1]] = 0
        self.n_group = n-4
        self.n_location = self.attributeMatrix.shape[0]
        self.pop_sum = np.sum(self.pop, axis=1)
        return self.attributeMatrix

    # # this part could be optimised
    # def cal_location_subgroup(self, locations, sub_pop, dis=5000, n_pnt=10000):
    #     n_local = len(locations[:, 1])
    #     n_subgroup = len(sub_pop[1, :])
    #     locality_sub = np.empty[n_local, n_subgroup]
    #     for index_sub in range(0, n_subgroup):
    #         locality_sub[:, index_sub] = self.cal_locality_sum(locations, sub_pop[:, index_sub], dis, n_pnt)
    #     return np.asarray(locality_sub)

    def readCostMatrix(self, filePath):
        """
        This function is used in case a cost matrix was already computed outside. It allows
        import of a local file to be represented as a distance matrix.

        :param filePath: path with file to be read
        :return: distance matrix [n,n]
        """
        # TODO has to be tested and validated
        self.costMatrix = np.matrix(pd.read_csv(filePath, header=None))
        # n = self.costMatrix.shape[1]
        # self.costMatrix = self.costMatrix[:,1:n]
        self.costMatrix = self.costMatrix.astype(np.float)
        self.costMatrix[np.isinf(self.costMatrix)] = 0
        self.costMatrix = np.nan_to_num(self.costMatrix)
        self.costMatrix = self.costMatrix
        return self.costMatrix

    def cal_localityMatrix(self, bandwidth=5000, weightmethod=1):  # n_pnt=1000 param not being used
        """
        This function calculate the local population intensity for groups.

        :param bandwidth: bandwidth for neighbors in meters
        :param weightmethod: 1 for gaussian, 2 for bi-square and empty for moving window
        :return: 2d list of population intensity
        """
        n_local = self.location.shape[0]
        n_subgroup = self.pop.shape[1]
        locality_temp = np.empty([n_local, n_subgroup])
        for index in range(0, n_local):
            for index_sub in range(0, n_subgroup):
                cost = cdist(self.location[index, :], self.location)
                weight = self.getWeight(cost, bandwidth, weightmethod)
                #weight = self.getWeight(self.costMatrix[index,:].T,bandwidth,weightmethod)
                locality_temp[index, index_sub] = np.sum(weight * np.asarray(self.pop[:, index_sub]))/np.sum(weight)
        self.locality = locality_temp
        self.locality[np.where(self.locality < 0)[0], np.where(self.locality < 0)[1]] = 0
        return locality_temp

    # def getNeighbor(self, index, locations, dis=5000, n_pnt=10000):
    #     """
    #     Not being used...
    #     :param index:
    #     :param locations:
    #     :param dis:
    #     :param n_pnt:
    #     :return:
    #     """
    #     loc = locations[index, :]
    #     distance = np.asarray(locations[:, 0] - loc[0, 0]) * np.asarray(locations[:, 0] - loc[0, 0]) + \
    #                np.asarray(locations[:, 1] - loc[0, 1]) * np.asarray(locations[:, 1] - loc[0, 1])
    #     distance = np.sqrt(distance)
    #     sel_loc = np.where(distance < dis)
    #     sel_location = np.empty([len(sel_loc[0]), 2])
    #     sel_location[:, 0] = sel_loc[0].T
    #     sel_location[:, 1] = distance[sel_loc[0], 0]
    #     return sel_location
    #
    # locality_sum = is the sum of pop intensity in number of areas - is an array of size j
    # locality_sub is the sub of pop intensity in number of areas - is an matrix of size j*m

    def cal_localDissimilarity(self):
        """
        Compute local dissimilarity
        :return: array with results
        """
        if len(self.locality) == 0:
            self.locality = self.pop  # cal_localityMatrix() using default values
        lj = np.asarray(np.sum(self.locality, axis=1))
        tjm = self.locality * 1.0 / lj[:, None]
        tm = np.sum(self.pop, axis=0) * 1.0 / np.sum(self.pop)
        I = np.sum(np.asarray(tm) * np.asarray(1 - tm))
        sum_pop = np.sum(self.pop, axis=1)
        N = np.sum(self.pop)
        D_local = np.sum(1.0 * np.array(np.fabs(tjm - tm)) * np.asarray(sum_pop).ravel()[:, None] / (2 * N * I), axis=1)
        #np.savetxt("res/d_local.csv",D_local, delimiter=",")
        return D_local

    def cal_globalDissimilarity(self):
        """
        Call local dissimilarity and compute the sum from individual values.
        :return: global number result
        """
        d_local = self.cal_localDissimilarity()
        return np.sum(d_local)

    # # when m=n then it is the isolation
    # # spatial exposure index of group m to group n
    # def cal_globalExposure_backup(self):
    #     m = self.n_group
    #     exposure_rs = np.zeros((m,m))
    #     for i in range(m):
    #         for j in range(m):
    #            localExpo = np.asarray(self.pop[:,i])*1.0/np.sum(self.pop[:,i])
    #            localityRate = np.asarray(self.locality[:,j])*1.0/np.asarray(np.sum(self.locality,axis = 1))
    #            localityRate = np.vstack(localityRate)
    #            expo = np.asarray(localExpo)*np.array(localityRate)
    #            expo[np.isinf(expo)]=0
    #            expo[np.isnan(expo)]=0
    #            exposure_rs[i,j] = np.sum(expo)
    #     return exposure_rs

    def cal_localExposure(self):
        """
        This function computes the local spatial exposure index of group m to group n.
        in situations where m=n, then the result is the isolation index
        :return: 2d list with individual indexes
        """
        if len(self.locality) == 0:
            self.locality = self.cal_localityMatrix()
        m = self.n_group
        j = self.n_location
        exposure_rs = np.zeros((j, (m * m)))
        localExpo = np.asarray(self.pop) * 1.0 / np.asarray(np.sum(self.pop, axis=0)).ravel()
        localityRate = np.asarray(self.locality) * 1.0 / np.asarray(np.sum(self.locality, axis=1)).ravel()[:, None]
        for i in range(m):
            exposure_rs[:, ((i*m)+0):((i*m)+m)] = np.asarray(localityRate)*np.asarray(localExpo[:, i]).ravel()[:, None]
        exposure_rs[np.isinf(exposure_rs)] = 0
        exposure_rs[np.isnan(exposure_rs)] = 0
        return exposure_rs

    def cal_globalExposure(self):
        localexpo = self.cal_localExposure()
        rs = np.sum(localexpo, axis=0)
        return rs

    def getWeight(self, distance, bandwidth, weightmethod=1):
        """
        This function computes the weights for neighborhood. Default value is Gaussian(1)
        :param distance: distance in meters to be considered for weighting
        :param bandwidth: bandwidth in meters selected to perform neighborhood
        :param weightmethod: method to be used: 1-gussian , 2-bi square and empty-moving windows
        :return: weight value
        """
        distance = np.asarray(distance.T)
        #distance = distance.ravel()
        if weightmethod == 1:
            weight = np.exp((-0.5) * (distance/bandwidth) * (distance/bandwidth))
        elif weightmethod == 2:
            weight = (1 - (distance/bandwidth)*(distance/bandwidth)) * (1 - (distance/bandwidth)*(distance/bandwidth))
            sel = np.where(distance > bandwidth)
            weight[sel[0]] = 0
        else:
            weight = 1
            sel = np.where(distance > bandwidth)
            weight[sel[0], :] = 0
       # weight = weight/sum(weight)
        return weight

    def globalDiversity(self):
        """
        This function computes the global entropy index (diversity score)
        :return: diversity score
        """
        group_score = []
        pop_total = np.sum(self.pop_sum)
        prop = np.asarray(np.sum(self.pop, axis=0))
        for group in prop[0]:
            group_idx = group/pop_total * np.log(1 / (group/pop_total))
            group_score.append(group_idx)
        mscore = np.sum(group_score)
        return mscore

    def localDiversity(self):
        """
        This function computes the global entropy index (diversity score)
        :return: array with area indices
        """
        proportion = np.asarray(self.pop) / self.pop_sum
        area_idx = 1
        return proportion

    def entropyIndex(self, global_diversity):
        test = 1

        return test
