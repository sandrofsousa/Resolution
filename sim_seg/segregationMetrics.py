import numpy as np
import pandas as pd
from numpy import float
from scipy.spatial.distance import cdist
np.set_printoptions(threshold=np.inf)


class Segreg(object):
    def __init__(self):
        self.attributeMatrix = np.matrix([])    # attributes matrix full size - all columns
        self.location = []                      # x and y coordinates from file (2D lists)
        self.pop = []                           # groups to be analysed [:,4:n] (2D lists)
        self.pop_sum = []                       # sum of population groups from pop (1d array)
        self.locality = []                      # local population intensity for groups
        self.n_location = 0                     # length of list (n lines) (attributeMatrix.shape[0])
        self.n_group = 0                        # number of groups (attributeMatrix.shape[1] - 4)
        self.costMatrix = []                    # scipy cdist distance matrix

    def readAttributesFile(self, filePath):
        """
        This function reads the csv file and populate the class's attributes. Data has to be exactly in the
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

    def readCostMatrix(self, filePath):
        """
        This function is used in case a cost matrix was already computed. It allows
        the import of a local file to be represented as a distance matrix.
        :param filePath: path with file to be read
        :return: distance matrix with shape [n,n]
        """
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
        This function calculate the local population intensity for all groups.
        :param bandwidth: bandwidth for neighborhood in meters
        :param weightmethod: 1 for gaussian, 2 for bi-square and empty for moving window
        :return: 2d array like with population intensity for all groups
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

    def cal_localDissimilarity(self):
        """
        Compute local dissimilarity
        :return: 1d array like with results for all groups, size of localities
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
        This function call local dissimilarity and compute the sum from individual values.
        :return: display global value
        """
        local_diss = self.cal_localDissimilarity()
        global_diss = np.sum(local_diss)
        return global_diss

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
        """
        This function call local exposure function and sum the results for the global index.
        :return: displays global number result
        """
        m = self.n_group
        local_expo = self.cal_localExposure()
        global_exp = np.sum(local_expo, axis=0)
        global_exp = global_exp.reshape((m, m))
        return global_exp

    def getWeight(self, distance, bandwidth, weightmethod=1):
        """
        This function computes the weights for neighborhood. Default value is Gaussian(1)
        :param distance: distance in meters to be considered for weighting
        :param bandwidth: bandwidth in meters selected to perform neighborhood
        :param weightmethod: method to be used: 1-gussian , 2-bi square and empty-moving windows
        :return: weight value for internal use
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

    def cal_localEntropy(self, intensity=False):
        """
        This function computes the local entropy score for a unit area Ei (diversity). A unit within the
        metropolitan area, such as a census tract.
        :param intensity: if True it uses population intensity, otherwise uses raw data (non spatial).
        :return: 2d array with local indices
        """
        proportion = []
        if intensity is False:
            proportion = np.asarray(self.pop / self.pop_sum)
        else:
            proportion = np.asarray(self.locality / np.sum(self.locality))
        entropy = proportion * np.log(1 / proportion)
        entropy = np.nan_to_num(entropy)
        return entropy

    def cal_globalEntropy(self, intensity=False):
        """
        This function computes the global entropy score E (diversity). A metropolitan area’s entropy score.
        :param intensity: if True it uses population intensity, otherwise uses raw data (non spatial).
        :return: diversity score
        """
        group_score = []
        pop_total = []
        prop = []
        if intensity is False:
            pop_total = np.sum(self.pop_sum)
            prop = np.asarray(np.sum(self.pop, axis=0))[0]
        else:
            pop_total = np.sum(self.locality)
            prop = np.asarray(np.sum(self.locality, axis=0))
        for group in prop:
            group_idx = group / pop_total * np.log(1 / (group / pop_total))
            group_score.append(group_idx)
        entropy = np.sum(group_score)
        return entropy

    def cal_localIndexH(self):
        """
        This function computes the local entropy index H for all localities.
        The local_entropy (array like) local diversity and the
        global_entropy (value) diversity score are called as input.
        :return: array like with scores for n groups (size groups)
        """
        local_entropy = self.cal_localEntropy()
        global_entropy = self.cal_globalEntropy()
        et = np.asarray(global_entropy * np.sum(self.pop_sum))
        eei = np.asarray(global_entropy - local_entropy)
        h_local = eei * np.asarray(self.pop_sum) / et
        return h_local

    def cal_globalIndexH(self):
        """
        Function to compute global index H returning the sum of local values.
        cal_localIndexH is called as input for sum.
        :return: value with global
        """
        h_local = self.cal_localIndexH()
        h_global = np.sum(h_local, axis=0)
        return h_global
