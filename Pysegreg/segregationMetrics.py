import numpy as np
from scipy.spatial.distance import cdist


class Segreg(object):
    def __init__(self):
        self.attributeMatrix = np.matrix([])    # attributes matrix full size - all columns
        self.location = []                      # x and y coordinates from tract centroid (2D lists)
        self.pop = []                           # population of each groups by tract (2D lists)
        self.pop_sum = []                       # total population of the tract (sum all groups)
        self.locality = []                      # population intensity by groups by tract
        self.n_location = 0                     # length of list (n lines) (attributeMatrix.shape[0])
        self.n_group = 0                        # number of groups (attributeMatrix.shape[1] - 4)
        self.costMatrix = []                    # scipy cdist distance matrix
        self.tract_id = []                      # tract ids in string format

    def readAttributesFile(self, filepath):
        """
        This function reads the csv file and populate the class's attributes. Data has to be exactly in the
        following format or results will be wrong:
        area id,  x_coord, y_coord, attribute 1, attributes 2, attributes 3, attribute n...
        :param filepath: path with file to be read
        :return: attribute Matrix [n,n]
        """
        raw_data = np.genfromtxt(filepath, skip_header=1, delimiter=",", filling_values=0, dtype=None)
        data = [list(item)[1:] for item in raw_data]

        self.attributeMatrix = np.asmatrix(data)
        n = self.attributeMatrix.shape[1]
        self.location = self.attributeMatrix[:, 0:2]
        self.location = self.location.astype('float')
        self.pop = self.attributeMatrix[:, 2:n].astype('int')
        # self.pop[np.where(self.pop < 0)[0], np.where(self.pop < 0)[1]] = 0
        self.n_group = n-2
        self.n_location = self.attributeMatrix.shape[0]
        self.pop_sum = np.sum(self.pop, axis=1)
        self.tract_id = np.asarray([x[0] for x in raw_data]).astype(str)
        self.tract_id = self.tract_id.reshape((self.n_location, 1))

        return self.attributeMatrix

    def getWeight(self, distance, bandwidth, weightmethod=1):
        """
        This function computes the weights for neighborhood. Default value is Gaussian(1)
        :param distance: distance in meters to be considered for weighting
        :param bandwidth: bandwidth in meters selected to perform neighborhood
        :param weightmethod: method to be used: 1-gussian , 2-bi square and empty-moving windows
        :return: weight array for internal use
        """
        distance = np.asarray(distance.T)

        if weightmethod == 1:
            weight = np.exp((-0.5) * (distance/bandwidth) * (distance/bandwidth))

        elif weightmethod == 2:
            weight = (1 - (distance/bandwidth)*(distance/bandwidth)) * (1 - (distance/bandwidth)*(distance/bandwidth))
            sel = np.where(distance > bandwidth)
            weight[sel[0]] = 0

        elif weightmethod == 3:
            weight = (1 + (distance * 0))
            sel = np.where(distance > bandwidth)
            weight[sel[0]] = 0

        else:
            raise Exception('Invalid weight method selected!')

        return weight

    def cal_timeMatrix(self, bandwidth, weightmethod, matrix):
        """
        This function calculate the local population intensity for all groups based on a time matrix.
        :param bandwidth: bandwidth for neighborhood in meters
        :param weightmethod: 1 for gaussian, 2 for bi-square and empty for moving window
        :param matrix: path/file for input time matrix
        :return: 2d array like with population intensity for all groups
        """
        n_local = self.location.shape[0]
        n_subgroup = self.pop.shape[1]
        locality_temp = np.empty([n_local, n_subgroup])

        for index in range(0, n_local):
            for index_sub in range(0, n_subgroup):
                cost = matrix[index, :].reshape(1, n_local)
                weight = self.getWeight(cost, bandwidth, weightmethod)
                locality_temp[index, index_sub] = np.sum(weight * np.asarray(self.pop[:, index_sub])) / np.sum(weight)

        self.locality = locality_temp
        self.locality[np.where(self.locality < 0)[0], np.where(self.locality < 0)[1]] = 0

        return locality_temp

    def cal_localityMatrix(self, bandwidth=5000, weightmethod=1):
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
                locality_temp[index, index_sub] = np.sum(weight * np.asarray(self.pop[:, index_sub]))/np.sum(weight)

        self.locality = locality_temp
        self.locality[np.where(self.locality < 0)[0], np.where(self.locality < 0)[1]] = 0

        return locality_temp

    def cal_localDissimilarity(self):
        """
        Compute local dissimilarity for all groups.
        :return: 1d array like with results for all groups, size of localities
        """
        if len(self.locality) == 0:
            lj = np.ravel(self.pop_sum)
            tjm = np.asarray(self.pop) * 1.0 / lj[:, None]
            tm = np.sum(self.pop, axis=0) * 1.0 / np.sum(self.pop)
            index_i = np.sum(np.asarray(tm) * np.asarray(1 - tm))
            pop_total = np.sum(self.pop)
            local_diss = np.sum(1.0 * np.array(np.fabs(tjm - tm)) *
                                np.asarray(self.pop_sum).ravel()[:, None] / (2 * pop_total * index_i), axis=1)

        else:
            lj = np.asarray(np.sum(self.locality, axis=1))
            tjm = self.locality * 1.0 / lj[:, None]
            tm = np.sum(self.pop, axis=0) * 1.0 / np.sum(self.pop)
            index_i = np.sum(np.asarray(tm) * np.asarray(1 - tm))
            pop_total = np.sum(self.pop)
            local_diss = np.sum(1.0 * np.array(np.fabs(tjm - tm)) *
                                np.asarray(self.pop_sum).ravel()[:, None] / (2 * pop_total * index_i), axis=1)

        local_diss = np.nan_to_num(local_diss)

        return local_diss

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
        This function computes the local exposure index of group m to group n.
        in situations where m=n, then the result is the isolation index.
        :return: 2d list with individual indexes
        """
        m = self.n_group
        j = self.n_location
        exposure_rs = np.zeros((j, (m * m)))

        if len(self.locality) == 0:
            local_expo = np.asarray(self.pop) * 1.0 / np.asarray(np.sum(self.pop, axis=0)).ravel()
            locality_rate = np.asarray(self.pop) * 1.0 / np.asarray(np.sum(self.pop, axis=1)).ravel()[:, None]
            for i in range(m):
                exposure_rs[:, ((i * m) + 0):((i * m) + m)] = np.asarray(locality_rate) * \
                                                              np.asarray(local_expo[:, i]).ravel()[:, None]

        else:
            local_expo = np.asarray(self.pop) * 1.0 / np.asarray(np.sum(self.pop, axis=0)).ravel()
            locality_rate = np.asarray(self.locality) * 1.0 / np.asarray(np.sum(self.locality, axis=1)).ravel()[:, None]
            for i in range(m):
                exposure_rs[:, ((i * m) + 0):((i * m) + m)] = np.asarray(locality_rate) * \
                                                              np.asarray(local_expo[:, i]).ravel()[:, None]

        exposure_rs[np.isinf(exposure_rs)] = 0
        exposure_rs[np.isnan(exposure_rs)] = 0

        return exposure_rs

    def cal_globalExposure(self):
        """
        This function call local exposure function and sum the results for the global index.
        :return: displays global number result
        """
        m = self.n_group
        local_exp = self.cal_localExposure()
        global_exp = np.sum(local_exp, axis=0)
        global_exp = global_exp.reshape((m, m))

        return global_exp

    def cal_localEntropy(self):
        """
        This function computes the local entropy score for a unit area Ei (diversity). A unit within the
        metropolitan area, such as a census tract. If population intensity was previously computed,
        the spatial version will be returned, else the non spatial version will be selected (raw data).
        :return: 2d array with local indices
        """
        if len(self.locality) == 0:
            proportion = np.asarray(self.pop / self.pop_sum)

        else:
            polygon_sum = np.sum(self.locality, axis=1).reshape(self.n_location, 1)
            proportion = np.asarray(self.locality / polygon_sum)

        entropy = proportion * np.log(1 / proportion)

        entropy[np.isnan(entropy)] = 0
        entropy[np.isinf(entropy)] = 0
        entropy = np.sum(entropy, axis=1)
        entropy = entropy.reshape((self.n_location, 1))

        return entropy

    def cal_globalEntropy(self):
        """
        This function computes the global entropy score E (diversity). A metropolitan area's entropy score.
        :return: diversity score
        """
        group_score = []
        pop_total = np.sum(self.pop_sum)
        prop = np.asarray(np.sum(self.pop, axis=0))[0]

        # loop at sum of each population groups
        for group in prop:
            group_idx = group / pop_total * np.log(1 / (group / pop_total))
            group_score.append(group_idx)
        entropy = np.sum(group_score)

        return entropy

    def cal_localIndexH(self):
        """
        This function computes the local entropy index H for all localities. The functions cal_localEntropy() for
        local diversity and cal_globalEntropy for global entropy are called as input. If population intensity
        was previously computed, the spatial version will be returned, else the non spatial version will be
        selected (raw data).
        :return: array like with scores for n groups (size groups)
        """
        local_entropy = self.cal_localEntropy()
        global_entropy = self.cal_globalEntropy()

        et = global_entropy * np.sum(self.pop_sum)
        eei = np.asarray(global_entropy - local_entropy)
        h_local = np.asarray(self.pop_sum) * eei / et

        return h_local

    def cal_globalIndexH(self):
        """
        Function to compute global index H returning the sum of local values. The function cal_localIndexH is
        called as input for sum of individual values.
        :return: values with global index for each group.
        """
        h_local = self.cal_localIndexH()
        h_global = np.sum(h_local)

        return h_global
