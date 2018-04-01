import tensorflow as tf
import numpy as np
from collections import defaultdict


class LinearMRF(object):
    '''Class implementing the MRF used for denoising'''

    def __init__(self, width, height):
        '''Initializes the MRF model.

        Specifically, the model used is a grid graph, i.e. there is a
        node for every pixel in an image and a pair connecting every pair
        of adjacent pixels.

        The nodes are numbered sequentially, starting from the upper right
        corner of the image, first increasing to the right and then downward.
        For example, here are the numberings of the nodes of a 3x4 graph:
        0  1  2  3
        4  5  6  7
        8  9  10 11

        Pairs are represented as tuples (m, n), where m and n are the
        indices of the two nodes. For example, in the above graph,
        (5, 9) and (1, 2) are both pairs.

        After initialization, this class contains the following fields:
            width(int): the width in pixels of the images represented
            height(int): the height in pixels of the images represented
            unary_weight(tf.Variable): the linear weight for unary potentials
            pairwise_weight(tf.Variable): the linear weight for pairwise
                potentials
            pairs(list of tuples): represents the edges in the MRF. Each entry
                is a tuple containing two integers, each representing a node
            pair_inds(dict): provides the index within the list of pairs
                for a given pair
            neighbors(dict): the keys are nodes (integers), and the values
                are a list of integers representing the nodes adjacent to
                the specified node in the MRF
            pairwise_features(np.array): The pairwise features for the model
        '''
        self.width = width
        self.height = height
        self.unary_weight = tf.Variable(1.)
        self.pairwise_weight = tf.Variable(1.)

        self.pairs = []
        self.pair_inds = {}
        self.neighbors = defaultdict(list)
        #i is index
        i = 0

        # Initialize neighbors, pairs, and pair_inds
        for row in range(height):
            for col in range(width):
                ind1 = row*width + col

                if row > 0:
                    ind2 = (row-1)*width + col
                    self.pairs.append((ind2, ind1))
                    self.pair_inds[(ind2, ind1)] = i
                    self.pair_inds[(ind1, ind2)] = i

                    self.neighbors[ind1].append(ind2)
                    self.neighbors[ind2].append(ind1)

                    i += 1
                if col > 0:
                    ind2 = row*width + (col-1)
                    self.pairs.append((ind2, ind1))
                    self.pair_inds[(ind2, ind1)] = i
                    self.pair_inds[(ind1, ind2)] = i

                    self.neighbors[ind1].append(ind2)
                    self.neighbors[ind2].append(ind1)

                    i += 1

        # pre-compute pair features, since they do not depend on the image
        self.pairwise_features = self.get_pairwise_features()

    def get_unary_features(self, img):
        '''Calculates the full matrix of unary features for a provided
        set of values (which can be either the noisy observations or the
        true image)

        For a given node observation x_i and possible assignment
        y_i to that node, the feature function you should calculate is

        f_unary(x_i, y_i) = 1[x_i == y_i]

        where 1[] is the indicator function that equals 1 if the argument is
        true and 0 if the argument is false.

        As mentioned, this calculates the full matrix of unary features - you
        should use the following index scheme:

        result[i, 0] is the feature value for node x_i when y_i = 0
        result[i, 1] is the feature value for node x_i when y_i = 1

        Args:
            img(np.array): An array of length (width x height) representing
              the observations of an image
        Returns:
            (np.array): An array of size (width x height, 2) containing the
              features for an image
        '''
        unary_features = np.zeros((self.width*self.height,2))
        unary_features[:,0] = 1 - img
        unary_features[:,1] = img.copy()

        # for row in range(self.height):
        #     for col in range(self.width):
        #         index = row*self.width + col
        #         #possible assignment y_i
        #         Y_i = [0,1]
        #         for i in range(2):
        #             unary_features[index,i] = int(Y_i[i] == img[index])

        return unary_features

    def get_pairwise_features(self):
        '''Calculates the full matrix of pairwise features.

        For a given set of possible assignments y_i and y_j to
        a pair of nodes (i, j) in the graph, the feature function you should
        calculate is

        f_pairwise(y_i, y_j) = 1[y_i == y_j]

        where 1[] is the indicator function that equals 1 if the argument is
        true and 0 if the argument is false.

        This calculates the full matrix of pairwise features -
        should use the following index scheme, where i is the index for
        pair (m, n) as found in self.pair_inds (and m < n):

        result[i, 0] is the feature value for y_m = 0, y_n = 0
        result[i, 1] is the feature value for y_m = 0, y_n = 1
        result[i, 2] is the feature value for y_m = 1, y_n = 0
        result[i, 3] is the feature value for y_m = 1, y_n = 1

        Returns:
            (np.array): An array of size (len(pairs), 4) containing the
              pairwise features for an image
        '''
        pairwise_features = np.zeros((len(self.pairs),4))
        # Y_mn = [(0,0),(0,1),(1,0),(1,1)]
        #
        # for i in range(len(self.pairs)):
        #     for j in range(4):
        #         pairwise_features[i,j] = int(Y_mn[j][0] == Y_mn[j][1])

        for pair in self.pairs:
            pairwise_features[self.pair_inds[pair],0] = 1
            pairwise_features[self.pair_inds[pair],3] = 1

        return pairwise_features

    def calculate_unary_potentials(self, unary_features):
        '''Calculates the full matrix of unary potentials for a provided
        matrix of unary features.

        For a given node observation x_i and an assignment y_i to
        that node, the potential function you should calculate is

        phi_i(x_i, y_i) = w_unary * f_unary(x_i, y_i)

        where f_unary(x_i, y_i) is the value of the feature function for a
        given node/assignment

        Args:
            unary_features(np.array): a (height * width, 2)-sized matrix
              containing the features for a noisy sample (see
              get_unary_features for more details)
        Returns:
            (tf.Tensor): The unary potentials, which is a rank-2 tensor of the
              same size as the unary_features.
        '''
        unary_potentials = self.unary_weight * unary_features
        # unary_potentials = tf.Variable(unary_potentials,tf.float32)
        # unary_potentials = tf.cast(unary_potentials, tf.float32)


        return unary_potentials

    def calculate_pairwise_potentials(self, pairwise_features):
        '''Calculates the full matrix of pairwise potentials for a provided
        matrix of pairwise features.

        For a given pair of assignments y_i and y_j to nodes i
        and j, the potential function you should calculate is

        phi_ij(y_i, y_j) = w_pairwise * f_pairwise(y_i, y_j)

        where f_pairwise(y_i, y_j) is the value of the pairwise feature
        function for a given node/assignment

        Args:
            pairwise_features(np.array): a (len(pairs), 4)-sized matrix
              containing the pairwise features (see get_pairwise_features
              for more details)
        Returns:
            (tf.Tensor): The pairwise potentials, which is a rank-2 tensor of
              the same size as the pairwise_features
        '''
        pairwise_potentials = self.pairwise_weight * pairwise_features
        # pairwise_potentials = tf.Variable(pairwise_potentials,tf.float32)

        return pairwise_potentials

    def build_training_obj(self, img_features, unary_beliefs, pair_beliefs,
                           unary_potentials, pairwise_potentials):
        '''Builds the training objective, as specified in the handout.

        Hint: the image features can be thought of as a "correct" set of
        beliefs for that image

        Args:
            img_features(np.array): The unary feature representation of the
                true image (as returned by get_unary_features)
            unary_beliefs(list(tf.Tensor)): A list of the unary beliefs
                for each of the noisy samples
            pair_beliefs(tf.Tensor): A list of the pairwise beliefs for each
                of the noisy samples
            unary_potentials(list(tf.Tensor)): A list of the unary potentials
                for each of the noisy samples. Each entry is a rank-2 tensor
                of size (height x width, 2)
            pairwise_potentials(tf.Tensor): The pairwise potentials, which is
                a rank-2 tensor of size (height x width, 4)
        Returns:
            (tf.Tensor): the training objective, which is a rank-0 tensor
        '''
        obj = 0
        #ten noisy graphs
        for i in range(len(unary_beliefs)):
            unary_score = tf.reduce_sum(tf.multiply(unary_beliefs[i],unary_potentials[i]))
            # unary_score = tf.cast(unary_score, tf.float32)

            pairwise_score = tf.reduce_sum(tf.multiply(pairwise_potentials,pair_beliefs[i]))
            # pairwise_score = tf.cast(pairwise_score, tf.float32)

            # self.calculate_unary_potentials(img_features)

            F = tf.reduce_sum(img_features*unary_potentials[i]) + \
                tf.reduce_sum(self.get_pairwise_beliefs(img_features)*pairwise_potentials)
            # F = tf.cast(F, tf.float32)

            obj += unary_score + pairwise_score - F

        return obj

    def train(self, original_img, noisy_samples, lr, num_epochs,
              convergence_margin):
        '''Trains the model using the provided data and training parameters.

        Args:
            original_img(np.array): The true, denoised image
            noisy_samples(list(np.array)): Noisy samples of the true image
            lr(float): The learning rate for gradient descent
            num_epochs(int): The number of training iterations
            convergence_margin(float): The convergence margin for inference
                (see run_greedy_inference for more details)
        '''

        # Initialize placeholders for beliefs
        unary_belief_placeholders = []
        pairwise_belief_placeholders = []

        for i in range(len(noisy_samples)):
            unary_belief_placeholders.append(tf.placeholder(tf.float32,
                                             [self.height*self.width, 2]))
            pairwise_belief_placeholders.append(tf.placeholder(tf.float32,
                                                [len(self.pairs), 4]))



        # Compute features for original image and noisy samples
        img_features = self.get_unary_features(original_img)
        noisy_features = [self.get_unary_features(noisy)
                          for noisy in noisy_samples]

        # Compute initial beliefs. We initialize them to be identical to the
        # noisy features (meaning the beliefs are set such that the model
        # believes the noisy observations to be correct
        unary_beliefs = []
        pairwise_beliefs = []
        unary_beliefs = [feat.copy() for feat in noisy_features]
        pairwise_beliefs = [self.get_pairwise_beliefs(belief)
                            for belief in unary_beliefs]

        # Build the computation graph for training
        unary_potentials = [self.calculate_unary_potentials(feat)
                            for feat in noisy_features]
        pairwise_potentials = self.calculate_pairwise_potentials(
                self.pairwise_features)
        train_obj = self.build_training_obj(img_features,
                                            unary_belief_placeholders,
                                            pairwise_belief_placeholders,
                                            unary_potentials,
                                            pairwise_potentials)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(train_obj)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):
                print("EPOCH %d" % (epoch+1))

                # First: Calculate Potentials
                nodes = [pairwise_potentials] + unary_potentials
                results = sess.run(nodes)
                pairwise_pot_result = results[0]
                unary_pot_result = results[1:]

                # Second: Run Inference
                unary_beliefs, pairwise_beliefs = self.run_greedy_inference(
                        unary_beliefs, unary_pot_result, pairwise_pot_result,
                        convergence_margin)

                score = 0
                for unary_belief in unary_beliefs:
                    score += np.sum(unary_beliefs != img_features) /\
                            (len(unary_beliefs) * 2)
                score /= len(unary_beliefs)
                print("CURRENT SCORE: ", score)
                print("WEIGHTS: ")
                print(sess.run(self.unary_weight))
                print(sess.run(self.pairwise_weight))

                # Third: Update model parameters based on current beliefs
                feed_dict = {}
                for belief, placeholder in zip(unary_beliefs,
                                               unary_belief_placeholders):
                    feed_dict[placeholder] = belief
                for belief, placeholder in zip(pairwise_beliefs,
                                               pairwise_belief_placeholders):
                    feed_dict[placeholder] = belief
                nodes = [train, pairwise_potentials] + unary_potentials
                results = sess.run(nodes, feed_dict)

    def test(self, noisy_samples, convergence_margin):
        '''Given a list of noisy samples of an image, produce denoised
        versions of that image.

        Args:
            noisy_samples(list(np.array)): A list of noisy samples of an image
            convergence_margin(float): The convergence margin for inference
                (see run_greedy_inference for more details)
        Returns:
            (list(np.array)): The denoised images
        '''

        # Initialize Beliefs
        unary_beliefs = []
        noisy_features = [self.get_unary_features(noisy)
                          for noisy in noisy_samples]
        unary_beliefs = [feat.copy() for feat in noisy_features]

        unary_potentials = [self.calculate_unary_potentials(feat)
                            for feat in noisy_features]
        pairwise_potentials = self.calculate_pairwise_potentials(
                self.pairwise_features)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Calculate potentials
            nodes = [pairwise_potentials] + unary_potentials
            results = sess.run(nodes)
            pairwise_pot_result = results[0]
            unary_pot_result = results[1:]

        unary_beliefs, pairwise_beliefs = self.run_greedy_inference(
                unary_beliefs, unary_pot_result, pairwise_pot_result,
                convergence_margin)

        denoised_imgs = [self.beliefs2img(unary_belief)
                         for unary_belief in unary_beliefs]
        return denoised_imgs

    def beliefs2img(self, unary_belief):
        '''Converts a provided set of beliefs into the correct format for
        an image by setting each pixel to the value that has belief 1
        '''
        result = np.empty(len(unary_belief))
        for i in range(len(unary_belief)):
            if unary_belief[i, 0] == 1:
                result[i] = 0
            else:
                result[i] = 1
        return result.reshape((self.height, self.width))

    def run_greedy_inference(self, unary_beliefs, unary_pots, pairwise_pots,
                             convergence_margin):
        '''Runs our greedy inference procedure on the provided lists of
        beliefs/potentials. Note that we run inference for a maximum of 10
        iterations per image.
        '''
        new_unary_beliefs = []
        new_pairwise_beliefs = []

        for unary_belief, unary_pot in zip(unary_beliefs, unary_pots):
            itr = 0
            converged = False

            unary_belief = unary_belief.copy()

            while not converged:
                if itr > 10:
                    break

                itr += 1
                # print("not converged")

                new_unary_belief = self.inference_itr(unary_belief, unary_pot,
                                                      pairwise_pots)
                converged = self.check_convergence(new_unary_belief,
                                                   unary_belief,
                                                   convergence_margin)

                unary_belief = new_unary_belief

            pairwise_belief = self.get_pairwise_beliefs(unary_belief)
            new_unary_beliefs.append(unary_belief)
            new_pairwise_beliefs.append(pairwise_belief)
        return new_unary_beliefs, new_pairwise_beliefs

    def inference_itr(self, unary_beliefs, unary_pots, pairwise_pots):
        # print("inference_itr")
        '''Run a single iteration of inference with the provided beliefs
        and potentials.

        Here, inference should be implemented as a simple (randomized) greedy
        algorithm. The steps are as follows:
        1. Determine a random order of nodes
        2. For each node in the graph:
            a. Calculate the scores (using calculate_local_score) for that
               node having assignment 0 and 1
            b. Set the belief for the assignment having the larger score to 1
               and that for the assignment having the smaller score to 0

        Args:
            unary_beliefs(np.array): The input set of beliefs, having shape
            (width x height, 2)
            unary_pots(np.array): The unary potentials for the image, having
              shape (width x height, 2)
            pairwise_pots(np.array): The pairwise potentials for the image,
              having shape (len(self.pairs), 4)
        Returns:
            (np.array): The new set of unary beliefs, having the same shape
              as the input set of unary beliefs.
        '''
        unary_beliefs = unary_beliefs.copy()
        node = np.arange(len(unary_beliefs))
        np.random.shuffle(node)
        for i in node:
            score1 = self.calculate_local_score(i, 0, unary_beliefs,unary_pots, pairwise_pots)
            score2 = self.calculate_local_score(i, 1, unary_beliefs,unary_pots, pairwise_pots)
            if score1 > score2:
                unary_beliefs[i][0] = 1
                unary_beliefs[i][1] = 0
            else:
                unary_beliefs[i][0] = 0
                unary_beliefs[i][1] = 1

        return unary_beliefs

    def calculate_local_score(self, node, assignment, unary_beliefs,
                              unary_potentials, pairwise_potentials):

        '''Calculates the score of a "patch" surrounding the specified node,
        assuming that node has the specified assignment, given the current
        beliefs for the assignments of values to the pixels in the image

        This score consists of the sum of the unary potential for this node
        given this assignment, plus the pairwise potentials of all pairs
        that include this node given the assignment specified for this node
        and the assignment for the other nodes specified by the provided
        unary beliefs.

        Args:
            node(int): The node whose patch is being scored
            assignment(int): The assignment that should be considered for
              the node (either 0 or 1)
            unary_beliefs(np.array): The current set of unary beliefs for
              the image, having shape (width x height, 2)
            unary_potentials(np.array): The current set of unary potentials
              for the image, having shape (width x height, 2)
            pairwise_potentials(np.array): The current set of pairwise
              potentials for the image, having shape (len(self.pairs), 4)
        Returns:
            (float): The score of the patch
        '''
        score = 0.0
        pairs_list = self.neighbors[node]
        pair_potential = 0
        pair_score = 0

        for pair_node in pairs_list:
            if unary_beliefs[pair_node][0]==1:
                node_truth_value = 0
            elif unary_beliefs[pair_node][1]==1:
                node_truth_value = 1

            pair_index = self.pair_inds[(node,pair_node)]
            pair_pot = pairwise_potentials[pair_index]
            if assignment == 0 and node_truth_value == 0:
                pair_potential = pair_pot[0]
            elif assignment == 0 and node_truth_value == 1:
                pair_potential = pair_pot[1]
            elif assignment == 1 and node_truth_value == 0:
                pair_potential = pair_pot[2]
            elif assignment == 1 and node_truth_value == 1:
                pair_potential = pair_pot[3]

            pair_score += pair_potential

        score = np.sum(unary_potentials[node][assignment] + pair_score)

        return score

    def check_convergence(self, new_unary_beliefs, old_unary_beliefs,
                          convergence_margin):
        # print("check_convergence")
        '''Given two sets of unary beliefs, determine if inference has
        converged.

        Convergence occurs when the percentage of nodes in the graph whose
        beliefs have changed is less than the provided margin.

        Args:
            new_unary_beliefs(np.array): One set of unary beliefs, having
              the same shape as elsewhere in the code
            old_unary_beliefs(np.array): Another set of unary beliefs, having
              the same shape as elsewhere in the code
            convergence_margin(float): the threshold determining convergence.
              This should be a number between 0 and 1
        Returns:
            (bool): whether inference has converged
        '''
        convergence_rate = np.count_nonzero(np.subtract(new_unary_beliefs,old_unary_beliefs)) / (2.0*len(new_unary_beliefs))

        return (convergence_margin > convergence_rate)

    def get_pairwise_beliefs(self, unary_beliefs):
        '''Generates the appropriate pairwise beliefs for a specified set of
        unary beliefs.

        Due to the fact that all of the unary beliefs for this inference
        implementation are either 0 or 1, the pairwise beliefs are a
        simple deterministic function of the unary beliefs.

        Specifically, given a pair of nodes (m, n), the pairwise belief
        for assignment (y_m, y_n) = 1 iff the unary belief for node m with
        assignment y_m is 1 and the unary belief for node n with assignment
        y_n is 1.

        Args:
            unary_beliefs(np.array): The set of unary beliefs for a noisy
              sample. This array has shape (width x height, 2)
        Returns:
            (np.array): The set of pairwise beliefs. This array should have
              shape (len(self.pairs), 4) and is indexed the same way as
              specified in get_pairwise_features.
        '''
        # print("get_pairwise_beliefs")
        result = np.zeros((len(self.pairs), 4))
        for node,index in self.pair_inds.items():
            node1 = node[0]
            node2 = node[1]
            i = index
            y_node1 = unary_beliefs[node1][1]
            y_node2 = unary_beliefs[node2][1]

            if y_node1 ==0 and y_node2==0:
                result[i][0] = 1
                result[i][1] = 0
                result[i][2] = 0
                result[i][3] = 0

            if y_node1 ==0 and y_node2==1:
                result[i][0] = 0
                result[i][1] = 1
                result[i][2] = 0
                result[i][3] = 0

            if y_node1 ==1 and y_node2==0:
                result[i][0] = 0
                result[i][1] = 0
                result[i][2] = 1
                result[i][3] = 0

            if y_node1 ==1 and y_node2==1:
                result[i][0] = 0
                result[i][1] = 0
                result[i][2] = 0
                result[i][3] = 1

        return result
