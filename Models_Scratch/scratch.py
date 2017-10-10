import pandas as pd
import numpy as np
import operator

def accuracy(pred,true):
    correct = 0
    pred_len = len(pred)
    for i in range(pred_len):
        if pred[i] == true[i]:
            correct += 1
    return correct/pred_len

class decision_tree(object):
    def __init__(self, max_depth, min_num_split):
        self.max_depth = max_depth
        self.min_num_sample = min_num_split

    def gini_score(self, groups, classes):
        n_samples = sum([len(group) for group in groups])
        gini = 0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            #print(size)
            for class_val in classes:
                #print(group.shape)
                p = (group[:,-1] == class_val).sum() / size
                #print(p)
                score += p * p
            gini += (1.0 - score) * (size / n_samples)
            #print(gini)
        return gini

    def split(self, feat, val, Xy):
#         Xi_left = np.array([]).reshape(0,3)
#         Xi_right = np.array([]).reshape(0,3)
        Xi_left = np.array([]).reshape(0,self.Xy.shape[1])
        Xi_right = np.array([]).reshape(0,self.Xy.shape[1])
        for i in Xy:
            #print(i.shape)
            if i[feat] <= val:
                Xi_left = np.vstack((Xi_left,i))
            if i[feat] > val:
                Xi_right = np.vstack((Xi_right,i))
        return Xi_left, Xi_right

    def best_split(self, Xy):
        classes = np.unique(Xy[:,-1])
        best_feat = 999
        best_val = 999
        best_score = 999
        best_groups = None
        for feat in range(Xy.shape[1]-1):
            for i in Xy:
                groups = self.split(feat, i[feat], Xy)
                #print(groups)
                gini = self.gini_score(groups, classes)
                #print('feat {}, valued < {}, scored {}'.format(feat,i[feat], gini))
                if gini < best_score:
                    best_feat = feat
                    best_val = i[feat]
                    best_score = gini
                    best_groups = groups
        output = {}
        output['feat'] = best_feat
        output['val'] = best_val
        output['groups'] = best_groups
        return output

    def terminal_node(self, group):
        # errored out: couldn't np.unique(nothing) or something - doesn't happen all the time
        #print(group[:,-1])
        classes, counts = np.unique(group[:,-1],return_counts=True)
        return classes[np.argmax(counts)]

    def split_branch(self, node, depth):
        left_node, right_node = node['groups']
        del(node['groups'])
        if not isinstance(left_node,np.ndarray) or not isinstance(right_node,np.ndarray):
            node['left'] = node['right'] = self.terminal_node(left_node + right_node)
            return
        if depth >= self.max_depth:
            node['left'] = self.terminal_node(left_node)
            node['right'] = self.terminal_node(right_node)
            return
        if len(left_node) <= self.min_num_sample:
            node['left'] = self.terminal_node(left_node)
        else:
            node['left'] = self.best_split(left_node)
            self.split_branch(node['left'], depth+1)
        if len(right_node) <= self.min_num_sample:
            node['right'] = self.terminal_node(right_node)
        else:
            node['right'] = self.best_split(right_node)
            self.split_branch(node['right'], depth+1)

    def build_tree(self):
        '''Recursively build tree, unclear if this is the correct way

        '''

        self.root = self.best_split(self.Xy)
        #print(self.root)
        self.split_branch(self.root, 1) # i don't understand how this is working, pointed to node?
        #print(self.root)
        return self.root

    def fit(self,X,y):
        '''Save training data.

        '''
        self.X = X
        self.y = y
        self.Xy = np.column_stack((X, y))
        self.build_tree()

    def display_tree(self, depth=0):
        if isinstance(self.root,dict):
            print('{}[feat{} < {:.2f}]'.format(depth*'\t',(self.root['feat']+1), self.root['val']))
            display_tree(self.root['left'], depth+1)
            display_tree(self.root['right'], depth+1)
        else:
            print('{}[{}]'.format(depth*'\t', self.root))

    def predict_sample(self, node, sample):
        #print(node)
        if sample[node['feat']] < node['val']:
            if isinstance(node['left'],dict):
                return self.predict_sample(node['left'],sample)
            else:
                return node['left']
        else:
            if isinstance(node['right'],dict):
                return self.predict_sample(node['right'],sample)
            else:
                return node['right']

    def predict(self, X_test):
        self.y_pred = np.array([])
        for i in X_test:
            #print(i)
            self.y_pred = np.append(self.y_pred,self.predict_sample(self.root,i))
        return self.y_pred

class random_forest(decision_tree):
    def __init__(self, num_trees, max_depth=2, min_num_split=30, sample_ratio=1):
        self.max_depth = max_depth
        self.min_num_sample = min_num_split
        self.num_trees = num_trees
        self.ratio = sample_ratio

    def build_tree(self, Xy):
        '''Recursively build tree, unclear if this is the correct way

        '''
        self.root = self.best_split(Xy)
        #print(self.root)
        self.split_branch(self.root, 1) # i don't understand how this is working, pointed to node?
        #print(self.root)
        return self.root

    def best_split(self, Xy):
        classes = np.unique(Xy[:,-1])
        best_feat = 999
        best_val = 999
        best_score = 999
        best_groups = None
        n_feats = np.random.choice(Xy.shape[1]-1, self.n_feat, replace=False)
        #print(n_feats)
        for feat in n_feats:
            for i in Xy:
                groups = self.split(feat, i[feat], Xy)
                #print(groups)
                gini = self.gini_score(groups, classes)
                #print('feat {}, valued < {}, scored {}'.format(feat,i[feat], gini))
                if gini < best_score:
                    best_feat = feat
                    best_val = i[feat]
                    best_score = gini
                    best_groups = groups
        output = {}
        output['feat'] = best_feat
        output['val'] = best_val
        output['groups'] = best_groups
        return output

    def samp(self, Xy, ratio=1):
        n = int(np.round(len(Xy) * ratio))
        idx = np.random.randint(Xy.shape[0],size=n)
        return Xy[idx,:]

    def fit(self, X, y):
        '''Save training data.

        '''
        self.X = X
        self.y = y
        self.Xy = np.column_stack((X, y))

        self.n_feat = int(np.sqrt(X.shape[1]))

        self.trees = [self.build_tree(self.samp(self.Xy)) for i in range(self.num_trees)]

    def predict(self, X_test):
        self.y_preds = np.array([]).reshape(0,X_test.shape[0])
        for root in self.trees:
            y_pred = np.array([])
            for i in X_test:
                y_pred = np.append(y_pred,self.predict_sample(root,i))
            #print(y_pred.shape)
            self.y_preds = np.vstack((self.y_preds,y_pred))
        self.avg_preds = np.rint(self.y_preds.mean(axis=0))
        return self.avg_preds

class k_nearest_neighbors(object):
    def __init__(self,k=5,metric='euclidean',ties=True):
        '''K nearest neighbors model. Detemine an unknown sample with a
        majority vote of most similar known samples.

        k = number of neighbors to use
        metric = similarity measure
            'euclidean' is defined by the square root of the sum of squared differences between two arrays of numbers.
            'manhattan' is defined by the sum of the absolute distance between two arrays of numbers.
        ties = in the case of a majority tie, winner goes to the most frequently occuring class

        '''
        self.k = k
        self.metric = metric
        self.ties = ties

    def euclidean(self, a, b):
        dist = 0
        for i in range(len(a)):
            dist += np.square(a[i]-b[i])
        return np.sqrt(dist)

    def manhattan(self, a, b):
        dist = 0
        for i in range(len(a)):
            dist += np.abs(a[i]-b[i])
        return dist

    def find_neighbors(self, new_sample):
        '''List the k neighbors closest to the new sample.

        '''
        distances = []
        for i in range(len(self.X)):
            if self.metric == 'euclidean':
                distance = self.euclidean(self.X[i], new_sample)
            if self.metric == 'manhattan':
                distance = self.manhattan(self.X[i], new_sample)
            distances.append((self.y[i],distance))
        distances = sorted(distances,key=operator.itemgetter(1))

        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][0])
        return neighbors

    def majority_vote(self, neighbors):
        '''Determine majority class from the set of neighbors.

        '''
        class_votes = {}
        for i in range(len(neighbors)):
            sample_class = neighbors[i]
            if sample_class in class_votes:
                class_votes[sample_class] += 1
            else:
                class_votes[sample_class] = 1
        sorted_votes = sorted(class_votes.items())
        if self.ties:
            sorted_votes = self.tie(sorted_votes)
        return sorted_votes[0][0]

#          addition to inspect how often there are ties in counts
    def tie(self,sorted_votes):
        '''Determine when ties occur in the the neighbors. Of the tied classes,
        choose the class most frequent in the training data.

        Print out number of ties.
        '''
        tie = {}
        for pair in sorted_votes:
            count = pair[1]
            if count in tie:
                self.tie_count += 1
                #print('tie')
                tie[count].append(pair[0])
            else:
                tie[count] = [pair[0]]
            #print(tie)
        tie_class_frequency = {}
        if len(tie[count]) > 1:
            #print('tie')
            for tie_class in tie[count]:
                tie_class_frequency[tie_class] = np.count_nonzero(self.y == tie_class)
            max_class = max(tie_class_frequency, key=tie_class_frequency.get)
            #print(max_class)
            sorted_votes = [(max_class,1)]
        return sorted_votes

    def fit(self,X,y):
        '''Save training data.

        '''
        self.X = X
        self.y = y
        self.Xy = np.column_stack((X, y))

    def predict(self, X_test):
        '''Predict class for each value in array of new samples.

        '''
        self.tie_count = 0
        y_pred = []
        for i in range(len(X_test)):
            neighbors = self.find_neighbors(X_test[i])
            pred_class = self.majority_vote(neighbors)
            y_pred.append(pred_class)
        if self.ties:
            print('{} ties'.format(self.tie_count))
        return y_pred

class kmeans():
    def __init__(self, num_centroids=4, max_iter=1000):
        self.num_centroids = num_centroids
        self.max_iter = max_iter
        self.metric = metric
        
    def euclidean(self, a, b):
        return np.sqrt(((a-b)**2).sum(axis=0))

    def manhattan(self, a, b):
        return np.abs((a-b).sum(axis=0))

    def init_centroids(self):
        centroids = {}
        for k in range(self.num_centroids):
            centroids[k] = self.X[np.random.randint(self.X.shape[0])]
        return centroids

    def dist_from_centroid(self):
        self.dist_from = np.array([]).reshape(0,self.X.shape[0])
        if self.metric == 'euclidean':
            for k in self.centroids.keys():
                dist = np.array([])
                for i in range(self.X.shape[0]):
                    dist = np.append(dist,self.euclidean(self.centroids[k],self.X[i]))
                self.dist_from = np.vstack((self.dist_from,dist))

    def label(self):
        self.labels = np.argwhere(self.dist_from == np.min(self.dist_from,axis=0))
        if self.labels.shape[0] != self.X.shape[0]:
            idx = np.where(np.unique(self.labels[:,1], return_counts=True)[1] == 2)
            duplicate_asignments = np.where(self.labels[:,1] == idx)[1]
            for i, dup in enumerate(duplicate_asignments):
                if i == 0:
                    self.labels = np.delete(self.labels,dup,axis=0)
        #print(self.labels)

    def recenter(self):
        for k in self.centroids.keys():
            self.centroids_hist[k] = self.centroids[k]
            self.centroids[k] = self.X[self.labels[:,0] == k].mean(axis=0)

    def stop(self):
        if self.iterations > self.max_iter:
            return True
        for k in centroids.keys():
            if np.array_equal(km.centroids_hist[k], km.centroids[k]):
                return True
        return False

    def fit(self, X):
        self.X = X
        self.centroids = self.init_centroids()
        self.centroids_hist = self.init_centroids()
        self.iterations = 0

        #while not self.stop():#self.iterations < self.max_iter:
        while self.iterations < self.max_iter:
            self.dist_from_centroid()
            self.label()
            self.recenter()
            self.iterations += 1
