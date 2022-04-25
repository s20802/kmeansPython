class Cluster:
    
    def __init__(self, clusters_amount=2, epochs=50):

        self.clusters_n = clusters_amount
        self.epochs = epochs
        self.cluster_centers = None
        self.clusters = None

    def fit(self, data):
        self.init_centers(data)
        for epoch in range(self.epochs):
            clusters_dict = {}
            for cluster in range(self.clusters_n):
                clusters_dict[cluster] = None

            for datapoint in data:
                dist = None
                
                for index, center in enumerate(self.cluster_centers):
                    new_dist = self.euclid(datapoint, center)
                    if dist == None:
                        dist = new_dist
                        cluster_index = index

                    elif  new_dist < dist:
                        dist = new_dist
                        cluster_index = index
                if clusters_dict[cluster_index] is None:
                    clusters_dict[cluster_index] = [datapoint]
                
                else:
                    clusters_dict[cluster_index] = np.vstack((clusters_dict[cluster_index], [datapoint]))
            if epoch == 0:
                '''THIS STEP IS JUST TO STORE THE INITIAL SETUP TO CONSTRUCT THE GRAPHS'''
                Innit = clusters_dict

            for index, center in enumerate(clusters_dict):
                if index == 0:
                    self.cluster_centers = self.new_center(clusters_dict[center])
                
                else:
                    self.cluster_centers = np.vstack((self.cluster_centers, self.new_center(clusters_dict[center])))

                
        clusters_0 = self.assign_labels(Innit)
        '''clusters_0 IS JUST TO CONSTRUCT THE INITIAL GRAPH, IT HAS NO USE OTHERWISE'''
        self.clusters = self.assign_labels(clusters_dict)
        return [self.clusters, clusters_0]

    def assign_labels(self, data):

        for Index, key in enumerate(data):
            if Index == 0:
                clusters = np.hstack((data[key], [[key] for i in range(len(data[key]))]))
            
            else:
                clusters = np.vstack((clusters, np.hstack((data[key], [[key] for i in range(len(data[key]))]))))

        return clusters
        
    def init_centers(self, data):

        Set = set()
        Buffer = 0
        while Buffer != self.clusters_n:
            Index = random.randrange(len(data))
            
            if Index in Set:
                continue

            elif Buffer == 0:
                self.cluster_centers = np.array((data[Index]))
                Buffer+=1
                Set.add(Index)
            
            else:
                Buffer+=1
                Set.add(Index)
                self.cluster_centers = np.vstack((self.cluster_centers, data[Index]))

    def euclid(self, data_1, data_2):

        return np.sqrt(np.sum(np.square(data_1-data_2)))

    def new_center(self, data):

        return np.mean((data), axis=0)

    def predict (self, datapoint,  knn = 5):
        
        if knn > len(self.clusters):
            raise Exception("Number of neighbhors can't surpass the amount of data, please choose a lower number for 'knn'")
            
        Closest_neighbors = [-1]*knn
        Closest_neighbors_distances = [-1]*knn

        for data in self.clusters:
            distance = self.euclid(data[:-1], datapoint)

            for Index, neighbor_distance in enumerate(Closest_neighbors_distances):
                if neighbor_distance < 0:
                    Closest_neighbors[Index] = data
                    Closest_neighbors_distances[Index] = distance
                    break

                elif distance < neighbor_distance:
                    Closest_neighbors_distances = self.insert(Closest_neighbors_distances, distance, Index)
                    Closest_neighbors = self.insert(Closest_neighbors, data, Index)
                    break
        
        Counter = [0]*self.clusters_n

        for neighbor in Closest_neighbors:
            Counter[neighbor[-1]] += 1

        Prediction = 0
        Value = 0

        for Index, Count in enumerate(Counter):
            if Count > Value:
                Prediction = Index
                Value = Count

        return Prediction
        
    def insert (self, array, value, index):

        if index == 0:
            new_array = []
            new_array.append(value)
            for Index in range(len(array)-1):
                new_array.append(array[Index])
            array = new_array
        else:

            for Index in range(len(array)-index):
                array[-1-Index] = array[-2-Index]
            array[index] = value

        return array
