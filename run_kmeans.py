from pyspark.sql.functions import lit

from Kmeans.K_means import K_means , col

def main():
    path = 'data/'
    K_cluster = 5
    Kmeans = K_means(path)
    myrange = 100
    global n
    for n in range(1, int(Kmeans.numberOfwindows())):
        def rangeData(input_df):
            Kmeans.setIteration(n)
            # *** return dataFrame in spacial range can be customize
            # :param input_df : dataFrame
            if n == 1:
                firstdata = input_df.filter(col('id').between(0, n * myrange))
                Kmeans.savedata(firstdata,window= n)
                return firstdata
            elif n > 1:
                return input_df.filter(col('id').between((n * myrange) - (myrange - 1), n * myrange))
            # ***  n == 2 in range 2000
            # +------------------+------------------+---+------+
            # |               f_0|               f_1| id|weight|
            # +------------------+------------------+---+------+
            # |191630.43594529165| 122254.2981798003|2001|  1.0|
        newFrame = rangeData(Kmeans.makedataframe())
        vecAssembler = Kmeans.vectorModel()
        vector = vecAssembler.transform(newFrame).select('features', 'weight')
        newFrame.unpersist()

        print(n)
        if n == 1:
            result = Kmeans.kmeansOnData(vector , K_cluster=K_cluster)
            lastresult = None
        if n > 1:
            vectors = Kmeans.customUnion(vector, result)
            # +--------------------+----------+------+
            # | features            |prediction| weight |
            # +--------------------+----------+------+
            # | [190224.864622894... | null    | 1.0 |
            # | [191803.153696760... | 3       | 2.0 | last cluster center
            #  +--------------------+----------+------+
            # vectors = vectors.withColumn('weight', lit(1).cast('float'))
            Kmeans.saveInputData(vectors , window= n)
            lastresult = result
            result.unpersist()
            result = Kmeans.kmeansOnData(vectors.select('features', 'weight' ,'prediction') , K_cluster=K_cluster)

            vectors.unpersist()
            # +--------------------+----------+------
            # |  features = clusterCenters
            # +--------------------+----------+------
            if n == int(Kmeans.numberOfwindows() + 1):
                sc = Kmeans.getSparkContext()
                sc.stop()

        result.show(5, truncate=False)
        K_means.saveSummary(result, window=n)
        Kmeans.plotDataframe(newdata=vector, result=result, lastresult=lastresult, number=n)
        vector.unpersist()

if __name__ == '__main__':
    main()
