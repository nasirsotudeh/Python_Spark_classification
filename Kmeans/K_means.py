import os

import numpy as np
import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.stat import Summarizer
from pyspark.sql.functions import *

from pyspark.sql.types import StructType, FloatType
from pyspark.ml.clustering import KMeans
from conf.spark_start import sparkcontext, sparksession

from matplotlib import pyplot as plt
from pyspark.ml.functions import vector_to_array
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from pyspark.ml.evaluation import ClusteringEvaluator

from scipy.spatial import distance


class K_means(object):

    def __init__(self, path):
        """
            Param: path : './data/'
            Param: window : int number
        """
        self.setPath(path)
        self.setspark()

        self.data = self.getData()
        self.items = self.rowsplit()

    def setIteration(self, n):
        self.Iteratr = n

    def setPath(self, path):
        self.input_data = path

    def setspark(self):
        self.spark = sparksession()
        self.sc = sparkcontext()

    def getSparkContext(self):
        return self.sc

    def getData(self):
        userSchema = StructType().add('value', 'string')
        data = self.spark.getOrCreate().read.format('text').schema(userSchema).load(path=self.input_data)

        return data

    def rowsplit(self):
        row = self.data.head()
        s = (row.__str__().split(' '))
        rowparallel = self.sc.parallelize(s)
        items = rowparallel.count() - 1
        return items

    def numberOfwindows(self):
        nubrow = self.data.count()
        windnumb = nubrow / 1000
        # print(windnumb)
        return windnumb

    def setDensVectors(self, dcenters):
        self.dcenters = dcenters

    def getDensVectors(self):
        return self.dcenters()

    def setDfCenters(self, dfcenters):
        self.DfCenters = dfcenters

    def makedataframe(self):
        # items = rowsplit(dataframe)
        dataframe = self.data
        items = self.items
        for item in range(0, items):
            dataframe = dataframe.withColumn('f_{}'.format(item),
                                             split(dataframe.value, ' ').getItem(item).cast('double'))

        #   *** dstream.coalesce(1) : for use monotonically_increasing_id() to make col id , we must have 1 partition
        dataframe = dataframe.coalesce(1).withColumn('id', monotonically_increasing_id().cast('integer'))
        dataframe = dataframe.withColumn('weight', lit(1).cast('float')).drop(dataframe.value)
        # +------------------+------------------+---+------+
        # | f_0                |           f_1      |id | weight|
        # +------------------+------------------+---+------+
        # | 191412.07525194966 | 121427.26200208218 | 0 |   1.0 |
        # | 192059.92505690033 | 121879.55553366328 | 1 |   1.0 |
        dataframe.show()
        return dataframe

    # %%
    def savedata(self, data, window):
        outname = 'input_data{}.txt'.format(window)

        outdir = './output/summaries/kmeans{}'.format(window)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fullname = os.path.join(outdir, outname)

        data.toPandas().to_csv(fullname, index=False)

    def vectorModel(self):
        # *
        # * code
        # *
        return vecAssembler

    def getdistFromsummary(self, df):
        get_dist = udf(lambda features, center:
                       float(distance.euclidean(features, center)), FloatType())

        df_pred = df.withColumn('dist', get_dist(col('features'), col('Centers')))
        df_pred = df_pred.join(df_pred.groupBy('Centers').min('dist').alias('mindist'), 'Centers' , 'outer')
        return df_pred

    def savedataframe(self, df):
        window = self.Iteratr
        outname = 'summery{}.txt'.format(window)

        outdir = './output/summaries/dataFrame{}'.format(window)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fullname = os.path.join(outdir, outname)
        df.toPandas().to_csv(fullname, index=False)

    def joinCenters(self, df):
        print('start')
        dfC = df.select('prediction', 'Centers').sort(col('prediction').asc()).distinct()
        for item in dfC.collect():
            distance_udf = F.udf(lambda x , m: float(distance.euclidean(x, item['Centers'])) - float(m), FloatType())
            print('############################')
            f =item['prediction']
            df = df.withColumn('P_{0}'.format(str(item)), when(col('prediction') == f , 00.00).otherwise(distance_udf(col('features') , col('min(dist)'))))
            df.show(1000)
        self.savedataframe(df)
        return df



    # ##################################################
    def kmeansOnData(self, dataframe, K_cluster):

       *
        # +----------+-----+
        # |prediction|count|
        # +----------+-----+
        # |         0|   38|
        #
        # **
        # **
        # +--------------------+----------+------+------+
        # | clusterCenters array ----->  features | id
        # +--------------------+----------+------+------+
        # **
        # **
        # +--------------------+----------+------+
        # | features           |prediction| weight |
        # +--------------------+----------+------+
        return result

    # %%
    from pyspark.sql.functions import lit


    def customUnion(self, df1, df2):
        cols1 = df1.columns
        cols2 = df2.columns
        # **
        # ** code
        # **
        return appended

    # %%

    def vectorToarray(df):
        df = df.withColumn("f_", vector_to_array("features")).select([col("f_")[i] for i in range(2)]). \
            toPandas()
        return df

    def plotDataframe(self, number, newdata, result, lastresult=None):
        def minor_tick(x, pos):
            if not x % 100000.0:
                return ""
            return f"{x:.0f}"

        fig = plt.figure(dpi=300)
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        # plt.rcParams.update({'figure.figsize': (12, 10), 'figure.dpi': 100})
        plt.grid(which='minor', alpha=0.2)
        plt.grid()
        ax.tick_params(which='minor', width=1.0)
        ax.tick_params(axis='x', which='major', length=2, labelrotation=45.0, labelsize=8)
        ax.tick_params(axis='x', which='minor', length=2, labelsize=7, labelrotation=45.0, labelcolor='0.25', pad=0.5)
        ax.tick_params(axis='y', which='minor', length=2, labelsize=6, labelcolor='0.25')

        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.xaxis.set_minor_formatter(minor_tick)
        ax.yaxis.set_minor_formatter(minor_tick)

        if lastresult:
            df = K_means.vectorToarray(newdata)
            new = ax.scatter(df['f_[0]'], df['f_[1]'], alpha=0.1)

            df = K_means.vectorToarray(lastresult)
            lastclusters = ax.scatter(df['f_[0]'], df['f_[1]'], alpha=0.7)

            df = K_means.vectorToarray(result)
            newclusters = ax.scatter(df['f_[0]'], df['f_[1]'])

            plt.legend((new, lastclusters, newclusters),
                       ('New data', 'last clusters', 'new clusters'),
                       scatterpoints=1,
                       loc='center', ncol=3,
                       fontsize=8, bbox_to_anchor=(0.5, 1.012), shadow=False, frameon=False)
        else:
            df = K_means.vectorToarray(newdata)
            new = ax.scatter(df['f_[0]'], df['f_[1]'], alpha=0.1)

            df = K_means.vectorToarray(result)
            clusters = ax.scatter(df['f_[0]'], df['f_[1]'])
            plt.legend((new, clusters),
                       ('New data', 'clusters'),
                       scatterpoints=1, loc='center', ncol=3,
                       fontsize=8, bbox_to_anchor=(0.5, 1.012), shadow=False, frameon=False)

        outname = 'Wkmeans{}.png'.format(number)
        outdir = './output/plots'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fullname = os.path.join(outdir, outname)

        plt.savefig(fullname, bbox_inches='tight')
        # plt.show()
        plt.close()

    def saveSummary(df, window):
        df = df.withColumn("f_", vector_to_array("features")). \
            select([col("f_")[i] for i in range(2)] + ["prediction"] + ['weight'])

        outname = 'summery{}.txt'.format(window)

        outdir = './output/summaries/kmeans{}'.format(window)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fullname = os.path.join(outdir, outname)

    def saveInputData(self, df, window):
        df = df.select('prediction', 'weight', 'features').withColumn("f_", vector_to_array("features")). \
            select([col("f_")[i] for i in range(2)] + ['weight'])

        outname = 'input_data{}.txt'.format(window)

        outdir = './output/summaries/kmeans{}'.format(window)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fullname = os.path.join(outdir, outname)

        df.toPandas().to_csv(fullname, index=False)

