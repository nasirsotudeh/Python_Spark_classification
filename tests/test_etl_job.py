"""
test_etl_job.py
~~~~~~~~~~~~~~~

This module contains unit tests for the transformation steps of the ETL
job defined in etl_job.py. It makes use of a local version of PySpark
that is bundled with the PySpark package.
"""
import unittest

import json

from pyspark.sql.functions import mean, col

from Kmeans.K_means import col , K_means
from conf import spark_start

class SparkETLTests(unittest.TestCase):
    """Test suite for transformation in etl_job.py
    """

    def setUp(self):
        """Start Spark, define config and path to test data
        """
        self.spark, *_ = spark_start
        self.test_data_path = 'data/'

    def tearDown(self):
        """Stop Spark
        """
        self.spark.stop()

    def test_transform_data(self):
        K_cluster = 5
        Kmeans = K_means(self.test_data_path)

                # ***  n == 2 in range 2000
                # +------------------+------------------+---+------+
                # |               f_0|               f_1| id|weight|
                # +------------------+------------------+---+------+
                # |191630.43594529165| 122254.2981798003|2001|  1.0|

        firstdata = Kmeans.makedataframe().filter(col('id').between(0, 20))
        vecAssembler = Kmeans.vectorModel()
        vector = vecAssembler.transform(firstdata).select('features', 'weight')
        Kmeans.kmeansOnData(vector , K_cluster=K_cluster)




        # assert
        self.assertEqual(firstdata, 2)


if __name__ == '__main__':
    unittest.main()
