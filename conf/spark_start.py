"""
spark.py
~~~~~~~~

Module containing helper function for use with Apache Spark
"""
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from os import environ
from contextlib import contextmanager
from pyspark import SparkContext
from pyspark import SparkConf

# SPARK_MASTER = 'spark://192.168.80.128:7077'
SPARK_MASTER = 'local'
SPARK_APP_NAME = 'Kmeans'

def sparksession():
    spark = SparkSession.builder.appName("readstream")
    return spark


def sparkcontext():
    conf = SparkConf().setMaster(SPARK_MASTER) \
        .setAppName(SPARK_APP_NAME)

    try:
        spark_context = SparkContext(conf=conf)
        spark_context = spark_context.getOrCreate()
    except:
        SparkContext('*').stop()
        spark_context = SparkContext().getOrCreate()


    sc = spark_context
    ssc = StreamingContext(sc, 10)
    ssc.checkpoint('file:///tmp/spark')
    return sc