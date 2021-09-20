
## Kmeans python pyspark 
##### directory tree

```
├── run_kmeans.py
├── requirment.txt
├── README.md
├── output
│   ├── summaries
│   │   ├── kmeans2
│   │   │   ├── summery2.txt
│   │   │   └── input_data2.txt
│   │   └── kmeans1
│   │       ├── summery1.txt
│   │       └── input_data1.txt
│   └── plots
│       ├── Wkmeans2.png
│       └── Wkmeans1.png
├── Kmeans
│   ├── K_means.py
│   └── __init__.py
├── data
│   └── data_128000_8_25_2noshuffle
└── conf
    ├── spark_start.py
    └── __init__.py
       
```

Automated Testing

In order to test with Spark, we use the pyspark Python package, which is bundled with the Spark JARs required to programmatically start-up and tear-down a local Spark instance, on a per-test-suite basis (we recommend using the setUp and tearDown methods in unittest.TestCase to do this once per test-suite). Note, that using pyspark to run Spark is an alternative way of developing with Spark as opposed to using the PySpark shell or spark-submit.

Given that we have chosen to structure our ETL jobs in such a way as to isolate the 'Transformation' step into its own function (see 'Structure of an ETL job' above), we are free to feed it a small slice of 'real-world' production data that has been persisted locally - e.g. in tests/test_data or some easily accessible network directory - and check it against known results (e.g. computed manually or interactively within a Python interactive console session).

To execute the example unit test for this project run,

```pipenv run python -m unittest tests/test_*.py
```

### install pyspark

download with command
```
wget https://downloads.apache.org/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz
```
Tar file
```
tar -xvzf spark-*
```
move to /opt/
```
mv spark-3.0.1-bin-hadoop2.7/ /opt/spark
```
##### .profile config
```
root@ubuntu1804:~# echo "export SPARK_HOME=/opt/spark" >> ~/.profile

root@ubuntu1804:~# echo "export PATH=$PATH:/opt/spark/bin:/opt/spark/sbin" >> ~/.profile

root@ubuntu1804:~# echo "export PYSPARK_PYTHON=/usr/bin/python3" >> ~/.profile

```

source ~/.profile

#### .bashrc config
```
export SPARK_HOME=/opt/spark
export PATH=$SPARK_HOME/bin:$PATH:$SPARK_HOME/python:$PATH
export PATH=$PATH:$SPARK_HOME/bin
export PYTHONPATH=$SPARK_HOME/PYTHON:$PYTHONPATH
export PYSPARK_PYTHON=python3.8

```
### config PYCHARM

create new project with python3.8 for example and pip env

follow path
 -> edit configurations -> environment variables and specific SPARK_HOME adn PYTHONPATH py4j
```
 SPARK_HOME =/opt/spark;PYTHONPATH=/opt/spark/python/lib/py4j-0.10.9-src.zip
```

pip intstall requirment.txt

```pip install pyspark```
____
In cluster job run master for manage  and slave fo UI support
`./start-master.sh -h 192.168.80.128 -p 8080`
`./start-slave.sh spark://192.168.80.128:8080`

````
SPARK_MASTER = 'spark://192.168.80.128:8080'
SPARK_APP_NAME = 'Kmeans'

````

The written code has items that are read in a range of 1000 from the file and a kmeans step is applied
Each item with the kmeans ++ method
Is selected.

The center of each cluster is considered as a property and merges with a thousand other points, of which another thousand points weigh 1 and their prediction is null.
The previous step is repeated in the same way.

```from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=50, maxIter=1, featuresCol='features', initSteps=3, weightCol='weight')

def setParams(self, featuresCol="features", predictionCol="prediction", k=2,
                  initMode="k-means||", initSteps=2, tol=1e-4, maxIter=20, seed=None,
                  distanceMeasure="euclidean", weightCol=None):
```
                  
                              
initMode="k-means|| => Kmmeans ++ 
 
weightCol='weight' set weight column

##### * Get the cluster centers, represented as a list of NumPy arrays sorted by prediction

```
    kmm = kmeans.fit(vector)
    centers = kmm.clusterCenters()
    
[192028.64430965524,121590.81737107041]
[190810.6852672185,119556.44188337866]
[190971.6772525595,122856.41394189643]

 ```
we have one dataframe with prediction column and weight 
we must know prediction cluster center so change to dataframe cluster centers array

` dcenters = [e.tolist() for e in centers]
    dfcenters = sc.parallelize(dcenters).toDF([])`
    
    
now we merge perdiction dataframe so we have for each cluster center prediction and weight.

    # +--------------------+----------+------+------+
    # | clusterCenters array ----->  features | id
    # +--------------------+----------+------+------+
    
```
    vecAssemblers = VectorAssembler(inputCols=[v for v in dfcenters.columns], outputCol="features")
    dfcenters = vecAssemblers.transform(dfcenters)
    dfcenters = dfcenters.coalesce(1).withColumn('id', monotonically_increasing_id().cast('integer'))

    result = dfcenters.join(newdf, dfcenters.id == newdf.prediction, 'outer')
    result = result.withColumnRenamed('count', 'weight').select('features', 'prediction', 'weight').sort('prediction')
    # result.show(100)

    # +--------------------+----------+------+
    # | features           |prediction| weight |
    # +--------------------+----------+------+
```
in result features are cluster centers


` makedataframe`: Converts any multidimensional data to a custom data frame regardless of the number of items.

```
def makedataframe(dataframe):
    items = rowsplit(dataframe)
    for item in range(0, items):
        dataframe = dataframe.withColumn('f_{}'.format(item), split(dataframe.value, ' ').getItem(item).cast('double'))

```   
`vectorModel`:
We convert all created columns, each of which is a dimension of our point, into a property features
```
def vectorModel(items):
    mylist = []
    for item in range(0, items):
        col = "f_{}".format(item)
        mylist.append(col)
    # print('list = ' + mylist.__str__())
    vecAssembler = VectorAssembler(inputCols=mylist, outputCol="features")
    return vecAssembler
```   

merge Previous  Kmeans centers with more weight to next vectors

```
 if n > 1:
     vectors = customUnion(vector, result)

```
-------
merge tow dataframe 
```
def customUnion(df1, df2):
    cols1 = df1.columns
    cols2 = df2.columns
    total_cols = sorted(cols1 + list(set(cols2) - set(cols1)))

    def expr(mycols, allcols):
        def processCols(colname):
            if colname in mycols:
                return colname
            else:
                return lit(None).alias(colname)

        cols = map(processCols, allcols)
        return list(cols)

    appended = df1.select(expr(cols1, total_cols)).union(df2.select(expr(cols2, total_cols)))
    return appended
```