import os, sys
from pyspark.storagelevel import StorageLevel

# 1) Point Spark to JDK 17
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH","")

# 2) (Belt-and-braces) allow security manager if stray newer JDK got picked
os.environ["SPARK_SUBMIT_OPTS"] = "-Djava.security.manager=allow"

# 3) Now import Spark and start session
from pyspark.sql import SparkSession,functions as F
from pyspark import SparkContext

# clean any zombie context
if SparkContext._active_spark_context is not None:
    SparkContext._active_spark_context.stop()


spark = (SparkSession.builder
  .appName("NOAA Reader")
  .config("spark.driver.memory", "12g")
  # Read smaller batches from Parquet to reduce per-task memory spikes
  .config("spark.sql.parquet.enableVectorizedReader", "false")   # or keep true but reduce batch size below
  .config("spark.sql.parquet.columnarReaderBatchSize", "1024")   # default 4096
  # Make input splits smaller so each task handles less at once
  .config("spark.sql.files.maxPartitionBytes", "64m")            # default 128m
  # Keep planning scalable for many files
  .config("spark.sql.sources.parallelPartitionDiscovery.threshold", "100000")
  .config("spark.sql.adaptive.enabled", "true")
  .getOrCreate())

# Read Parquet
df = spark.read.parquet("data/noaa_parquet/2024_compacted")
df = df.repartition(200, "year", "month")
df = df.persist(StorageLevel.MEMORY_AND_DISK)
df.printSchema()
df.show(5, truncate=False)
# df.cache()

# # Number of rows
print("Row count:", df.count())