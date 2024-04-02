from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql.functions import *

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
logger = glueContext.get_logger()

# Read image metadata into Spark DF directly from S3
ddi_meta_df = spark.read\
                    .format("csv")\
                    .option("header", "true")\
                    .load("s3://poc-skin-disease-detection-cv/ddidiversedermatologyimages/metadata/raw/")

# Print original schema & data to CloudWatch
logger.info("DDI Original Schema:")
ddi_meta_df.printSchema()
ddi_meta_df.show(5)

# Drop unneeded "features" from DF
drop_cols = ["_c0", "DDI_ID"]
for column in drop_cols:
    ddi_meta_df = ddi_meta_df.drop(col(column))
    
# Convert "malignant" target variable from strings to binary output
ddi_meta_df = ddi_meta_df.withColumn(
    "malignant",
    when(col("malignant") == "True", 1).otherwise(0)
)
    
# Print transformed schema & data to CloudWatch    
logger.info("DDI Transformed Schema:")
ddi_meta_df.printSchema()
ddi_meta_df.show(5)

# Write Spark DF to S3 for crawling into Glue Data Catalog
ddi_meta_df.write\
            .format("csv")\
            .option("header", "true")\
            .mode("overwrite")\
            .save("s3://poc-skin-disease-detection-cv/ddidiversedermatologyimages/metadata/transform/")
