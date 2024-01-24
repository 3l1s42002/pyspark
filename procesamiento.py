from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col

# Configurar la sesión Spark
spark = SparkSession.builder.appName("nombre_aplicacion").getOrCreate()

# Cargar datos desde el archivo CSV
df = spark.read.csv("hdfs://ruta/al/archivo/datafinal.csv", sep=';', header=True, inferSchema=True)

# Convertir las columnas a numéricas y reemplazar comas por puntos
numeric_columns = ['O3', 'CO', 'NO2', 'SO2', 'PM2_5']
for col_name in numeric_columns:
    df = df.withColumn(col_name, col(col_name).cast("double").cast("string").cast("double"))

# Seleccionar las columnas relevantes
feature_columns = ['O3', 'CO', 'NO2', 'SO2', 'PM2_5']
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df = assembler.transform(df)

# Escalar los datos
scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=False)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Aplicar KMeans a cada columna
num_clusters = 3
kmeans = KMeans(k=num_clusters, seed=0)
model = kmeans.fit(df)
df = model.transform(df)

# Guardar el DataFrame en un archivo de Parquet (u otro formato distribuido)
df.write.parquet("hdfs://ruta/al/directorio/datafinal_clusters.parquet")
