
/*
Used to load in millionsong datasets
Written by Eric Simmons
*/


object load
{

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

	//load in traning set, part = number of partions
	//splitYear = smallest year that will appear in the dataset, set to 0 to disable this option
	def loadTrain(spark: SparkSession, part: Int, splitYear: Int): DataFrame = {
			val trainDF = load(spark, part, splitYear, "hdfs://cloudlabMaster:9000/user/hduser/MillionSong/UCI/YearPredictionTrain.txt")
		  	return trainDF
	}

	//load in test set, partitions are inherited from HDFS
	//splitYear = smallest year that will appear in the dataset, set to 0 to disable this option
	def loadTest(spark: SparkSession, splitYear: Int): DataFrame = {
		  val testDF = load(spark, splitYear, "hdfs://cloudlabMaster:9000/user/hduser/MillionSong/UCI/YearPredictionTest.txt")
		  return testDF
	}

	//load in test set, part = number of partions
	//splitYear = smallest year that will appear in the dataset, set to 0 to disable this option
	def loadTest(spark: SparkSession, part: Int, splitYear: Int): DataFrame = {
		  val testDF = load(spark, part, splitYear, "hdfs://cloudlabMaster:9000/user/hduser/MillionSong/UCI/YearPredictionTest.txt")
		  return testDF
	}

	//used by loadTest without the paritition option
	private def load(spark: SparkSession, splitYear: Int, file: String): DataFrame = {
		import spark.implicits._ 
			val orginRDD = spark.sparkContext.textFile(file)
		 	val schemaString = "label features"
		 	val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, DoubleType, nullable = true))
		 	val schema = StructType(Array(StructField("label", DoubleType, true), StructField("features", VectorType, true)))
		 	val orginRow = orginRDD.map(_.split(","))
		  					.map(attributes => Row( attributes(0).toDouble,
		  					Vectors.dense(attributes.drop(1).take(attributes.length)
		  						.map(str => str.toDouble))))
		  	val finalDF = if(splitYear != 0) spark.createDataFrame(orginRow, schema).filter($"label" > splitYear) else spark.createDataFrame(orginRow, schema)
		  	return finalDF
	}

	//logic for loadTrain and loadTest with the partition option
	private def load(spark: SparkSession, part: Int, splitYear: Int, file: String): DataFrame = {
		import spark.implicits._ 
			val orginRDD = spark.sparkContext.textFile(file, part)
		 	val schemaString = "label features"
		 	val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, DoubleType, nullable = true))
		 	val schema = StructType(Array(StructField("label", DoubleType, true), StructField("features", VectorType, true)))
		 	val orginRow = orginRDD.map(_.split(","))
		  					.map(attributes => Row( attributes(0).toDouble,
		  					Vectors.dense(attributes.drop(1).take(attributes.length)
		  						.map(str => str.toDouble))))
		  	val finalDF = if(splitYear != 0) spark.createDataFrame(orginRow, schema).filter($"label" > splitYear) else spark.createDataFrame(orginRow, schema)
		  	return finalDF
	}
}