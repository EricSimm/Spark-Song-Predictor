/*
Original Tuning Code,
Unused
Runs multiple random forest models using a for loop
Written by Eric Simmons
*/

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType  
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.functions._


object MillionSongForestEval
{

	def main(args: Array[String])
	{
		   val spark = SparkSession
		  .builder()
		  .appName("RandomForestRegEval")
		  .getOrCreate()
			import spark.implicits._


			//load training data and map to a dataframe and cache it
		  val trainRDD = spark.sparkContext.textFile("hdfs://cloudlabMaster:9000/user/hduser/MillionSong/UCI/YearPredictionTrain.txt", 21)
		  val schemaString = "label features"
		  val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, DoubleType, nullable = true))
		  val schema = StructType(Array(StructField("label", DoubleType, true), StructField("features", VectorType, true)))
		  val trainRow = trainRDD.map(_.split(","))
		  					.map(attributes => Row( attributes(0).toDouble,
		  						Vectors.dense(attributes.drop(1).take(attributes.length)
		  							.map(str => str.toDouble))))

		 	 val splitYear = 1980
		  	val trainDF = if(splitYear != 0) spark.createDataFrame(trainRow, schema).filter($"label" > splitYear).cache else spark.createDataFrame(trainRow, schema).cache

		  	val testRDD = spark.sparkContext.textFile("hdfs://cloudlabMaster:9000/user/hduser/MillionSong/UCI/YearPredictionTest.txt")
		  	val testRow = testRDD.map(_.split(","))
		  					.map(attributes => Row( attributes(0).toDouble,
		  						Vectors.dense(attributes.drop(1).take(attributes.length)
		  							.map(str => str.toDouble))))
		  val testDF = if(splitYear != 0) spark.createDataFrame(testRow, schema).filter($"label" > splitYear).cache else spark.createDataFrame(testRow, schema).cache


	  	  //create indexer and regressor, treat as continuous
		  val featureIndexer = new VectorIndexer()
		  	.setInputCol("features")
		  	.setOutputCol("indexedFeatures")
		  	.setMaxCategories(2)
		  	.fit(trainDF)

		  //attributes to set for random forest
		  val trees = 200
		  val start = 16
		  val stop = 30

		  import scala.collection.mutable.ListBuffer
		  val tuning : ListBuffer[(Int, Double, Double)] = ListBuffer()

		for (depth <- Range(start,stop)) {
		  val rf = new RandomForestRegressor()
		  	.setLabelCol("label")
		  	.setFeaturesCol("indexedFeatures")
		  	.setNumTrees(trees)
		  	.setMaxDepth(depth)


		  val pipeline = new Pipeline()
		  	.setStages(Array(featureIndexer, rf))

		  val model = pipeline.fit(trainDF)


		  val predictions = model.transform(testDF)

		  val RMSEevaluator = new RegressionEvaluator()
		  	.setLabelCol("label")
		  	.setPredictionCol("prediction")
		  	.setMetricName("rmse")

		  val MAEevaluator = new RegressionEvaluator()
		  	.setLabelCol("label")
		  	.setPredictionCol("prediction")
		  	.setMetricName("mae")

		  	val rmse = RMSEevaluator.evaluate(predictions)
		  	val mae = MAEevaluator.evaluate(predictions)

		  	println(depth + " " + rmse + " " + mae)
		  	tuning+=((depth, rmse, mae))
		  }

		  tuning.foreach{
		  	x => println("Depth: " + x._1 + " RMSE: " + x._2 + " MAE: " + x._3)
		  }

	}
}
