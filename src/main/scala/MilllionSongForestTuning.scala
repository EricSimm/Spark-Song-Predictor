/*
Allows a grid search of hyperparameters of a model
Outputs evaluation statistics of best model and its hyperparameters
Written By: Eric Simmons
*/

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType  
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator

object MillionSongForestTuning
{

	def main(args: Array[String])
	{
		//create spark session object
		   val spark = SparkSession
		  .builder()
		  .appName("RandomForestRegTuning")
		  .getOrCreate()
			import spark.implicits._
		  //attributes to set for random forest
		  val trees = 200
		  val splitYear = 0

			//load training data and map to a dataframe and cache it
		  val trainDF = load.loadTrain(spark,42,splitYear).cache

		  val testDF = load.loadTest(spark,42, splitYear).cache


	  	  //create indexer and regressor, treat as continuous
		  val featureIndexer = new VectorIndexer()
		  	.setInputCol("features")
		  	.setOutputCol("indexedFeatures")
		  	.setMaxCategories(2)
		  	.fit(trainDF)


		  	//create regressor and pipeline
		  val rf = new RandomForestRegressor()
		  	.setLabelCol("label")
		  	.setFeaturesCol("indexedFeatures")
		  	.setNumTrees(trees)

		  val pipeline = new Pipeline()
		  	.setStages(Array(featureIndexer, rf))

		  	//add parameters that wished to be tuned
		 	val paramGrid = new ParamGridBuilder()
		  		.addGrid(rf.maxDepth, Array(15,16,17,18,20,22))
		  		.build()

		  	//create model
		  	val trainValSplit = new TrainValidationSplit()
		  		.setEstimator(pipeline)
		  		.setEvaluator(new RegressionEvaluator)
		  		.setEstimatorParamMaps(paramGrid)
		  		.setTrainRatio(0.8)
		  		.setParallelism(1)

		  	//output evaluation
		  	val model = trainValSplit.fit(trainDF)

		  	val predictions = model.transform(testDF)

		  val RMSEevaluator = new RegressionEvaluator()
		  	.setLabelCol("label")
		  	.setPredictionCol("prediction")
		  	.setMetricName("rmse")

		  val MSEevaluator = new RegressionEvaluator()
		  	.setLabelCol("label")
		  	.setPredictionCol("prediction")
		  	.setMetricName("mse")

		  val MAEevaluator = new RegressionEvaluator()
		  	.setLabelCol("label")
		  	.setPredictionCol("prediction")
		  	.setMetricName("mae")

		  val rmse = RMSEevaluator.evaluate(predictions)
		  val mse = MSEevaluator.evaluate(predictions)
		  val mae = MAEevaluator.evaluate(predictions)




		  println("\n\n\nRoot Mean Squared Error = " + rmse +
		  	"\n\nMean Squared Error = " + mse +
		  	"\n\nMean Absolute Error = " + mae +
		  	"\n\n Model: " + model.getEstimatorParamMaps.zip(model.validationMetrics).minBy(_._2)._1)


	}
}
