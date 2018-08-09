import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType  
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.LinearRegression

object MillSongLinReg
{

	def main(args: Array[String])
	{
		   val spark = SparkSession
		  .builder()
		  .appName("LinearRegressionSong")
		  .getOrCreate()
			import spark.implicits._

			val save = true
			val splitYear = 0

			//load training data and map to a dataframe and cache it

			  val trainDF = load.loadTrain(spark, 42, 0).cache.cache


		  val lr = new LinearRegression()
  			.setElasticNetParam(0.8)
  			.setStandardization(false)
  			.setMaxIter(9000000)
  			.setRegParam(0.001)
  			.setRegParam(0.5)

  		//start recording training time
		 val startTime = System.nanoTime
  		// Fit the model
		val lirModel =lr.fit(trainDF)
		//finish recording time
		val finalTimeSec = (System.nanoTime - startTime) * 1E-11

		println(s"Weights: ${lirModel.coefficients} Intercept: ${lirModel.intercept}")

		//load in test RDD and create predictions also record time

		val testDF = load.loadTrain(spark, 42, splitYear).cache

		val predictions = lirModel.transform(testDF)

		  //start evaluating random forest tree
		val predictDiff = predictions.withColumn("difference", abs($"prediction" - $"label"))


		val interval = 5
		val countDiff = predictDiff.withColumn("rangeDiff", $"difference" - ($"difference" % interval)).withColumn("rangeDiff", concat($"rangeDiff", lit(" - "), $"rangeDiff" + interval)).groupBy($"rangeDiff").count.sort("rangeDiff")
		val total = countDiff.agg(sum("count")).first.get(0)
		countDiff.withColumn("percent", round($"count" / total * 100, 2)).show(100)

		predictDiff.groupBy($"label").agg(mean("difference"),
		  	count("difference"),
		  	max("difference"),
		  	min("difference"),
		  	stddev("difference")).sort("label").show(100)


		val mae = predict.predictWTime(spark, predictions, finalTimeSec)


		val trainingSummary = lirModel.summary
   		println(s"numIterations: ${trainingSummary.totalIterations}")
    	println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    	trainingSummary.residuals.show()
    	println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    	println(s"r2: ${trainingSummary.r2}")

		if (save == true)
		  	lirModel.write.overwrite().save("hdfs://cloudlabMaster:9000/user/hduser/MillionSong/model/LinReg_" + mae + "_" + splitYear)
		spark.stop()
	}
}