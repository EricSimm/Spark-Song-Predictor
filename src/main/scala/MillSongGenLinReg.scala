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
import org.apache.spark.ml.regression.GeneralizedLinearRegression

object MillSongGenLinReg
{

	def main(args: Array[String])
	{
		   val spark = SparkSession
		  .builder()
		  .appName("GeneralLinearRegressionSong")
		  .getOrCreate()
			import spark.implicits._



			//load training data and map to a dataframe and cache it
		  val trainRDD = spark.sparkContext.textFile("hdfs://cloudLabMaster:9000/user/hduser/MillionSong/UCI/YearPredictionTrain.txt", 21)
		  val schemaString = "label features"
		  val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, DoubleType, nullable = true))
		  val schema = StructType(Array(StructField("label", DoubleType, true), StructField("features", VectorType, true)))
		  val trainRow = trainRDD.map(_.split(","))
		  					.map(attributes => Row( attributes(0).toDouble,
		  						Vectors.dense(attributes.drop(1).take(attributes.length)
		  							.map(str => str.toDouble))))

		  val splitYear = 0
		  val trainDF = if(splitYear != 0) spark.createDataFrame(trainRow, schema).filter($"label" > splitYear).cache else spark.createDataFrame(trainRow, schema).cache


		  val glr = new GeneralizedLinearRegression()
  			.setFamily("gaussian")
  			.setLink("identity")
  			.setMaxIter(900000000)
  			.setRegParam(0.001)

  		//start recording training time
		 val startTime = System.nanoTime
  		// Fit the model
		val lirModel = glr.fit(trainDF)
		//finish recording time
		val finalTimeSec = (System.nanoTime - startTime) * 1E-9

		println(s"Weights: ${lirModel.coefficients} Intercept: ${lirModel.intercept}")

		//load in test RDD and create predictions also record time
		val testRDD = spark.sparkContext.textFile("hdfs://master:9000/user/hduser/MillionSong/UCI/YearPredictionTest.txt")
		val testRow = testRDD.map(_.split(","))
		  					.map(attributes => Row( attributes(0).toDouble,
		  						Vectors.dense(attributes.drop(1).take(attributes.length)
		  							.map(str => str.toDouble))))
		val testDF = if(splitYear != 0) spark.createDataFrame(testRow, schema).filter($"label" > splitYear) else spark.createDataFrame(testRow, schema)

		val predictions = lirModel.transform(testDF)

		  //start evaluating random forest tree
		val predictDiff = predictions.withColumn("difference", abs($"prediction" - $"label"))


		val interval = 2
		val countDiff = predictDiff.withColumn("rangeDiff", $"difference" - ($"difference" % interval)).withColumn("rangeDiff", concat($"rangeDiff", lit(" - "), $"rangeDiff" + interval)).groupBy($"rangeDiff").count.sort("rangeDiff")
		val total = countDiff.agg(sum("count")).first.get(0)
		countDiff.withColumn("percent", round($"count" / total * 100, 2)).show(100)

		predictDiff.groupBy($"label").agg(mean("difference"),
		  	count("difference"),
		  	max("difference"),
		  	min("difference"),
		  	stddev("difference")).sort("label").show(100)

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
		  	"\n\nTimetaken = " + math.floor(finalTimeSec/60).toInt + " minutes and " + finalTimeSec % 60 + " seconds" +
		  	"\nTimetaken(seconds) = " + finalTimeSec + " seconds\n")

	}
}