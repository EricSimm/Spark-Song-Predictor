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
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.sql.functions._


//trains a model using a GBT and evaluates the model
object MillSongGBT
{

	def main(args: Array[String])
	{
		   val spark = SparkSession
		  .builder()
		  .appName("GBTReg")
		  .getOrCreate()
			import spark.implicits._



			//load training data and map to a dataframe and cache it
		  val splitYear = 1980
		  val it = 245
		  val depth = 5
		  val save = true
		  val steps = 0.11

		  println("splitYear: " + splitYear + "\nIterations: " + it + "\nMaxDepth: " + depth + "\nStepSize: " + steps)

		  val trainDF = load.loadTrain(spark , 42, splitYear)


	  	  //create indexer and regressor, treat as continuous
		  val featureIndexer = new VectorIndexer()
		  	.setInputCol("features")
		  	.setOutputCol("indexedFeatures")
		  	.setMaxCategories(2)
		  	.fit(trainDF)



		  val gbt = new GBTRegressor()
		  	.setLabelCol("label")
		  	.setFeaturesCol("indexedFeatures")
		  	.setMaxIter(it)
		  	.setMaxDepth(depth)
		  	.setStepSize(steps)

		  val pipeline = new Pipeline()
		  	.setStages(Array(featureIndexer, gbt))

		 //record train time and train the model
		  val startTime = System.nanoTime
		  val model = pipeline.fit(trainDF)

		  val finalTimeSec = (System.nanoTime - startTime) * 1E-9

		  trainDF.unpersist()

		  //load test set and make predictions based off test set
		  val testDF = load.loadTest(spark, splitYear)

		  val predictions = model.transform(testDF)


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

		  val mae = predict.predictWTime(spark, predictions, finalTimeSec)
		  println("splitYear: " + splitYear + "\nIterations: " + it + "\nMaxDepth: " + depth + "\nStepSize: " + steps)
		  if (save == true)
		  	model.write.save("hdfs://cloudlabMaster:9000/user/hduser/MillionSong/model/GBT_" + mae + "_" + splitYear)

		  spark.stop()


	}
}
