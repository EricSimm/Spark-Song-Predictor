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
import org.apache.spark.storage.StorageLevel

object MillSongForest
{

	def main(args: Array[String])
	{
		   val spark = SparkSession
		  .builder()
		  .appName("RandomForestReg")
		  .getOrCreate()
			import spark.implicits._



			//load training data and map to a dataframe and cache it
			val splitYear = 0


		  val trainDF = load.loadTrain(spark, 42, splitYear).cache

		  /*val intervalTrain = 10
	      val trainCount = trainDF.groupBy("label").count().sort("label")
	      trainCount.show(89)
		  val trainRange = trainDF.withColumn("Year", $"label" - ($"label" % intervalTrain)).withColumn("Year", concat($"Year", lit(" - "), $"Year" + intervalTrain)).groupBy($"Year").count.sort("Year")
		  val totalTrain = trainCount.agg(sum("count")).first.get(0)
		  trainRange.withColumn("percent", round($"count" / totalTrain * 100, 2)).show(100)
		  */
	  	  //create indexer and regressor, treat as continuous
		  val featureIndexer = new VectorIndexer()
		  	.setInputCol("features")
		  	.setOutputCol("indexedFeatures")
		  	.setMaxCategories(2)
		  	.fit(trainDF)

		  //attributes to set for random forest
		  val trees = 200
		  val depth = 15
		  val mem = 0
		  val save = true

		  println("\n\nTrees: " + trees + "\nMax Depth: " + depth + "\nMaxMemory: " + mem + "\nSplitYear: " + splitYear + "\n\n")

		  val rf = new RandomForestRegressor()
		  	.setLabelCol("label")
		  	.setFeaturesCol("indexedFeatures")
		  	.setNumTrees(trees)
		  	.setMaxDepth(depth)
		  	//.setMaxMemoryInMB(mem)

		  val pipeline = new Pipeline()
		  	.setStages(Array(featureIndexer, rf))
			//start recording time
		  val startTime = System.nanoTime
		  val model = pipeline.fit(trainDF)
		  val finalTimeSec = (System.nanoTime - startTime) * 1E-9
		  trainDF.unpersist()

		  //load in test RDD and create predictions also record time

		  val testDF = load.loadTest(spark, 42, splitYear).cache()


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

		  if (save == true)
		  	model.write.overwrite().save("hdfs://cloudlabMaster:9000/user/hduser/MillionSong/model/RandomForest_" + mae + "_" + splitYear)

		  spark.stop()

	}
}
