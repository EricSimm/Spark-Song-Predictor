/*
Loads in models from HDFS and Outputs predictions based on test set loaded into HDFS
Written By Eric Simmons
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
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType  
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import scala.util.Try

object MillionSong
{

	def main(args: Array[String])
	{
		//create Spark and Hadoop Configurations objects
		   val spark = SparkSession
		  .builder()
		  .appName("RandomForestReg")
		  .getOrCreate()
			import spark.implicits._

			val conf = spark.sparkContext.hadoopConfiguration
			val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI("hdfs://cloudlabMaster:9000/user/hduser/MillionSong/model"),conf)
			var modelName = "n"
			var fileExist = false

			//Displays all models in HDFS and asks user for model and how to display model evaluators
			import  org.apache.hadoop.fs.{FileSystem,Path}
			val listF = fs.listStatus(new Path("hdfs://cloudlabMaster:9000/user/hduser/MillionSong/model"))
			do
			{
				println("Files in Hadoop Directory:")
				listF.foreach(x=>println(x.getPath.getName))
				print("Enter the name of the model you wish to run (q to exit): ")
				modelName = scala.io.StdIn.readLine()
				if(modelName.toLowerCase == "q")
				{
				spark.stop()
				System.exit(1)
				}
				fileExist = fs.exists(new Path("hdfs://cloudlabMaster:9000/user/hduser/MillionSong/model/" + modelName))

				if(!fileExist)
					println("\n\nFile does not exist in hdfs://cloudlabMaster:9000/user/hduser/MillionSong/model/")

			}
			while(!fileExist)


			print("Enter the lowest year you wish to test on (0 to filter none, cannot be negative): ")
			var sYear = scala.io.StdIn.readLine()

			while(!Try(sYear.toInt).isSuccess)
			{
				println("Please enter a valid numeric value")
				print("Enter the lowest year you wish to test on: ")
				sYear = scala.io.StdIn.readLine()
			}
			val splitYear = sYear.toInt - 1



			print("Interval to examine years (0 will group all together): ")
			var i = scala.io.StdIn.readLine()

			while(!Try(i.toInt).isSuccess)
			{
				println("Not a numeric value")
				print("Interval to examine years: ")
				i = scala.io.StdIn.readLine()
			}

			val interval = scala.math.abs(i.toInt)

		//load in model and output predictions
		  val model = PipelineModel.load("hdfs://cloudlabMaster:9000/user/hduser/MillionSong/model/" + modelName)



		  val testDF = load.loadTest(spark, 42, splitYear)


		  val predictions = model.transform(testDF)


		  val startTimeSec = System.nanoTime
		  val predictDiff = predictions.withColumn("difference", abs($"prediction" - $"label"))
		  val finalTimeSec = (System.nanoTime - startTimeSec) * 1E-9

		  val countDiff = predictDiff.withColumn("rangeDiff", $"difference" - ($"difference" % interval)).withColumn("rangeDiff", concat($"rangeDiff", lit(" - "), $"rangeDiff" + interval)).groupBy($"rangeDiff").count.sort("rangeDiff")
		  val total = countDiff.agg(sum("count")).first.get(0)
		  countDiff.withColumn("percent", round($"count" / total * 100, 2)).show(100)

		  predictDiff.groupBy($"label").agg(mean("difference"),
		  	count("difference"),
		  	max("difference"),
		  	min("difference"),
		  	stddev("difference")).sort("label").show(100)

		  val mae = predict.predict(spark, predictions)

		  spark.stop()

	}
}
