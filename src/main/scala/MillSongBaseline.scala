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
 
 
object MillSongBaseline
{
 
    def main(args: Array[String])
    {
           val spark = SparkSession
          .builder()
          .appName("FindMeanOfMillionSong")
          .getOrCreate()
            import spark.implicits._

        val splitYear = 1990
 
        val trainDF= spark.sparkContext.textFile("hdfs://cloudlabMaster:9000/user/hduser/MillionSong/UCI/YearPredictionTrain.txt")
                        .map(_.split(",")).map(attribute => attribute(0).toDouble).toDF("train").filter($"train" > splitYear)
 
 
        val testDF = spark.sparkContext.textFile("hdfs://cloudlabMaster:9000/user/hduser/MillionSong/UCI/YearPredictionTest.txt")
                        .map(_.split(",")).map(attribute => attribute(0).toDouble).toDF("test").filter($"test" > splitYear)

        val diffDF = testDF.withColumn("difference", abs($"test" - trainDF.agg(avg("train")).head().getDouble(0))).withColumn("differenceSquared", pow($"test" - trainDF.agg(avg("train")).head().getDouble(0), 2))
 
        val MSE = diffDF.agg(sum("differenceSquared")).head().getDouble(0) / diffDF.count()
         
        println("Average: " + trainDF.agg(avg("train")).head().getDouble(0))
        println ("Mean Squared Error: " + MSE)
        println("Root Mean Squared Error: " + scala.math.sqrt(MSE))
        println("Absolute Error: " +  diffDF.agg(sum("difference")).head().getDouble(0) / diffDF.count())
 
    spark.stop()
 
 
    }
}
