import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType  
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.DataFrame
import scala.math.BigDecimal


object predict
{
	def predict(spark: SparkSession, predictions: DataFrame)
	{
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
		  	"\n\nMean Absolute Error = " + mae)

	}

	//given a DataFrame of predictions and FinalTime in seconds
	//prints out RMSE, MSE, MAE and final time converted to minutes and final time in seconds
	def predictWTime(spark: SparkSession, predictions: DataFrame, finalTime: Double): Double =
	{
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
		  	"\n\nTime taken to = " + math.floor(finalTime/60).toInt + " minutes and " + finalTime % 60 + " seconds" +
		  	"\nTime taken(seconds) = " + finalTime + " seconds\n")
		  return BigDecimal(mae).setScale(4, BigDecimal.RoundingMode.HALF_UP).toDouble
	}
}
