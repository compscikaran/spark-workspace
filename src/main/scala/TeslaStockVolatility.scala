import org.apache.spark.sql.{Encoders, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._


object TeslaStockVolatility {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Stock")
      .master("local[*]").getOrCreate();
    val teslaData = spark.read.option("header", true).option("inferSchema", true).csv("/src/main/resources/TSLA.csv");
    teslaData.show();

    val w = Window.partitionBy().orderBy("Date")
    val shifted = teslaData.withColumn("Shifted", lag(col("Close"), 1, 0).over(w))
    val withReturns = shifted.withColumn("Log Returns", log(col("Close")/col("Shifted")))
    val avgReturn = withReturns.select("Log Returns").agg(avg("Log Returns")).as(Encoders.DOUBLE).collect()(0)*100;
    val withDeviation = withReturns.withColumn("Deviation", pow(col("Log Returns") - avgReturn,lit(2)))
    val volatility = withDeviation.agg(sqrt(sum(col("Deviation"))/withDeviation.count)).as(Encoders.DOUBLE).collect()(0)*100
    println(volatility);
  }
}
