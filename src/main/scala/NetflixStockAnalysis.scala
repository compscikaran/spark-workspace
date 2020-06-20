import org.apache.log4j.{Level, Logger};
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions._;


object NetflixStockAnalysis {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR);
    val spark = SparkSession.builder()
      .appName("Stock")
      .master("local[*]").getOrCreate();
    val df = spark.read.option("header", true).option("inferSchema", true).csv("src/main/resources/Netflix_2011_2016.csv");
    println(df.count());
    val columnList = df.columns;
    println(columnList);
    df.describe().show()
    df.withColumn("HV Ratio", df("High")/df("Volume")).show();
    df.orderBy(df("High").desc).show()
    df.select(mean(df("Close"))).show()
    df.select(max(df("Volume")), min(df("Volume"))).show()
    println(df.filter("Close < 600").count());
    println((df.filter("High > 500").count()*1.0/df.count())*100)
    df.select(corr("High","Volume")).show()
    df.withColumn("year", year(df("Date"))).select("year","High").groupBy("year").max("High").orderBy("year").show()
    df.withColumn("month", month(df("Date"))).select("month", "Close").groupBy("month").mean("Close").orderBy("month").show()
  }
}
