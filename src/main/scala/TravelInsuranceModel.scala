import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.Pipeline;
import org.apache.spark.sql.Encoders;


object TravelInsuranceModel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Insurance")
      .master("local[*]").getOrCreate();

    val df = spark.read.option("header", true).option("inferSchema", true).csv("src/main/resources/travel insurance.csv");

    println(df.printSchema());
    println(df.count());
    df.groupBy("Claim").count().show();
    println(df.columns);
    val dataframes = df.randomSplit(Array(0.8,0.2));
    val trainTest = dataframes(0);
    val holdout = dataframes(1);


    val agencyIndexer = new StringIndexer().setHandleInvalid("keep").setInputCol("Agency").setOutputCol("agencyIndex");
    val typeIndex = new StringIndexer().setHandleInvalid("keep").setInputCol("Agency Type").setOutputCol("typeIndex");
    val productIndex = new StringIndexer().setHandleInvalid("keep").setInputCol("Distribution Channel").setOutputCol("productIndex");
    val destinationIndexer = new StringIndexer().setHandleInvalid("keep").setInputCol("Destination").setOutputCol("destinationIndex");
    val genderIndexer = new StringIndexer().setHandleInvalid("keep").setInputCol("Gender").setOutputCol("genderIndex");
    val claimIndexer = new StringIndexer().setHandleInvalid("keep").setInputCol("Claim").setOutputCol("label");

    val trainWithLabel = claimIndexer.fit(trainTest).transform(trainTest);
    val holdoutWithLabel = claimIndexer.fit(holdout).transform(holdout);

    val ohe = new OneHotEncoder().setInputCols(Array("agencyIndex", "typeIndex",
      "productIndex", "destinationIndex" ,"genderIndex"))
      .setOutputCols(Array("agencyVector", "typeVector", "productVector",
        "destinationVector", "genderVector"));

    val assembler = new VectorAssembler().setInputCols(Array("Duration", "Net Sales", "Commision (in value)", "Age",
      "agencyVector", "typeVector", "productVector", "destinationVector", "genderVector"))
      .setOutputCol("features");

    val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC");

    val logRegModel = new LogisticRegression();

    val pgb = new ParamGridBuilder();
    val paramMap = pgb.addGrid(logRegModel.regParam, Array(0.01,0.1,0.3,0.5,0.7,1))
      .addGrid(logRegModel.elasticNetParam,Array(0,0.5,1)).build();

    val tvs = new TrainValidationSplit().setEstimator(logRegModel).setEvaluator(evaluator)
      .setEstimatorParamMaps(paramMap).setTrainRatio(0.75);

    val pipeline = new Pipeline().setStages(Array(agencyIndexer,typeIndex,productIndex,destinationIndexer,genderIndexer,ohe,assembler,tvs));

    val pipelineModel = pipeline.fit(trainWithLabel);

    val holdoutToEval = pipelineModel.transform(holdoutWithLabel).drop("rawPrediction", "probability", "prediction")


    val bestModel = pipelineModel.stages(pipelineModel.stages.size-1).asInstanceOf[TrainValidationSplitModel].bestModel

    val logisticRegression = bestModel.asInstanceOf[LogisticRegressionModel].evaluate(holdoutToEval)

    holdoutWithLabel.groupBy("label").count().show()



    val predictions = logisticRegression.predictions.select("prediction").as(Encoders.DOUBLE).collect();
    val labels = logisticRegression.predictions.select("label").as(Encoders.DOUBLE).collect();
    val predictionsAndLabels = predictions zip labels
    val rdd = spark.sparkContext.parallelize(predictionsAndLabels);
    val metrics = new BinaryClassificationMetrics(rdd);

    println(logisticRegression.accuracy);
    println(logisticRegression.fMeasureByLabel);
    println(metrics.areaUnderROC);
  }
}
