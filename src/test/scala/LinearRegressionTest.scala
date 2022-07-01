package org.apache.spark.ml.kamenev

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

//noinspection UnstableApiUsage
class LinearRegressionTest extends AnyFlatSpec with Matchers {
    private val DELTA = 1e-3
    private val ROWS: Int = 10000

    private val spark: SparkSession = SparkSession
        .builder()
        .master("local[*]")
        .appName("lrm")
        .getOrCreate()

    import spark.implicits._

    private val weights: DenseVector[Double] = DenseVector(1.5, 0.3, -0.7)
    private val biasedWeights: DenseVector[Double] = DenseVector.vertcat(DenseVector(0.0), weights)

    private val x: DenseMatrix[Double] = DenseMatrix.rand[Double](rows = ROWS, cols = 3)
    private val y: DenseVector[Double] = x * weights
    private val matrix: DenseMatrix[Double] = DenseMatrix.horzcat(x, y.asDenseMatrix.t)

    private val frame: DataFrame = matrix(*, ::)
        .iterator
        .map(r => Tuple4(r(0), r(1), r(2), r(3)))
        .toSeq
        .toDF("x_1", "x_2", "x_3", "y")

    private val assembler: VectorAssembler = new VectorAssembler().setInputCols(Array("x_1", "x_2", "x_3")).setOutputCol("x")
    private val dataset: Dataset[_] = assembler.transform(frame).select("x", "y")

    private def validate(weights: DenseVector[Double]): Unit = {
        for (i <- 0 until weights.length) {
            weights(i) should be(biasedWeights(i) +- DELTA)
        }
    }

    private def validateDataframe(transformed: DataFrame): Unit = {
        transformed.columns should be(Seq("x", "y", "y_pred"))

        transformed.collect().length should be(dataset.collect().length)

        val predicted: Array[Row] = transformed.select("y_pred").collect()

        for (i <- 0 until ROWS) {
            predicted.toVector(i).getDouble(0) should be(y(i) +- DELTA)
        }
    }

    "LRM" should "predict" in {
        val model = new LinearRegressionModel(biasedWeights)
        val transform = model.transform(dataset)

        validateDataframe(transform)
    }

    "LRM" should "save and read" in {
        val regression = new LinearRegression()
        val pipeline = new Pipeline().setStages(Array(regression))
        val fit = pipeline.fit(dataset)

        val tempDir = Files.createTempDir()
        fit.write.overwrite().save(tempDir.getAbsolutePath)

        val model = PipelineModel.load(tempDir.getAbsolutePath)
        val transform = model.transform(dataset)

        validateDataframe(transform)
    }

    "LR" should "fit without bias" in {
        val estimator = new LinearRegression().setLearnBias(false)
        val fit = estimator.fit(dataset)

        validate(fit.weights)
    }

    "LR" should "fit with bias" in {
        val estimator = new LinearRegression().setLearnBias(true)
        val fit = estimator.fit(dataset)

        validate(fit.weights)
    }

    "LR" should "save and read" in {
        val regression = new LinearRegression()
        val pipeline = new Pipeline().setStages(Array(regression))

        val tempDir = Files.createTempDir()
        pipeline.write.overwrite().save(tempDir.getAbsolutePath)

        val readPipeline = Pipeline.load(tempDir.getAbsolutePath)
        val model = readPipeline.fit(dataset).stages(0)

        validate(model.asInstanceOf[LinearRegressionModel].weights)
    }
}
