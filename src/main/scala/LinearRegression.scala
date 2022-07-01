package org.apache.spark.ml.kamenev

import breeze.linalg.functions.euclideanDistance
import breeze.linalg.{DenseVector, sum}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

import scala.util.control.Breaks.{break, breakable}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {
    def this() = this(Identifiable.randomUID("lr"))

    override def fit(dataset: Dataset[_]): LinearRegressionModel = {
        val constValue = if (isLearnBias) 1.0 else 0.0
        val biasCol = lit(constValue)

        val biasedDataset = dataset.withColumn("bias", biasCol)

        val vectorAssembler = new VectorAssembler().setInputCols(Array("bias", "x", "y")).setOutputCol("bias_x_y")
        implicit val encoder: Encoder[Vector] = ExpressionEncoder()
        val vectors = vectorAssembler.transform(biasedDataset).select("bias_x_y").as[Vector]

        val size = vectors.first().size - 1

        var prev = DenseVector.fill(size) {
            Double.PositiveInfinity
        }
        val curWeights = DenseVector.fill(size) {
            0.0
        }

        breakable {
            for (_ <- 0L until getMaxIteration) {
                if (euclideanDistance(curWeights.toDenseVector, prev.toDenseVector) <= getTolerance) {
                    break
                }

                val summary = vectors.rdd.mapPartitions(data => {
                    val summarizer = new MultivariateOnlineSummarizer()

                    data.foreach(row => {
                        val x = row.asBreeze(0 until size).toDenseVector

                        val yTrue = row.asBreeze(-1)
                        val yPred = x.dot(curWeights)
                        val diff = yTrue - yPred

                        val weightsDelta = -2.0 * x * diff

                        summarizer.add(mllib.linalg.Vectors.fromBreeze(weightsDelta))
                    })

                    Iterator(summarizer)
                }).reduce(_ merge _)

                prev = curWeights.copy
                curWeights -= getLearningRate * summary.mean.asBreeze
            }
        }

        copyValues(new LinearRegressionModel(curWeights).setParent(this))
    }

    override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

trait LinearRegressionParams extends HasInputCol with HasOutputCol {
    def setInputCol(value: String): this.type = set(inputCol, value)

    setDefault(inputCol -> "x")

    def setOutputCol(value: String): this.type = set(outputCol, value)

    setDefault(outputCol -> "y")

    val predictionCol: Param[String] = {
        new Param[String](parent = this, name = "predictionCol", doc = "Prediction column name")
    }

    def getPredictionCol: String = $(predictionCol)

    def setPredictionCol(value: String): this.type = set(predictionCol, value)

    setDefault(predictionCol -> "y_pred")

    val learningRate: DoubleParam = {
        new DoubleParam(parent = this, name = "learningRate", doc = "learning rate")
    }

    def getLearningRate: Double = $(learningRate)

    def setLearningRate(value: Double): this.type = set(learningRate, value)

    setDefault(learningRate -> 0.05)

    val maxIteration: LongParam = {
        new LongParam(parent = this, name = "maxIteration", doc = "max number of iterations")
    }

    def getMaxIteration: Long = $(maxIteration)

    def setMaxIteration(value: Long): this.type = set(maxIteration, value)

    setDefault(maxIteration -> 10000)

    val tolerance: DoubleParam = {
        new DoubleParam(parent = this, name = "tolerance", doc = "tolerance for weights change to converge")
    }

    def getTolerance: Double = $(tolerance)

    def setTolerance(value: Double): this.type = set(tolerance, value)

    setDefault(tolerance -> 1e-6)

    val learnBias: BooleanParam = {
        new BooleanParam(parent = this, name = "learn bias", doc = "whether to learn bias weight")
    }

    def isLearnBias: Boolean = $(learnBias)

    def setLearnBias(value: Boolean): this.type = set(learnBias, value)

    setDefault(learnBias -> false)

    protected def validateAndTransformSchema(schema: StructType): StructType = {
        SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

        if (schema.fieldNames.contains(getOutputCol)) {
            SchemaUtils.checkColumnType(schema, getOutputCol, DoubleType)

            schema
        } else {
            SchemaUtils.appendColumn(schema, schema(getOutputCol).copy(getOutputCol))
        }
    }
}

class LinearRegressionModel(override val uid: String, val weights: DenseVector[Double]) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {
    def this(weights: DenseVector[Double]) = this(Identifiable.randomUID("lrm"), weights)

    override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weights), extra)

    override def transform(dataset: Dataset[_]): DataFrame = {
        val transforming = {
            dataset.sqlContext.udf.register(s"${uid}_transforming",
                (v: Vector) => {
                    weights(0) + sum(v.asBreeze.toDenseVector * weights(1 until weights.length))
                }
            )
        }

        dataset.withColumn(getPredictionCol, transforming(dataset(getInputCol)))
    }

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

    override def write: MLWriter = new DefaultParamsWriter(this) {
        protected override def saveImpl(path: String): Unit = {
            super.saveImpl(path)

            val vectors = Tuple1(Vectors.fromBreeze(weights))
            sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
        }
    }
}
