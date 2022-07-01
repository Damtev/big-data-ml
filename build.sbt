name := "Big Data ML"

version := "0.1"

scalaVersion := "2.13.8"

idePackagePrefix := Some("org.apache.spark.ml.kamenev")

libraryDependencies ++= Seq(
    "org.scalanlp" %% "breeze" % "1.2",
    "org.apache.spark" %% "spark-core" % "3.2.0",
    "org.apache.spark" %% "spark-mllib" % "3.2.0",
    "org.scalatest" %% "scalatest" % "3.2.12" % "test"
)