name := "MillionSong"

version := "1.0"

scalaVersion := "2.11.6"

resolvers += "Spark Packages Repo" at "https://dl.bintray.com/spark-packages/maven/"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.3.0",
  "org.apache.spark" %% "spark-sql" % "2.3.0",
  "org.apache.hadoop" % "hadoop-hdfs" % "2.9.0",
  "org.apache.spark" %% "spark-mllib" % "2.3.0"
)
