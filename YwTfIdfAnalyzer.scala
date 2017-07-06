/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package com.yanzezheng.spark.test

// $example on$

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{SparseVector => SV, Vector}
import org.slf4j.LoggerFactory

import scala.collection.mutable

//import com.zte.bigdata.vmax.machinelearning.common.{LogSupport, CreateSparkContext}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.mllib.feature.{HashingTF => MllibHashingTF}
import org.apache.spark.sql.functions.{col, udf}

// $example off$
import org.apache.spark.sql.SparkSession

object YwTfIdfAnalyzer {

  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName("TfIdfExample")
     // .master("local[2]")
      .getOrCreate()
    val logger = LoggerFactory.getLogger("YwTfIdfAnalyzer")
    Logger.getRootLogger.setLevel(Level.WARN)
    // $example on$
       // 保证RDD可以转换为DataFrame
    import spark.implicits._
    // 因为要提取theme的关键词, 所以需要先做聚合,将相同文章放到一起
    // select concat_ws(" ".collect_list(content) group by theme) from table_tf_idf 应该也可以
   /* System.setProperty("spark.sql.warehouse.dir","/user/hive/warehouse");
    spark.sql("use nlp")
    val sentenceData = spark.sql(s"select cbid,content from nlp.t_ed_qqbook_nlp_bookcontent_sign where ds=20170628 limit 100").rdd
      .map(row => (row.getString(0), row.getString(1)))// 去掉英文标点
      .reduceByKey((x, y) => x + y) //聚合
      .toDF("cbid", "content") //转回DF*/
    val sentenceData = spark.sparkContext.textFile("/user/books/*02.txt")
      //.map(p => new String(p.getBytes, 0, p.length, "GBK"))
      .map(line => {
      var linarr =  line.split("\\|")
      if (linarr.size>3){
        //println(linarr(1))
        (linarr(0),linarr(2))
      }else{
        (" ","")
      }
    }
      )
      .map(arr => (arr._1,arr._2))
      .reduceByKey((x,y)=>x+y) //聚合
      .toDF("cbid", "content") //转回DF
    spark.udf.register(
      "trans_content", // 函数名称
      (row: String) => {
        //val list=AnaylyzerTools.anaylyzerWords(row.toString()) //分词处理
        //val str = AnaylyzerTools.listToStr(list)
        val str = AnaylyzerTools.jieBaAnalyze(row.toString())
        str
      }// 函数
    )
    sentenceData.registerTempTable("tmp_trans_content")
    //sentenceData.select("cbid","ccid","trans_content(content)")
    val sentenceData1 =  spark.sql("select cbid,trans_content(content) as contents  from  tmp_trans_content")
    //sentenceData1.show()
    val tokenizer = new Tokenizer().setInputCol("contents").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData1)
   // wordsData.show()

    val mllibHashingTF = new MllibHashingTF(1 << 18)
    val mapWords = wordsData.select("words").rdd.map(row => row.getAs[mutable.WrappedArray[String]](0)).flatMap(x => x).map(w => (mllibHashingTF.indexOf(w), w)).collect.toMap

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")//.setNumFeatures(1000)
    val featurizedData = hashingTF.transform(wordsData)
    //featurizedData.select("words").show(false)
    //featurizedData.select("rawFeatures").show(false)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
   // var dd = rescaledData.select("cbid","ccid","words", "features").orderBy("features").limit()
   // rescaledData.select("features", "cbid").take(3).foreach(println)
    val takeTopN = udf { (v:Vector) =>
      (v.toSparse.indices zip v.toSparse.values)
        .sortBy(-_._2) //负值就是从大到小
        .take(200)
        .map(x => mapWords.getOrElse(x._1, "null") + ":" + f"${x._2}%.3f".toString) // 冒号分隔单词和值,值取小数后三位
        .mkString(";") } // 词语和值的对以;隔开(别用逗号,会与hive表格式的TERMINATED BY ','冲突)

    rescaledData.select(col("cbid"), takeTopN(col("features")).as("features")).write.json("/user/books/nlpdata3")//registerTempTable("tfidf_result_table")//.write.format("csv").save("hdfs://spark04:8020/user/books/nlpdata2")

    //.show(false)//.registerTempTable("tfidf_result_table").show(false)
    // 保存数据框到文件
    //rescaledData.select("gender", "age", "education").write.format("csv").save("hdfs://ns1/datafile/wangxiao/data123.csv")

  /* spark.sql( s"""drop table if exists nlp.t_ed_qqbook_nlp_bookcontent_result""")
    spark.sql(
      s"""create table IF NOT EXISTS nlp.t_ed_qqbook_nlp_bookcontent_result (
      cbid  String,
      features   String
      )
      ROW FORMAT DELIMITED FIELDS TERMINATED BY ','""")
    val sql = s"insert overwrite table nlp.t_ed_qqbook_nlp_bookcontent_result select cbid,features from tfidf_result_table"
    try {
      spark.sql(sql)
    } catch {
      case e: Exception => logger.error("""load data to table error""")
        throw e
    }*/
    spark.stop()
  }
}

// scalastyle:on println
