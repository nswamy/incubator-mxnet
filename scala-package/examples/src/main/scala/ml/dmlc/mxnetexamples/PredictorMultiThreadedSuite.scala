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

package ml.dmlc.mxnetexamples

import java.util.concurrent._

import ml.dmlc.mxnet.infer.Predictor
import ml.dmlc.mxnet.{DataDesc, NDArray, Shape}
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.util.Random

class PredictorMultiThreadedSuite(val input: PredictorMultiThreadedSuite.Input) {

  val totalTimeoutInMillis = 600 * 1000
  val taskTimeOutInMillis = 1 * 1000

  val inputDescriptor: DataDesc = new DataDesc("data", Shape(1, 3, 224, 224))

  val predictor = new Predictor(input.modelPathPrefix, IndexedSeq(inputDescriptor))

  private val threadFactory = new ThreadFactory {

    override def newThread(r: Runnable): Thread = new Thread(r) {
      setName(classOf[PredictorMultiThreadedSuite].getCanonicalName
        + "-numThreads: %d".format(input.numProducerThreads))
    }
  }

  def generateSyntheticData(size: Int = 224*224*3): Array[Float] = {
    Array.fill[Float](size)(Random.nextFloat)
  }

  private val threadPool: ExecutorService =
    Executors.newFixedThreadPool(input.numProducerThreads, threadFactory)


  private val completion = new ExecutorCompletionService[Unit](threadPool)

  val task = new Callable[Unit] {
    override def call(): Unit = {
      val inputData = generateSyntheticData(inputDescriptor.shape.product)
      val inputND = NDArray.array(inputData, inputDescriptor.shape)
      predictor.predict(IndexedSeq(inputData))
      val result = predictor.predictWithNDArray(IndexedSeq(inputND))
      result.foreach(_.dispose())
      inputND.dispose()

    }
  }

  def runPredict(): Unit = {
    for (i <- 0 until input.numTimesToTest by 100) {
      val tasks: List[Callable[Unit]] = Array.fill[Callable[Unit]](100)(task).toList
      val result = threadPool.invokeAll(
        tasks.asJava, totalTimeoutInMillis,
        TimeUnit.MILLISECONDS).asScala
      result.foreach((_.get()))
      // scalastyle:off println
      println("done with %d iterations.".format(i))
      // scalastyle:on println
    }
  }
}

object PredictorMultiThreadedSuite {

  private val logger = LoggerFactory.getLogger(classOf[PredictorMultiThreadedSuite])

  def main(args: Array[String]): Unit = {

    val input = new Input

    val parser: CmdLineParser = new CmdLineParser(input)

    try {
      parser.parseArgument(args.toList.asJava)
      val test = new PredictorMultiThreadedSuite(input)
      test.runPredict()

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        // scalastyle:off println
        println(ex.getMessage)
        println(ex)
        // scalastyle:on println
        //        parser.printUsage(System.err)
        sys.exit(1)
      }
    }

  }

  class Input {
    @Option(name = "--resnet50-model-dir", usage = "the input model directory")
    val modelPathPrefix: String = "/resnet/resnet-50"
    @Option(name = "--num-times", usage = "number of times to test")
    val numTimesToTest: Int = 10000
    @Option(name = "--num-producer-threads", usage = "number of input producer threads")
    val numProducerThreads: Int = 10
  }
}

