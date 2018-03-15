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

package ml.dmlc.mxnet.infer

import ml.dmlc.mxnet.{Context, DataDesc, NDArray}
import java.io.File

import org.slf4j.LoggerFactory

import scala.io
import scala.collection.mutable.ListBuffer

trait ClassifierBase {

  /**
    * Takes an Array of Floats and returns corresponding labels, score tuples.
    * @param input: IndexedSequence one-dimensional array of Floats.
    * @param topK: (Optional) How many top_k(sorting will be based on the last axis)
    *             elements to return, if not passed returns unsorted output.
    * @return IndexedSequence of (Label, Score) tuples.
    */
  def classify(input: IndexedSeq[Array[Float]],
               topK: Option[Int] = None): IndexedSeq[(String, Float)]

  /**
    * Takes a Sequence of NDArrays and returns Label, Score tuples.
    * @param input: Indexed Sequence of NDArrays
    * @param topK: (Optional) How many top_k(sorting will be based on the last axis)
    *             elements to return, if not passed returns unsorted output.
    * @return Traversable Sequence of (Label, Score) tuple, Score will be in the form of NDArray
    */
  def classifyWithNDArray(input: IndexedSeq[NDArray],
                          topK: Option[Int] = None): IndexedSeq[IndexedSeq[(String, Float)]]
}

/**
  * A class for classifier tasks
  * @param modelPathPrefix PathPrefix from where to load the symbol, parameters and synset.txt
  *                        Example: file://model-dir/resnet-152(containing resnet-152-symbol.json
  *                        file://model-dir/synset.txt
  * @param inputDescriptors Descriptors defining the input node names, shape,
  *                         layout and Type parameters
  * @param contexts Device Contexts on which you want to run Inference, defaults to CPU.
  * @param epoch Model epoch to load, defaults to 0.
  */
class Classifier(modelPathPrefix: String,
                 protected val inputDescriptors: IndexedSeq[DataDesc],
                 protected val contexts: Array[Context] = Context.cpu(),
                 protected val epoch: Option[Int] = Some(0))
  extends ClassifierBase {

  private val logger = LoggerFactory.getLogger(classOf[Classifier])

  protected[mxnet] val predictor: PredictBase = getPredictor()

  protected[mxnet] val synsetFilePath = getSynsetFilePath(modelPathPrefix)

  protected[mxnet] val synset = readSynsetFile(synsetFilePath)

  protected[mxnet] val handler = MXNetHandler()

  /**
    * Takes a flat arrays as input and returns a List of (Label, tuple)
    * @param input: IndexedSequence one-dimensional array of Floats.
    * @param topK: (Optional) How many top_k(sorting will be based on the last axis)
    *             elements to return, if not passed returns unsorted output.
    * @return IndexedSequence of (Label, Score) tuples.
    */
  override def classify(input: IndexedSeq[Array[Float]],
                        topK: Option[Int] = None): IndexedSeq[(String, Float)] = {

    // considering only the first output
    val predictResult = predictor.predict(input)(0)
    var result: IndexedSeq[(String, Float)] = IndexedSeq.empty

    if (topK.isDefined) {
      val sortedIndex = predictResult.zipWithIndex.sortBy(-_._1).map(_._2).take(topK.get)
      result = sortedIndex.map(i => (synset(i), predictResult(i))).toIndexedSeq
    } else {
      result = synset.zip(predictResult).toIndexedSeq
    }
    result
  }

  /**
    * Takes input as NDArrays, useful when you want to perform multiple operations on
    * the input Array or when you want to pass a batch of input.
    * @param input: Indexed Sequence of NDArrays
    * @param topK: (Optional) How many top_k(sorting will be based on the last axis)
    *             elements to return, if not passed returns unsorted output.
    * @return Traversable Sequence of (Label, Score) tuple, Score will be in the form of NDArray
    */
  override def classifyWithNDArray(input: IndexedSeq[NDArray], topK: Option[Int] = None)
  : IndexedSeq[IndexedSeq[(String, Float)]] = {

    // considering only the first output
    val predictResultND: NDArray = predictor.predictWithNDArray(input)(0)

    val predictResult: ListBuffer[Array[Float]] = ListBuffer[Array[Float]]()

    // iterating over the individual items(batch size is in axis 0)
    for (i <- 0 until predictResultND.shape(0)) {
      val r = predictResultND.at(i)
      predictResult += r.toArray
      r.dispose()
    }

    var result: ListBuffer[IndexedSeq[(String, Float)]] =
      ListBuffer.empty[IndexedSeq[(String, Float)]]

    if (topK.isDefined) {
      val sortedIndices = predictResult.map(r =>
        r.zipWithIndex.sortBy(-_._1).map(_._2).take(topK.get)
      )
      for (i <- sortedIndices.indices) {
        result += sortedIndices(i).map(sIndx =>
          (synset(sIndx), predictResult(i)(sIndx))).toIndexedSeq
      }
    } else {
      for (i <- predictResult.indices) {
        result += synset.zip(predictResult(i)).toIndexedSeq
      }
    }

    handler.execute(predictResultND.dispose())

    result.toIndexedSeq
  }

  def getSynsetFilePath(modelPathPrefix: String): String = {
    val dirPath = modelPathPrefix.substring(0, 1 + modelPathPrefix.lastIndexOf(File.separator))
    val d = new File(dirPath)
    require(d.exists && d.isDirectory, "directory: %s not found".format(dirPath))

    val s = new File(dirPath + "synset.txt")
    require(s.exists() && s.isFile, "File synset.txt should exist inside modelPath: %s".format
    (dirPath + "synset.txt"))

    s.getCanonicalPath
  }

  def readSynsetFile(synsetFilePath: String): List[String] = {
    val f = io.Source.fromFile(synsetFilePath)
    try {
      f.getLines().toList
    } finally {
      f.close
    }
  }

  def getPredictor(): PredictBase = {
      new Predictor(modelPathPrefix, inputDescriptors, contexts, epoch)
  }

}
