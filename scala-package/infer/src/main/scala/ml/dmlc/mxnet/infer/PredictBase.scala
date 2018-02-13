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

import ml.dmlc.mxnet.io.NDArrayIter
import ml.dmlc.mxnet.{DataDesc, NDArray, Shape}
import ml.dmlc.mxnet.module.Module

import scala.collection.mutable.ListBuffer
import org.slf4j.LoggerFactory

/**
  * Base Trait for MXNNet Predictor classes.
  */
private[mxnet] trait PredictBase {

  /**
   * This method will take input as IndexedSeq one dimensional arrays and creates
   * NDArray needed for inference. The array will be reshaped based on the input descriptors.
   * @param input: A IndexedSequence of Scala one-dimensional array, An IndexedSequence is
   *             is needed when the model has more than one input/output
   * @return IndexedSequence array of outputs.
   */
  def predict(input: IndexedSeq[Array[Float]]): IndexedSeq[Array[Float]]

  /**
   * Predict using NDArray as input. This method is useful when the input is a batch of data
   * or when multiple operations on the input/output have to performed.
   * Note: User is responsible for managing allocation/deallocation of NDArrays.
   * @param input: IndexedSequence NDArrays.
   * @return output of Predictions as NDArrays.
   */
  def predictWithNDArray(input: IndexedSeq[NDArray]): IndexedSeq[NDArray]

}

/**
 * Implementation of predict routines.
 *
 * @param modelPathPrefix PathPrefix from where to load the model.
 *                        Example: file://model-dir/resnet-152(containing resnet-152-symbol.json,
 * @param inputDescriptors Descriptors defining the input node names, shape,
 *                         layout and Type parameters
 * @param outputDescriptors Descriptors defining the output node names, shape,
 *                          layout and Type parameters
 */
class Predictor(modelPathPrefix: String,
             protected val inputDescriptors: IndexedSeq[DataDesc],
             protected var outputDescriptors:
             Option[IndexedSeq[DataDesc]] = None) extends PredictBase {

  private val logger = LoggerFactory.getLogger(classOf[Predictor])

  protected var batchIndex = inputDescriptors(0).layout.indexOf('N')
  protected var batchSize = if (batchIndex != -1 ) inputDescriptors(0).shape(batchIndex) else 1

  protected var iDescriptors = inputDescriptors

  inputDescriptors.foreach((f: DataDesc) => require(f.layout.indexOf('N') == batchIndex,
    "batch size should be in the same index for all inputs"))

  if (batchIndex != -1) {
    inputDescriptors.foreach((f: DataDesc) => require(f.shape(batchIndex) == batchSize,
      "batch size should be same for all inputs"))
  } else {
    // TODO: this is assuming that the input needs a batch
    iDescriptors = inputDescriptors.map((f : DataDesc) => new DataDesc(f.name,
    Shape(1 +: f.shape.toVector), f.dtype, 'N' +: f.layout) )
    batchIndex = 1
  }

  protected val mxNetHandler = MXNetHandler()

  protected val mod = loadModule()

  /**
   * This method will take input as IndexedSeq one dimensional arrays and creates
   * NDArray needed for inference. The array will be reshaped based on the input descriptors.
   *
   * @param input : A IndexedSequence of Scala one-dimensional array, An IndexedSequence is
   *              is needed when the model has more than one input/output
   * @return IndexedSequence array of outputs.
   */
  override def predict(input: IndexedSeq[Array[Float]])
  : IndexedSeq[Array[Float]] = {

    require(input.length == inputDescriptors.length, "number of inputs provided: %d" +
      " does not match number of inputs in inputDescriptors: %d".format(input.length,
        inputDescriptors.length))

    for((i, d) <- input.zip(inputDescriptors)) {
      require (i.length == d.shape.product/batchSize, "number of elements:" +
        " %d in the input does not match the shape:%s".format( i.length, d.shape.toString()))
    }

    var inputND: ListBuffer[NDArray] = ListBuffer.empty[NDArray]

    for((i, d) <- input.zip(inputDescriptors)) {
      val shape = d.shape.toVector.patch(from = batchIndex, patch = Vector(1), replaced = 1)

      inputND += mxNetHandler.execute(NDArray.array(i, Shape(shape)))

    }

    // rebind with batchsize 1
    if (batchSize != 1) {
      val desc = iDescriptors.map((f : DataDesc) => new DataDesc(f.name,
        Shape(f.shape.toVector.patch(batchIndex, Vector(1), 1)), f.dtype, f.layout) )
      mxNetHandler.execute(mod.bind(desc, outputDescriptors, forceRebind = true,
        forTraining = false))
    }

    val resultND = mxNetHandler.execute(mod.predict(new NDArrayIter(inputND.toIndexedSeq)))

    val result = resultND.map((f : NDArray) => f.toArray)

    mxNetHandler.execute(inputND.foreach(_.dispose))
    mxNetHandler.execute(resultND.foreach(_.dispose))

    // rebind to batchSize
    if (batchSize != 1) {
      mxNetHandler.execute(mod.bind(inputDescriptors, forTraining = false, forceRebind = true))
    }

    result
  }

  /**
   * Predict using NDArray as input. This method is useful when the input is a batch of data
   * or when multiple operations on the input/output have to performed.
   * Note: User is responsible for managing allocation/deallocation of NDArrays.
   *
   * @param inputBatch : IndexedSequence NDArrays.
   * @return output of Predictions as NDArrays.
   */
  override def predictWithNDArray(inputBatch: IndexedSeq[NDArray]): IndexedSeq[NDArray] = {

    require(inputBatch.length == inputDescriptors.length, "number of inputs provided: %d" +
      " do not match number of inputs in inputDescriptors: %d".format(inputBatch.length,
        inputDescriptors.length))

    // Shape validation, remove this when backend throws better error messages.
    for((i, d) <- inputBatch.zip(iDescriptors)) {

      require(inputBatch(0).shape(batchIndex) == i.shape(batchIndex),
         "All inputs should be of same batch size")
      require(i.shape.drop(batchIndex) == d.shape.drop(batchIndex),
        "Input Data Shape: %s should match the inputDescriptor shape: %s except batchSize".format(
          i.shape.toString, d.shape.toString))
    }

    val inputBatchSize = inputBatch(0).shape(batchIndex)

    // rebind with the new batchSize
    if (batchSize != inputBatchSize) {

      val desc = iDescriptors.map((f : DataDesc) => new DataDesc(f.name,
        Shape(f.shape.toVector.patch(batchIndex, Vector(inputBatchSize), 1)), f.dtype, f.layout) )

      mxNetHandler.execute(mod.bind(desc, outputDescriptors, forceRebind = true,
        forTraining = false))
    }

    val resultND = mxNetHandler.execute(mod.predict(new NDArrayIter(inputBatch)))

    if (batchSize != inputBatchSize) {
      mxNetHandler.execute(mod.bind(iDescriptors, outputDescriptors, forceRebind = true,
        forTraining = false))
    }

    resultND
  }

  def loadModule(): Module = {
    val mod = mxNetHandler.execute(Module.loadCheckpoint(modelPathPrefix, 0))
    mxNetHandler.execute(mod.bind(inputDescriptors, forTraining = false))
    mod
  }
}