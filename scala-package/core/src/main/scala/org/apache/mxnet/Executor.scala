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

package org.apache.mxnet

import java.lang.ref.{PhantomReference, ReferenceQueue}
import java.util.concurrent.ConcurrentHashMap

import org.apache.mxnet.Base._
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer

object Executor {
  // Get the dictionary given name and ndarray pairs.
  private[mxnet] def getDict(names: Seq[String],
                             ndarrays: Seq[NDArray]): Map[String, NDArray] = {
    require(names.toSet.size == names.length, "Duplicate names detected")
    (names zip ndarrays).toMap
  }
}

class ExecPhantomRef(referent: Executor, val execHandle: ExecutorHandle)
  extends PhantomReference[Executor](referent, ExecPhantomRef.execRefQueue) {
}

object ExecPhantomRef {
  // using ConcurrentHashMap since ConcurrentHashSet is unavailable
  // this is just holding the SymPhantomRef, so it does not get garbage collected
  private val execPhantomRefs = new ConcurrentHashMap[ExecPhantomRef, ExecutorHandle]()
  private val execRefQueue = new ReferenceQueue[Executor]
  private val logger = LoggerFactory.getLogger(classOf[ExecPhantomRef])

  def register(e: Executor, execHandle: ExecutorHandle) : Unit = {
    execPhantomRefs.put(new ExecPhantomRef(e, execHandle), ndHandle)
  }

  def cleanup: Unit = {
    var ref = execRefQueue.poll().asInstanceOf[ExecPhantomRef]
    while (ref != null) {
      _LIB.mxExecutorFree(ref.execHandle)
      execPhantomRefs.remove(ref)
      ref = execRefQueue.poll().asInstanceOf[ExecPhantomRef]
    }
  }
}
/**
 * Symbolic Executor component of MXNet <br />
 * <b>
 * WARNING: it is your responsibility to clear this object through dispose().
 * </b>
 *
 * @author Yizhi Liu
 *
 * Constructor: please use Symbol.bind and Symbol.simpleBind instead.
 * @param handle ExecutorHandle generated by calling Bind
 * @param symbol
 * @see Symbol.bind : to create executor
 */
class Executor private[mxnet](private[mxnet] val handle: ExecutorHandle,
                              private[mxnet] val symbol: Symbol) extends WarnIfNotDisposed {
  private[mxnet] var argArrays: Array[NDArray] = null
  private[mxnet] var gradArrays: Array[NDArray] = null
  private[mxnet] var auxArrays: Array[NDArray] = null
  val outputs: Array[NDArray] = getOutputs
  protected var _argDict: Map[String, NDArray] = null
  protected var _gradDict: Map[String, NDArray] = null
  protected var _auxDict: Map[String, NDArray] = null
  protected var monitorCallback: MXMonitorCallback = null
  private[mxnet] var _ctx: Context = null
  private[mxnet] var _gradsReq: Iterable[_] = null
  private[mxnet] var _group2ctx: Map[String, Context] = null
  private val logger: Logger = LoggerFactory.getLogger(classOf[Executor])

  private var disposed = false
  protected def isDisposed = disposed

  ExecPhantomRef.register(this, handle)
  def dispose(): Unit = {
//    if (!disposed) {
//      outputs.foreach(_.dispose())
//      _LIB.mxExecutorFree(handle)
//      disposed = true
//    }
  }

  /**
   * Return a new executor with the same symbol and shared memory,
   * but different input/output shapes.
   * For runtime reshaping, variable length sequences, etc.
   * The returned executor shares state with the current one,
   * and cannot be used in parallel with it.
   * @param partialShaping Whether to allow changing the shape of unspecified arguments.
   * @param allowUpSizing Whether to allow allocating new ndarrays that's larger than the original.
   * @param kwargs Map of string to Shape.
   *                - new shape for arguments.
   * @return
   * executor A new executor that shares memory with this.
   */
  def reshape(partialShaping: Boolean = false, allowUpSizing: Boolean = false,
    kwargs: Map[String, Shape]): Executor = {
     val (argShapes, _, auxShapes) = this.symbol.inferShape(kwargs)
     require(argShapes != null, "Insufficient argument shapes provided.")

    var newArgDict = Map[String, NDArray]()
    var newGradDict = Map[String, NDArray]()

    this.symbol.listArguments().zipWithIndex.foreach { case (name, i) =>
      val newShape = argShapes(i)
      val arr = this.argArrays(i)
      val dArr = if (this.gradArrays == null) null else this.gradArrays(i)
      if (partialShaping || kwargs.contains(name) || newShape.equals(arr.shape)) {
        if (newShape.product > arr.shape.product) {
          require(allowUpSizing, s"New shape of arg:$name larger than original. " +
                        "First making a big executor and then down sizing it " +
                        "is more efficient than the reverse." +
                        "If you really want to up size, set allowUpSizing = true " +
                        "to enable allocation of new arrays.")
          newArgDict = newArgDict + (name -> NDArray.empty(newShape, arr.context))
          if (dArr != null) {
            newGradDict = newGradDict + (name -> NDArray.empty(newShape, dArr.context))
          }
        } else {
          newArgDict = newArgDict + (name -> arr.reshape(newShape.toArray))
          if (dArr != null) {
            newGradDict = newGradDict + (name -> dArr.reshape(newShape.toArray))
          }
        }
      } else {
        throw new  AssertionError(s"Shape of unspecified array arg:$name changed." +
                    "This can cause the new executor to not share parameters " +
                    "with the old one. Please check for error in network." +
                    "If this is intended, set partialShaping = true to suppress this warning.")
      }
    }

    var newAuxDict = Map[String, NDArray]()
    val zip3 = (this.symbol.listAuxiliaryStates(), auxShapes, this.auxArrays).zipped
    zip3.foreach { case (name, newShape, arr) =>
      if (partialShaping || newShape.equals(arr.shape)) {
        if (newShape.product > arr.shape.product) {
          require(allowUpSizing, s"New shape of aux:$name larger than original. " +
                        "First making a big executor and then down sizing it " +
                        "is more efficient than the reverse." +
                        "If you really want to up size, set allowUpSizing = true " +
                        "to enable allocation of new arrays.")
          newAuxDict = newAuxDict + (name -> NDArray.empty(newShape, arr.context))
        } else {
          newAuxDict = newAuxDict + (name -> arr.reshape(newShape.toArray))
        }
      } else {
        throw new  AssertionError(s"Shape of unspecified array aux:$name changed." +
                  "This can cause the new executor to not share parameters " +
                  "with the old one. Please check for error in network." +
                  "If this is intended, set partialShaping = true to suppress this warning.")
      }
    }
    if (this._gradsReq.isInstanceOf[Seq[_]]) {
      this.symbol.bind(this._ctx,
                          newArgDict,
                          newGradDict,
                          this._gradsReq.asInstanceOf[Seq[String]],
                          newAuxDict,
                          this._group2ctx,
                          this)
    } else {
      this.symbol.bind(this._ctx,
                          newArgDict,
                          newGradDict,
                          this._gradsReq.asInstanceOf[Map[String, String]],
                          newAuxDict,
                          this._group2ctx,
                          this)
    }
  }

  /**
   * list all the output ndarray
   * @return A list of ndarray binded to the heads of executor.
   */
  private def getOutputs: Array[NDArray] = {
    val ndHandles = ArrayBuffer[NDArrayHandle]()
    checkCall(_LIB.mxExecutorOutputs(handle, ndHandles))
    ndHandles.toArray.map(new NDArray(_, addToCollector = false))
  }

  /**
   * Calculate the outputs specified by the binded symbol.
   * @param isTrain whether this forward is for evaluation purpose.
   * @param kwargs Additional specification of input arguments.
   */
  def forward(isTrain: Boolean, kwargs: (String, NDArray)*): Unit = {
    kwargs.foreach { case (name, array) =>
      require(argDict.contains(name), s"Unknown argument $name")
      array.copyTo(argDict(name))
    }
    checkCall(_LIB.mxExecutorForward(handle, if (isTrain) 1 else 0))
  }

  def forward(): Unit = {
    forward(isTrain = false)
  }

  /**
   * Do backward pass to get the gradient of arguments.
   * @param outGrads Gradient on the outputs to be propagated back.
   *                 This parameter is only needed when bind is called
   *                 on outputs that are not a loss function.
   */
  def backward(outGrads: Array[NDArray]): Unit = {
    require(outGrads != null)
    val ndArrayPtrs = outGrads.map(_.handle)
    checkCall(_LIB.mxExecutorBackward(handle, ndArrayPtrs))
  }

  def backward(outGrad: NDArray): Unit = {
    require(outGrad != null)
    backward(Array(outGrad))
  }

  def backward(): Unit = {
    backward(Array.empty[NDArray])
  }

  /**
   * Install callback.
   * @param callback Takes a string and an NDArrayHandle.
   */
  def setMonitorCallback(callback: MXMonitorCallback): Unit = {
    monitorCallback = callback
    checkCall(_LIB.mxExecutorSetMonitorCallback(handle, monitorCallback))
  }

  /**
   * Get dictionary representation of argument arrrays.
   * @return The dictionary that maps name of arguments to NDArrays.
   * @throws IllegalArgumentException if there are duplicated names in the arguments.
   */
  def argDict: Map[String, NDArray] = {
    if (_argDict == null) {
      _argDict = Executor.getDict(symbol.listArguments(), argArrays)
    }
    _argDict
  }

  /**
   * Get dictionary representation of gradient arrays.
   * @return The dictionary that maps name of arguments to gradient arrays.
   * @throws IllegalArgumentException if there are duplicated names in the grads.
   */
  def gradDict: Map[String, NDArray] = {
    if (_gradDict == null) {
      _gradDict = Executor.getDict(symbol.listArguments(), gradArrays)
    }
    _gradDict
  }

  /**
   * Get dictionary representation of auxiliary states arrays.
   * @return The dictionary that maps name of auxiliary states to NDArrays.
   * @throws IllegalArgumentException if there are duplicated names in the auxiliary states.
   */
  def auxDict: Map[String, NDArray] = {
    if (_auxDict == null) {
      _auxDict = Executor.getDict(symbol.listAuxiliaryStates(), auxArrays)
    }
    _auxDict
  }

  /**
   * Copy parameters from arg_params, aux_params into executor's internal array.
   * @param argParams : dict of name to NDArray of arguments
   * @param auxParams : dict of name to NDArray of auxiliary states.
   * @param allowExtraParams
   *        Whether allow extra parameters that are not needed by symbol
   *        If this is True, no error will be thrown when arg_params or aux_params
   *        contain extra parameters that is not needed by the executor.
   * @throws IllegalArgumentException
   *         If there is additional parameters in the dict but allow_extra_params=False
   */
  def copyParamsFrom(argParams: Map[String, NDArray],
                     auxParams: Map[String, NDArray],
                     allowExtraParams: Boolean = false): Unit = {
    argParams.foreach { case (name, array) =>
      if (argDict.contains(name)) {
        array.copyTo(argDict(name))
      } else {
        require(allowExtraParams, s"Find name $name that is not in the arguments")
      }
    }
    if (auxParams != null) {
      auxParams.foreach { case (name, array) =>
        if (auxDict.contains(name)) {
          array.copyTo(auxDict(name))
        } else {
          require(allowExtraParams, s"Find name $name that is not in the auxiliary states")
        }
      }
    }
  }

  def copyParamsFrom(argParams: Map[String, NDArray], allowExtraParams: Boolean): Unit = {
    copyParamsFrom(argParams, null, allowExtraParams)
  }

  def copyParamsFrom(argParams: Map[String, NDArray]): Unit = {
    copyParamsFrom(argParams, allowExtraParams = false)
  }

  /**
   * Get a debug string about internal execution plan.
   * @return Debug string of the executor.
   */
  def debugStr: String = {
    val str = new RefString
    checkCall(_LIB.mxExecutorPrint(handle, str))
    str.value
  }
}
