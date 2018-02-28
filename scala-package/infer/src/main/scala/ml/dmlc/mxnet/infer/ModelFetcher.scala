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

import java.io.File

import org.slf4j.LoggerFactory

/**
  * Fetch model files from source
  */
trait ModelFetcher {

  var symbolFilePath: String
  var paramsFilePath: String
  var synsetFilePaths: IndexedSeq[String]

  val symbolFile : File
  val paramsFile : File
  val synsetFiles : IndexedSeq[File]

}

/**
  * Factory that creates an appropriate ModelFetcher based on the model prefix
  */
object ModelFetcher {

  def apply(modelPathPrefix: String, epoch: Option[Int] = Some[Int](0)): ModelFetcher = {
    /**
      * Prefix could start with file://, ~/, or /User/myName/
      */
    // scalastyle:off println
    println(modelPathPrefix)
    // scalastyle:on println

    if (modelPathPrefix.startsWith("file://") || modelPathPrefix.startsWith(File.pathSeparator) ) {
      new OnDiskModelFetcher(modelPathPrefix.replace("file://", File.pathSeparator), epoch)
    }
    else if (modelPathPrefix.startsWith("~")) {
      new OnDiskModelFetcher(modelPathPrefix.replace("~", System.getProperty("user.home")), epoch)
    }
    else {
      throw new IllegalAccessException("Undefined source")
    }
  }

}

class OnDiskModelFetcher(protected val modelPathPrefix: String,
                         protected val epoch: Option[Int] = Some[Int](0))
  extends ModelFetcher {

  private val synsetPattern = """(.*)(/synset[0-9]?.txt$)""".r

  private val logger = LoggerFactory.getLogger(classOf[ModelFetcher])

  override var symbolFilePath = s"$modelPathPrefix-symbol.json"
  override var paramsFilePath = s"%s-%4d.params".format(modelPathPrefix, epoch.get)
  override var synsetFilePaths: IndexedSeq[String] = getSynsetFilePaths

  val (s  , p, syn) = fetchModelFiles
  override val symbolFile = s
  override val paramsFile = p
  override val synsetFiles = syn

  protected def fetchModelFiles(): (File, File, IndexedSeq[File]) = {

    val symbolFile = new File(symbolFilePath)
    val paramsFile = new File(paramsFilePath)
    var synsetFiles = IndexedSeq.empty[File]

    require(symbolFile.exists(), "Symbol file does not exist: %s".
      format(symbolFilePath))
    require(paramsFile.exists(), "Params file does not exist: %s".
      format(paramsFilePath))

    if (synsetFilePaths.isEmpty) {
      logger.info("Label(synset) files were not found")
    }
    else {
      synsetFiles = synsetFilePaths.map(new File(_)).toIndexedSeq
    }

    (symbolFile, paramsFile, synsetFiles)

  }

  private def getSynsetFilePaths(): IndexedSeq[String] = {

    val dirPath = modelPathPrefix.substring(0, 1 + modelPathPrefix.lastIndexOf(File.pathSeparator))
    val d = new File(dirPath)
    require(d.exists && d.isDirectory, "directory: %s not found".format(dirPath))

    d.listFiles.filter(_.isFile).map(_.getCanonicalPath).
      flatMap(synsetPattern.unapplySeq(_)).sortWith(_(1) < _(1)).map(_.mkString)

  }

}



