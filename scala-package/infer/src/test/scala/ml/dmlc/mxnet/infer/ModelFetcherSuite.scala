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

import org.scalatest.mockito.MockitoSugar
import org.scalatest.{BeforeAndAfterAll, FunSuite, Ignore}

@Ignore
class ModelFetcherSuite extends FunSuite with BeforeAndAfterAll with MockitoSugar{

  private val modelDir = "model-files"

  val currentDirectory = new java.io.File(".").getCanonicalPath
  val modelPathPrefix = currentDirectory + File.pathSeparator + modelDir + File.pathSeparator +
    "test"

  val symbolFile = modelPathPrefix + "-symbol.json"
  val paramsFile = modelPathPrefix + "-0000.params"

  override def beforeAll() {
    val dir = new File(modelDir)
    dir.delete()
  }

  override def afterAll() {
    val dir = new File(modelDir)
    dir.delete()
  }

  test("test file fetching") {

    val dir = new File(modelDir)
    assert(dir.mkdir(), "failed model dir creation")

    val symbol = new File(symbolFile).createNewFile()
    val params = new File(paramsFile).createNewFile()

    for (i <- 0 until 5) {
      val synsetFile = "synset%d.txt".format(0)
      new File(currentDirectory + File.pathSeparator + synsetFile).createNewFile()
    }

    val fileFetcher = ModelFetcher(modelPathPrefix)

    assert(fileFetcher.symbolFilePath == symbolFile)
    assert(fileFetcher.paramsFilePath == paramsFile)
    assert(fileFetcher.synsetFilePaths.length == 5)

    for (i <- 0 until fileFetcher.synsetFilePaths.length) {
      val synsetFile = "synset%d.txt".format(0)
      assert(currentDirectory + File.pathSeparator + synsetFile == fileFetcher.synsetFilePaths(i))

      val dir = new File(modelDir)
      dir.delete()

    }
  }

}
