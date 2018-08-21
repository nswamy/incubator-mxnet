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

import java.util.concurrent.{Executors, ScheduledExecutorService, ThreadFactory, TimeUnit}

import org.slf4j.{Logger, LoggerFactory}

import scala.util.Try

object GCManager {
  private val logger = LoggerFactory.getLogger(classOf[GCManager])
  private val gcCallFrequencyProperty = "mxnet.gc.callFrequency.InSeconds"
  private val defaultGCCallFrequencyInSeconds = 5

  private lazy val gcFrequency =
    Try(System.getProperty(GCManager.gcCallFrequencyProperty).toInt).getOrElse(
      GCManager.defaultGCCallFrequencyInSeconds
    )

  private val scheduledExecutor: ScheduledExecutorService =
    Executors.newSingleThreadScheduledExecutor(new ThreadFactory {
    override def newThread(r: Runnable): Thread = new Thread(r) {
      setName(classOf[GCManager].getCanonicalName)
      setDaemon(true)
    }
  })

  scheduledExecutor.scheduleAtFixedRate(new Runnable {
      override def run(): Unit = {
        logger.info("Calling System.gc")
        System.gc()
        NDPhantomRef.cleanup
        SymPhantomRef.cleanup
        ExecPhantomRef.cleanup
      }
    },
    GCManager.gcFrequency,
    GCManager.gcFrequency,
    TimeUnit.SECONDS
  )
}

trait GCManager {
}

