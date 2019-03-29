/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "nquadratic_op-inl.h"

namespace mxnet {
  namespace op {
    DMLC_REGISTER_PARAMETER(NQuadraticOpParam);

    NNVM_REGISTER_OP(nquadratic)
      .describe(R"code(nswamy version of quadratic function.)code" ADD_FILELINE)
      .set_attr_parser(ParamParser<NQuadraticOpParam>)
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<nnvm::FListInputNames>("FListInputNames",
                                       [](const NodeAttrs &attrs) {
                                         return std::vector<std::string>{"data"};
                                       })
      .set_attr<mxnet::FInferShape>("FInferShape", NQuadraticOpShape)
      .set_attr<nnvm::FInferType>("FInferType", NQuadraticOpType)
      .set_attr<mxnet::FCompute>("FCompute<cpu>", NQuadraticOpForward < cpu > )
      .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_nbackward_quadratic"})
      .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                      [](const NodeAttrs &attrs) {
                                        return std::vector<std::pair<int, int> >{{0, 0}};
                                      })
      .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
      .add_arguments(NQuadraticOpParam::__FIELDS__());

    NNVM_REGISTER_OP(_nbackward_quadratic)
      .set_attr_parser(ParamParser<NQuadraticOpParam>)
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<nnvm::TIsBackward>("TIsBackward", true)
      .set_attr<FCompute>("FCompute<cpu>", NQuadraticOpBackward < cpu > );
  }
}