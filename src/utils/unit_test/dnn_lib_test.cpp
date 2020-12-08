////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

// MUST include this
#include <catch2/catch.hpp>

#include <lbann/base.hpp>
#include <lbann/utils/dnn_enums.hpp>
#include <lbann/utils/exception.hpp>
#include <lbann/utils/dnn_lib/helpers.hpp>

#include <lbann/utils/dnn_lib/convolution.hpp>
//#include <lbann/utils/dnn_lib/dropout.hpp>
//#include <lbann/utils/dnn_lib/local_response_normalization.hpp>
//#include <lbann/utils/dnn_lib/pooling.hpp>
#include <lbann/utils/dnn_lib/softmax.hpp>

using namespace lbann;

TEMPLATE_TEST_CASE("Computing convolution layers", "[dnn_lib]", float, double)
{
  SECTION("convolution forward")
  {
    int N = 8;
    int in_c = 1, in_h = 5, in_w = 5;
    int out_c = 1, out_h = 6, out_w = 6;
    int filt_k = 1, filt_c = 1, filt_h = 2, filt_w = 2;
    int pad_h = 1, pad_w = 1;
    int str_h = 1, str_w = 1;
    int dil_h = 1, dil_w = 1;

    const dnn_lib::ScalingParamType<TestType> alpha = 1.;
    dnn_lib::TensorDescriptor xDesc;
    xDesc.set(dnn_lib::get_data_type<TestType>(),
              { N, in_c, in_h, in_w });
    El::Matrix<TestType, El::Device::GPU> x(in_c * in_h * in_w, N);
    dnn_lib::FilterDescriptor wDesc;
    wDesc.set(dnn_lib::get_data_type<TestType>(),
              dnn_lib::DNN_TENSOR_NCHW,
              { filt_k, filt_c, filt_h, filt_w });
    El::Matrix<TestType, El::Device::GPU> w(filt_k * filt_c * filt_h * filt_w,
                                            N);
    dnn_lib::ConvolutionDescriptor convDesc;
    convDesc.set({ pad_h, pad_w },
                 { str_h, str_w },
                 { dil_h, dil_w },
                 dnn_lib::get_data_type<TestType>());
    fwd_conv_alg alg = fwd_conv_alg::GEMM;
    size_t workspace_size = (1 << 30) / sizeof(TestType);
    El::Matrix<TestType, El::Device::GPU> workSpace(workspace_size, 1);
    const dnn_lib::ScalingParamType<TestType> beta = 0.;
    dnn_lib::TensorDescriptor yDesc;
    yDesc.set(dnn_lib::get_data_type<TestType>(),
              { N, out_c, out_h, out_w });
    El::Matrix<TestType, El::Device::GPU> y(out_c * out_h * out_w, N);

    REQUIRE_NOTHROW(
      dnn_lib::convolution_forward(alpha,
                                   xDesc, x,
                                   wDesc, w,
                                   convDesc, alg,
                                   workSpace,
                                   beta,
                                   yDesc, y));
  }
}

TEMPLATE_TEST_CASE("Computing softmax layers", "[dnn_lib]", float, double)
{
  int N = 8, labels_n = 2;
  SECTION("softmax forward")
  {
    const dnn_lib::ScalingParamType<TestType> alpha = 1.;
    dnn_lib::TensorDescriptor xDesc;
    xDesc.set(dnn_lib::get_data_type<TestType>(), { N, 1, labels_n });
    El::Matrix<TestType, El::Device::GPU> x(labels_n, N);
    const dnn_lib::ScalingParamType<TestType> beta = 0.;
    dnn_lib::TensorDescriptor yDesc;
    yDesc.set(dnn_lib::get_data_type<TestType>(), { N, 1, labels_n });
    El::Matrix<TestType, El::Device::GPU> y(labels_n, N);
    softmax_mode mode = softmax_mode::CHANNEL;
    softmax_alg alg = softmax_alg::ACCURATE;

    REQUIRE_NOTHROW(
      dnn_lib::softmax_forward(alpha, xDesc, x, beta, yDesc, y, mode, alg));
  }

  SECTION("softmax backward")
  {
    const dnn_lib::ScalingParamType<TestType> alpha = 1.;
    dnn_lib::TensorDescriptor yDesc;
    yDesc.set(dnn_lib::get_data_type<TestType>(), { N, 1, labels_n });
    El::Matrix<TestType, El::Device::GPU> y(labels_n, N);
    dnn_lib::TensorDescriptor dyDesc;
    dyDesc.set(dnn_lib::get_data_type<TestType>(), { N, 1, labels_n });
    El::Matrix<TestType, El::Device::GPU> dy(labels_n, N);
    const dnn_lib::ScalingParamType<TestType> beta = 0.;
    dnn_lib::TensorDescriptor dxDesc;
    dxDesc.set(dnn_lib::get_data_type<TestType>(), { N, 1, labels_n });
    El::Matrix<TestType, El::Device::GPU> dx(labels_n, N);
    softmax_mode mode = softmax_mode::CHANNEL;
    softmax_alg alg = softmax_alg::ACCURATE;

    REQUIRE_NOTHROW(
      dnn_lib::softmax_backward(alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx, mode, alg));
  }
}
