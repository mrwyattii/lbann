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
#include <lbann/utils/dnn_lib/dropout.hpp>
//#include <lbann/utils/dnn_lib/local_response_normalization.hpp>
//#include <lbann/utils/dnn_lib/pooling.hpp>
#include <lbann/utils/dnn_lib/softmax.hpp>

using namespace lbann;

TEMPLATE_TEST_CASE("Add tensors", "[dnn_lib]", float, double)
{
  SECTION("add tensor")
  {
    int N = 2, c = 4, h = 8, w = 16;
    const dnn_lib::ScalingParamType<TestType> alpha = 1.;
    const dnn_lib::ScalingParamType<TestType> beta = 0.;

    dnn_lib::TensorDescriptor aDesc;
    aDesc.set(dnn_lib::get_data_type<TestType>(), { N, c, h, w });
    El::Matrix<TestType, El::Device::GPU> A(c * h * w, N);
    dnn_lib::TensorDescriptor cDesc;
    cDesc.set(dnn_lib::get_data_type<TestType>(), { N, c, h, w });
    El::Matrix<TestType, El::Device::GPU> C(c * h * w, N);

    REQUIRE_NOTHROW(dnn_lib::add_tensor(alpha, aDesc, A, beta, cDesc, C));
  }
}

TEMPLATE_TEST_CASE("Computing convolution layers", "[dnn_lib]", float, double)
{
    int N = 8;
    int in_c = 1, in_h = 5, in_w = 5;
    int out_c = 1, out_h = 6, out_w = 6;
    int filt_k = 1, filt_c = 1, filt_h = 2, filt_w = 2;
    int pad_h = 1, pad_w = 1;
    int str_h = 1, str_w = 1;
    int dil_h = 1, dil_w = 1;
    const dnn_lib::ScalingParamType<TestType> alpha = 1.;
    const dnn_lib::ScalingParamType<TestType> beta = 0.;

  SECTION("convolution forward")
  {
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

  SECTION("convolution backward data")
  {
    dnn_lib::FilterDescriptor wDesc;
    wDesc.set(dnn_lib::get_data_type<TestType>(),
              dnn_lib::DNN_TENSOR_NCHW,
              { filt_k, filt_c, filt_h, filt_w });
    El::Matrix<TestType, El::Device::GPU> w(filt_k * filt_c * filt_h * filt_w,
                                            N);
    dnn_lib::TensorDescriptor dyDesc;
    dyDesc.set(dnn_lib::get_data_type<TestType>(),
               { N, out_c, out_h, out_w });
    El::Matrix<TestType, El::Device::GPU> dy(out_c * out_h * out_w, N);
    dnn_lib::ConvolutionDescriptor convDesc;
    convDesc.set({ pad_h, pad_w },
                 { str_h, str_w },
                 { dil_h, dil_w },
                 dnn_lib::get_data_type<TestType>());
    bwd_data_conv_alg alg = bwd_data_conv_alg::CUDNN_ALGO_0;
    size_t workspace_size = (1 << 30) / sizeof(TestType);
    El::Matrix<TestType, El::Device::GPU> workSpace(workspace_size, 1);
    dnn_lib::TensorDescriptor dxDesc;
    dxDesc.set(dnn_lib::get_data_type<TestType>(),
               { N, in_c, in_h, in_w });
    El::Matrix<TestType, El::Device::GPU> dx(in_c * in_h * in_w, N);

    REQUIRE_NOTHROW(
      dnn_lib::convolution_backward_data(alpha,
                                         wDesc, w,
                                         dyDesc, dy,
                                         convDesc, alg,
                                         workSpace,
                                         beta,
                                         dxDesc, dx));
  }

  SECTION("convolution backward bias")
  {
    dnn_lib::TensorDescriptor dyDesc;
    dyDesc.set(dnn_lib::get_data_type<TestType>(),
               { N, out_c, out_h, out_w });
    El::Matrix<TestType, El::Device::GPU> dy(out_c * out_h * out_w, N);
    dnn_lib::TensorDescriptor dbDesc;
    dbDesc.set(dnn_lib::get_data_type<TestType>(),
               { 1, out_c, 1, 1 });
    El::Matrix<TestType, El::Device::GPU> db(out_c, 1);

    REQUIRE_NOTHROW(
      dnn_lib::convolution_backward_bias(alpha, dyDesc, dy,
                                         beta, dbDesc, db));
  }

  SECTION("convolution backward filter")
  {
    dnn_lib::TensorDescriptor xDesc;
    xDesc.set(dnn_lib::get_data_type<TestType>(),
              { N, in_c, in_h, in_w });
    El::Matrix<TestType, El::Device::GPU> x(in_c * in_h * in_w, N);
    dnn_lib::TensorDescriptor dyDesc;
    dyDesc.set(dnn_lib::get_data_type<TestType>(),
               { N, out_c, out_h, out_w });
    El::Matrix<TestType, El::Device::GPU> dy(out_c * out_h * out_w, N);
    dnn_lib::ConvolutionDescriptor convDesc;
    convDesc.set({ pad_h, pad_w },
                 { str_h, str_w },
                 { dil_h, dil_w },
                 dnn_lib::get_data_type<TestType>());
    bwd_filter_conv_alg alg = bwd_filter_conv_alg::CUDNN_ALGO_0;
    size_t workspace_size = (1 << 30) / sizeof(TestType);
    El::Matrix<TestType, El::Device::GPU> workSpace(workspace_size, 1);
    dnn_lib::FilterDescriptor dwDesc;
    dwDesc.set(dnn_lib::get_data_type<TestType>(),
               dnn_lib::DNN_TENSOR_NCHW,
               { filt_k, filt_c, filt_h, filt_w });
    El::Matrix<TestType, El::Device::GPU> dw(filt_k * filt_c * filt_h * filt_w,
                                             N);

    REQUIRE_NOTHROW(
      dnn_lib::convolution_backward_filter(alpha,
                                           xDesc, x,
                                           dyDesc, dy,
                                           convDesc, alg,
                                           workSpace,
                                           beta,
                                           dwDesc, dw));
  }
}

TEMPLATE_TEST_CASE("Computing dropout layers", "[dnn_lib]", float, double)
{
  int N = 8, c = 1, h = 5, w = 5;
  float dropout = 0.25;
  int seed = 1337;
  dnn_lib::DropoutDescriptor dropoutDesc;
  size_t states_size = dnn_lib::get_dropout_states_size() / sizeof(TestType);
  El::Matrix<TestType, El::Device::GPU> states(states_size, 1);
  dropoutDesc.set(dropout, states.Buffer(), states_size * sizeof(TestType), seed);

  SECTION("dropout forward")
  {
    dnn_lib::TensorDescriptor xDesc;
    xDesc.set(dnn_lib::get_data_type<TestType>(), { N, c, h, w });
    El::Matrix<TestType, El::Device::GPU> x(c * h * w, N);
    dnn_lib::TensorDescriptor yDesc;
    yDesc.set(dnn_lib::get_data_type<TestType>(), { N, c, h, w });
    El::Matrix<TestType, El::Device::GPU> y(c * h * w, N);
    size_t workspace_size = dnn_lib::get_dropout_reserve_space_size(xDesc);
    El::Matrix<TestType, El::Device::GPU> workSpace(workspace_size, 1);

    REQUIRE_NOTHROW(
      dnn_lib::dropout_forward(dropoutDesc,
                               xDesc, x,
                               yDesc, y,
                               workSpace));
  }
  SECTION("dropout backward")
  {
    dnn_lib::TensorDescriptor dxDesc;
    dxDesc.set(dnn_lib::get_data_type<TestType>(), { N, c, h, w });
    El::Matrix<TestType, El::Device::GPU> dx(c * h * w, N);
    dnn_lib::TensorDescriptor dyDesc;
    dyDesc.set(dnn_lib::get_data_type<TestType>(), { N, c, h, w });
    El::Matrix<TestType, El::Device::GPU> dy(c * h * w, N);
    size_t workspace_size = dnn_lib::get_dropout_reserve_space_size(dxDesc);
    El::Matrix<TestType, El::Device::GPU> workSpace(workspace_size, 1);

    REQUIRE_NOTHROW(
      dnn_lib::dropout_forward(dropoutDesc,
                               dyDesc, dy,
                               dxDesc, dx,
                               workSpace));
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
