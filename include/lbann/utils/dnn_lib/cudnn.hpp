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
#ifndef LBANN_UTILS_DNN_LIB_CUDNN_HPP_
#define LBANN_UTILS_DNN_LIB_CUDNN_HPP_

namespace lbann
{
#if defined LBANN_HAS_CUDNN

#include <cudnn.h>

namespace dnn_lib
{
using dnnHandle_t = cudnn::cudnnHandle_t;
using dnnDataType_t = cudnn::cudnnDataType_t;
using dnnTensorDescriptor_t = cudnn::cudnnTensorDescriptor_t;
using dnnFilterDescriptor_t = cudnn::cudnnFilterDescriptor_t;
using dnnTensorFormat_t = cudnn::cudnnTensorFormat_t;
using dnnDropoutDescriptor_t = cudnn::cudnnDropoutDescriptor_t;
using dnnRNNDescriptor_t = cudnn::cudnnRNNDescriptor_t;
using dnnRNNAlgo_t = cudnn::cudnnRNNAlgo_t;
using dnnRNNMode_t = cudnn::cudnnRNNMode_t;
using dnnRNNBiasMode_t = cudnn::cudnnRNNBiasMode_t;
using dnnDirectionMode_t = cudnn::cudnnDirectionMode_t;
using dnnRNNInputMode_t = cudnn::cudnnRNNInputMode_t;
using dnnMathType_t = cudnn::cudnnMathType_t;
using dnnRNNDataDescriptor_t = cudnn::cudnnRNNDataDescriptor_t;
using dnnRNNDataLayout_t = cudnn::cudnnRNNDataLayout_t;
using dnnConvolutionDescriptor_t = cudnn::cudnnConvolutionDescriptor_t;
using dnnConvolutionMode_t = cudnn::cudnnConvolutionMode_t;
using dnnActivationDescriptor_t = cudnn::cudnnActivationDescriptor_t;
using dnnActivationMode_t = cudnn::cudnnActivationMode_t;
using dnnNanPropagation_t = cudnn::cudnnNanPropagation_t;
using dnnPoolingDescriptor_t = cudnn::cudnnPoolingDescriptor_t;
using dnnPoolingMode_t = cudnn::cudnnPoolingMode_t;
using dnnLRNDescriptor_t = cudnn::cudnnLRNDescriptor_t;
using dnnConvolutionFwdAlgo_t = cudnn::cudnnConvolutionFwdAlgo_t;
using dnnConvolutionBwdDataAlgo_t = cudnn::cudnnConvolutionBwdDataAlgo_t;
using dnnConvolutionBwdFilterAlgo_t = cudnn::cudnnConvolutionBwdFilterAlgo_t;

}// namespace dnn_lib
#endif // defined LBANN_HAS_CUDNN
}// namespace lbann
#endif // LBANN_UTILS_DNN_LIB_CUDNN_HPP_
