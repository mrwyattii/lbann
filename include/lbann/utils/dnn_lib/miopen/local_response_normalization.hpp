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
#ifndef LBANN_UTILS_DNN_LIB_MIOPEN_LRN_HPP_
#define LBANN_UTILS_DNN_LIB_MIOPEN_LRN_HPP_

#include "lbann/utils/ml_enums.hpp"
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "utils.hpp"

namespace lbann
{

#ifdef LBANN_HAS_MIOPEN
namespace dnn_lib
{

using namespace miopen;

template <typename TensorDataType, typename ScalarParameterType>
void lrn_cross_channel_forward(LRNDescriptor const& normDesc,
                               ScalarParameterType const& alpha_in,
                               TensorDescriptor const& xDesc,
                               El::AbstractMatrix<TensorDataType> const& x,
                               ScalarParameterType const& beta_in,
                               TensorDescriptor const& yDesc,
                               El::AbstractMatrix<TensorDataType>& y,
                               El::SyncInfo<El::Device::GPU> const& si,
                               El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
                               lrn_mode mode = lrn_mode::CROSS_CHANNEL_DIM1)
{

  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  if (workspace.Height() == 0 || workspace.Width() == 0) { // Training use-case
    CHECK_MIOPEN(miopenLRNForward(handle_manager.get(),
                                  normDesc,
                                  //miopen::to_miopen(mode),
                                  &alpha,
                                  xDesc,
                                  x.LockedBuffer(),
                                  &beta,
                                  yDesc,
                                  y.Buffer(),
                                  true,
                                  workSpace.Buffer()));
  }
  else {                                                  // Inference use-case
    CHECK_MIOPEN(miopenLRNForward(handle_manager.get(),
                                  normDesc,
                                  //miopen::to_miopen(mode),
                                  &alpha,
                                  xDesc,
                                  x.LockedBuffer(),
                                  &beta,
                                  yDesc,
                                  y.Buffer(),
                                  false,
                                  nullptr));
  }
}

template <typename TensorDataType, typename ScalarParameterType>
void lrn_cross_channel_forward(LRNDescriptor const& normDesc,
                               ScalarParameterType const& alpha_in,
                               TensorDescriptor const& xDesc,
                               El::AbstractMatrix<TensorDataType> const& x,
                               ScalarParameterType const& beta_in,
                               TensorDescriptor const& yDesc,
                               El::AbstractMatrix<TensorDataType>& y,
                               El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
                               lrn_mode mode = lrn_mode::CROSS_CHANNEL_DIM1)
{

  auto multisync = El::MakeMultiSync(gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(y),
                                     gpu::get_sync_info(x));
  lrn_cross_channel_forward(normDesc,
                            alpha_in, xDesc, x,
                            beta_in, yDesc, y,
                            workSpace,
                            multisync, mode);
}

template <typename TensorDataType, typename ScalarParameterType>
void lrn_cross_channel_backward(LRNDescriptor const& normDesc,
                                ScalarParameterType const& alpha_in,
                                TensorDescriptor const& yDesc,
                                El::AbstractMatrix<TensorDataType> const& y,
                                TensorDescriptor const& dyDesc,
                                El::AbstractMatrix<TensorDataType> const& dy,
                                TensorDescriptor const& xDesc,
                                El::AbstractMatrix<TensorDataType> const& x,
                                ScalarParameterType const& beta_in,
                                TensorDescriptor const& dxDesc,
                                El::AbstractMatrix<TensorDataType>& dx,
                                El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
                                El::SyncInfo<El::Device::GPU> const& si,
                                lrn_mode mode = lrn_mode::CROSS_CHANNEL_DIM1)
{

  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  if (workspace.Height() == 0 || workspace.Width() == 0) { // Training use-case
    CHECK_MIOPEN(miopenLRNBackward(handle_manager.get(),
                                   normDesc,
                                   //miopen::to_miopen(mode),
                                   &alpha,
                                   yDesc,
                                   y.LockedBuffer(),
                                   dyDesc,
                                   dy.LockedBuffer(),
                                   xDesc,
                                   x.LockedBuffer(),
                                   &beta,
                                   dxDesc,
                                   dx.Buffer(),
                                   workSpace.Buffer()));
  }
  else {                                                  // Inference use-case
    CHECK_MIOPEN(miopenLRNBackward(handle_manager.get(),
                                   normDesc,
                                   //miopen::to_miopen(mode),
                                   &alpha,
                                   yDesc,
                                   y.LockedBuffer(),
                                   dyDesc,
                                   dy.LockedBuffer(),
                                   xDesc,
                                   x.LockedBuffer(),
                                   &beta,
                                   dxDesc,
                                   dx.Buffer(),
                                   nullptr));
  }
}

template <typename TensorDataType, typename ScalarParameterType>
void lrn_cross_channel_backward(LRNDescriptor const& normDesc,
                                ScalarParameterType const& alpha_in,
                                TensorDescriptor const& yDesc,
                                El::AbstractMatrix<TensorDataType> const& y,
                                TensorDescriptor const& dyDesc,
                                El::AbstractMatrix<TensorDataType> const& dy,
                                TensorDescriptor const& xDesc,
                                El::AbstractMatrix<TensorDataType> const& x,
                                ScalarParameterType const& beta_in,
                                TensorDescriptor const& dxDesc,
                                El::AbstractMatrix<TensorDataType>& dx,
                                El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
                                lrn_mode mode = lrn_mode::CROSS_CHANNEL_DIM1)
{

  auto multisync = El::MakeMultiSync(gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(dx),
                                     gpu::get_sync_info(x),
                                     gpu::get_sync_info(dy),
                                     gpu::get_sync_info(y));
  lrn_cross_channel_backward(normDesc,
                             alpha_in, yDesc, y, dyDesc, dy,
                             xDesc, x, beta_in, dxDesc, dx,
                             workSpace,
                             multisync, mode);
}

}// namespace dnn_lib
#endif // LBANN_HAS_MIOPEN
}// namespace lbann
#endif // LBANN_UTILS_DNN_LIB_MIOPEN_LRN_HPP_