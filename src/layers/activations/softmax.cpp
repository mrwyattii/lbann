////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/activations/softmax.hpp"
#ifdef LBANN_HAS_CUDNN
#include "lbann/utils/cublas_wrapper.hpp"
#endif  // LBANN_HAS_CUDNN

namespace lbann {

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarMRMat<El::Device::CPU>(grid);
}

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarVCMat<El::Device::CPU>(grid);
}

#ifdef LBANN_HAS_GPU
template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarMRMat<El::Device::GPU>(grid);
}

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
  ::setup_matrices(const El::Grid& grid) {
  activation_layer::setup_matrices(grid);
  if (m_workspace != nullptr) { delete m_workspace; }
  m_workspace = new StarVCMat<El::Device::GPU>(grid);
}
#endif // LBANN_HAS_GPU

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {
  fp_compute_cpu();
}

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {
  bp_compute_cpu();
}

#ifdef LBANN_HAS_GPU
template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  // Local matrices.
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  auto& local_workspace = m_workspace->Matrix();

  const El::Int local_height = local_input.Height();
  const El::Int local_width = local_input.Width();

  // Find the maximum entry in each local column.
  softmax_cuda::max_local_col_entry(
    local_height, local_width, local_input.LockedBuffer(),
    local_input.LDim(), local_workspace.Buffer(), El::GPUManager::Stream());
  // Find the global max entry in each column.
  m_comm->allreduce(*m_workspace, m_workspace->RedundantComm(), El::mpi::MAX,
                    std::type_index(typeid(Al::nccl_backend)));

  // Exponentiate activations and compute column sums.
  // This subtracts by the column max for stability.
  softmax_cuda::exp_and_col_sum(
    local_height, local_width, local_input.LockedBuffer(),
    local_input.LDim(), local_output.Buffer(), local_output.LDim(),
    local_workspace.Buffer(), El::GPUManager::Stream());
  // Compute the global sums for each column.
  m_comm->allreduce(*m_workspace, m_workspace->RedundantComm(), El::mpi::SUM,
                    std::type_index(typeid(Al::nccl_backend)));

  // Divide activations by the column sums.
  // This rounds small values to avoid denormalization.
  softmax_cuda::div_by_col_sums_and_cutoff(
    local_height, local_width, local_output.Buffer(),
    local_output.LDim(), local_workspace.Buffer(), m_min_output,
    El::GPUManager::Stream());
#endif  // LBANN_HAS_CUDNN
}

template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  // Local matrices.
  const auto& local_output = get_local_activations();
  const auto& local_grad_wrt_output = get_local_prev_error_signals();
  auto& local_grad_wrt_input = get_local_error_signals();
  auto& local_workspace = m_workspace->Matrix();

  const El::Int local_height = local_output.Height();
  const El::Int local_width = local_output.Width();

  // Compute dot products between output and gradient w.r.t. output.
  auto&& handle = El::GPUManager::cuBLASHandle();
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  for (El::Int col = 0; col < local_width; ++col) {
    cublas::dot(handle, local_height,
                local_output.LockedBuffer(0, col), 1,
                local_grad_wrt_output.LockedBuffer(0, col), 1,
                local_workspace.Buffer(0, col));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  m_comm->allreduce(*m_workspace, m_workspace->RedundantComm(), El::mpi::SUM,
                    std::type_index(typeid(Al::nccl_backend)));

  // Compute gradient w.r.t. input.
  // Applies a cutoff if needed to avoid denormalized floats.
  softmax_cuda::grad_wrt_input_and_cutoff(
    local_height, local_width, local_output.LockedBuffer(),
    local_output.LDim(), local_workspace.LockedBuffer(),
    local_grad_wrt_output.LockedBuffer(), local_grad_wrt_output.LDim(),
    local_grad_wrt_input.Buffer(), local_grad_wrt_input.LDim(),
    m_min_output, El::GPUManager::Stream());
#endif  // LBANN_HAS_CUDNN
}
#endif // LBANN_HAS_GPU

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  fp_compute_cpu();
}

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  bp_compute_cpu();
}

#ifdef LBANN_HAS_GPU
template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Matrices
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();

  // Apply softmax on the GPU
  CHECK_CUDNN(cudnnSoftmaxForward(this->m_cudnn->get_handle(),
                                  CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_INSTANCE,
                                  &one,
                                  this->m_prev_activations_cudnn_desc,
                                  local_input.LockedBuffer(),
                                  &zero,
                                  this->m_activations_cudnn_desc,
                                  local_output.Buffer()));

#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
  // Round to minimum value to avoid denormalized floats
  softmax_cuda::fp_cutoff(local_output.Height(),
                          local_output.Width(),
                          local_output.Buffer(),
                          local_output.LDim(),
                          m_min_output,
                          El::GPUManager::Stream());
#endif // LBANN_ENABLE_SOFTMAX_CUTOFF

#endif // LBANN_HAS_CUDNN
}

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;

  // Matrices
  const auto& local_output = get_local_activations();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();

  // Apply softmax on each GPU
  CHECK_CUDNN(cudnnSoftmaxBackward(this->m_cudnn->get_handle(),
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_INSTANCE,
                                   &one,
                                   this->m_activations_cudnn_desc,
                                   local_output.LockedBuffer(),
                                   this->m_prev_error_signals_cudnn_desc,
                                   local_gradient_wrt_output.LockedBuffer(),
                                   &one,
                                   this->m_error_signals_cudnn_desc,
                                   local_gradient_wrt_input.Buffer()));

#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
  // Round to minimum value to avoid denormalized floats
  softmax_cuda::bp_cutoff(local_output.Height(),
                          local_output.Width(),
                          local_output.LockedBuffer(),
                          local_output.LDim(),
                          local_gradient_wrt_input.Buffer(),
                          local_gradient_wrt_input.LDim(),
                          m_min_output,
                          El::GPUManager::Stream());
#endif // LBANN_ENABLE_SOFTMAX_CUTOFF

#endif // LBANN_HAS_CUDNN
}
#endif // LBANN_HAS_GPU

} // namespace lbann
