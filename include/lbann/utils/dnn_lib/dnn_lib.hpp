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

#ifndef LBANN_UTILS_CUDNN_HPP
#define LBANN_UTILS_CUDNN_HPP

#include "lbann/base.hpp"
#include "lbann/utils/gpu/helpers.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include <vector>

#ifdef LBANN_HAS_CUDNN

#include <cudnn.h>

// Error utility macros
#define CHECK_CUDNN_NODEBUG(cudnn_call)                         \
  do {                                                          \
    const cudnnStatus_t status_CHECK_CUDNN = (cudnn_call);      \
    if (status_CHECK_CUDNN != CUDNN_STATUS_SUCCESS) {           \
      cudaDeviceReset();                                        \
      LBANN_ERROR(std::string("cuDNN error (")                  \
                  + cudnnGetErrorString(status_CHECK_CUDNN)     \
                  + std::string(")"));                          \
    }                                                           \
  } while (0)
#define CHECK_CUDNN_DEBUG(cudnn_call)                           \
  do {                                                          \
    LBANN_CUDA_CHECK_LAST_ERROR(true);                          \
    CHECK_CUDNN_NODEBUG(cudnn_call);                            \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_CUDNN(cudnn_call) CHECK_CUDNN_DEBUG(cudnn_call)
#else
#define CHECK_CUDNN(cudnn_call) CHECK_CUDNN_NODEBUG(cudnn_call)
#endif // #ifdef LBANN_DEBUG

#define CHECK_CUDNN_DTOR(cudnn_call)            \
  try {                                         \
    CHECK_CUDNN(cudnn_call);                                            \
  }                                                                     \
  catch (std::exception const& e) {                                     \
    std::cerr << "Caught exception:\n\n    what(): "                    \
              << e.what() << "\n\nCalling std::terminate() now."        \
              <<  std::endl;                                            \
    std::terminate();                                                   \
  }                                                                     \
  catch (...) {                                                         \
    std::cerr << "Caught something that isn't an std::exception.\n\n"   \
              << "Calling std::terminate() now." << std::endl;          \
    std::terminate();                                                   \
  }


namespace lbann {

// Forward declaration
class Layer;

namespace cudnn {

using dnnHandle_t = cudnnHandle_t;
using dnnDataType_t = cudnnDataType_t;
using dnnTensorDescriptor_t = cudnnTensorDescriptor_t;
using dnnFilterDescriptor_t = cudnnFilterDescriptor_t;
using dnnTensorFormat_t = cudnnTensorFormat_t;
using dnnDropoutDescriptor_t = cudnnDropoutDescriptor_t;
using dnnRNNDescriptor_t = cudnnRNNDescriptor_t;
using dnnRNNAlgo_t = cudnnRNNAlgo_t;
using dnnRNNMode_t = cudnnRNNMode_t;
using dnnRNNBiasMode_t = cudnnRNNBiasMode_t;
using dnnDirectionMode_t = cudnnDirectionMode_t;
using dnnRNNInputMode_t = cudnnRNNInputMode_t;
using dnnMathType_t = cudnnMathType_t;
using dnnRNNDataDescriptor_t = cudnnRNNDataDescriptor_t;
using dnnRNNDataLayout_t = cudnnRNNDataLayout_t;
using dnnConvolutionDescriptor_t = cudnnConvolutionDescriptor_t;
using dnnConvolutionMode_t = cudnnConvolutionMode_t;
using dnnActivationDescriptor_t = cudnnActivationDescriptor_t;
using dnnActivationMode_t = cudnnActivationMode_t;
using dnnNanPropagation_t = cudnnNanPropagation_t;
using dnnPoolingDescriptor_t = cudnnPoolingDescriptor_t;
using dnnPoolingMode_t = cudnnPoolingMode_t;
using dnnLRNDescriptor_t = cudnnLRNDescriptor_t;
using dnnConvolutionFwdAlgo_t = cudnnConvolutionFwdAlgo_t;
using dnnConvolutionBwdDataAlgo_t = cudnnConvolutionBwdDataAlgo_t;
using dnnConvolutionBwdFilterAlgo_t = cudnnConvolutionBwdFilterAlgo_t;

template <typename T>
struct ScalingParameterT
{
  using type = T;
};

template <typename T>
using ScalingParamType = typename ScalingParameterT<T>::type;

#ifdef LBANN_HAS_GPU_FP16
template <>
struct ScalingParameterT<fp16>
{
  using type = float;
};
#endif // LBANN_USE_GPU_FP16

////////////////////////////////////////////////////////////
// Global cuDNN objects
////////////////////////////////////////////////////////////

/** Initialize global cuDNN objects. */
void initialize();
/** Destroy global cuDNN objects. */
void destroy();
/** Get cuDNN handle.
 *  This resets the active CUDA device and stream to the Hydrogen
 *  defaults. The cuDNN handle is initialized if needed.
 */
dnnHandle_t& get_handle();

////////////////////////////////////////////////////////////
// Helper functions for cuDNN types
////////////////////////////////////////////////////////////

/** Get cuDNN data type associated with DataType. */
template <typename TensorDataType>
dnnDataType_t get_data_type();

////////////////////////////////////////////////////////////
// Wrapper classes for cuDNN types
////////////////////////////////////////////////////////////

template <typename T>
using BackendHandleType = typename T::handle_type;

/** @brief Wrapper around @c cudnnTensorDescriptor_t */
class TensorDescriptor {
public:

  using handle_type = dnnTensorDescriptor_t;

public:

  explicit TensorDescriptor(dnnTensorDescriptor_t desc=nullptr);

  ~TensorDescriptor();

  // Copy-and-swap idiom
  TensorDescriptor(const TensorDescriptor&);
  TensorDescriptor(TensorDescriptor&&);
  TensorDescriptor& operator=(TensorDescriptor);
  friend void swap(TensorDescriptor& first, TensorDescriptor& second);

  /** @brief Take ownership of cuDNN object */
  void reset(dnnTensorDescriptor_t desc=nullptr);
  /** @brief Return cuDNN object and release ownership */
  dnnTensorDescriptor_t release() noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  dnnTensorDescriptor_t get() const noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  operator dnnTensorDescriptor_t() const noexcept;

  /** @brief Create cuDNN object
   *
   *  Does nothing if already created.
   */
  void create();
  /** @brief Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  void set(
    dnnDataType_t data_type,
    std::vector<int> dims,
    std::vector<int> strides = {});
  /** @brief Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  template <typename... IntTs>
  void set(
    dnnDataType_t data_type,
    IntTs... dims) {
    set(data_type, {static_cast<int>(dims)...});
  }

private:

  dnnTensorDescriptor_t desc_ = nullptr;

};

/** @brief Wrapper around @c cudnnFilterDescriptor_t */
class FilterDescriptor {
public:

  using handle_type = dnnFilterDescriptor_t;

public:

  explicit FilterDescriptor(dnnFilterDescriptor_t desc=nullptr);

  ~FilterDescriptor();

  // Copy-and-swap idiom
  FilterDescriptor(const FilterDescriptor&);
  FilterDescriptor(FilterDescriptor&&);
  FilterDescriptor& operator=(FilterDescriptor);
  friend void swap(FilterDescriptor& first, FilterDescriptor& second);

  /** @brief Take ownership of cuDNN object */
  void reset(dnnFilterDescriptor_t desc=nullptr);
  /** @brief Return cuDNN object and release ownership */
  dnnFilterDescriptor_t release() noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  dnnFilterDescriptor_t get() const noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  operator dnnFilterDescriptor_t() const noexcept;

  /** Create cuDNN object
   *
   *  Does nothing if already created.
   */
  void create();
  /** Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  void set(
    dnnDataType_t data_type,
    dnnTensorFormat_t format,
    const std::vector<int>& dims);
  /** Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  template <typename... IntTs>
  void set(
    dnnDataType_t data_type,
    dnnTensorFormat_t format,
    IntTs... dims) {
    set(data_type, format, {static_cast<int>(dims)...});
  }

private:

  dnnFilterDescriptor_t desc_ = nullptr;

};

/** @brief Wrapper around @c cudnnDropoutDescriptor_t */
class DropoutDescriptor {

public:

  explicit DropoutDescriptor(dnnDropoutDescriptor_t desc=nullptr);
  DropoutDescriptor(float dropout,
                    void* states,
                    size_t states_size,
                    unsigned long long seed)
  {
    this->set(dropout, states, states_size, seed);
  }

  ~DropoutDescriptor();

  // Copy-and-swap idiom
  DropoutDescriptor(const DropoutDescriptor&);
  DropoutDescriptor(DropoutDescriptor&&);
  DropoutDescriptor& operator=(DropoutDescriptor);
  friend void swap(DropoutDescriptor& first, DropoutDescriptor& second);

  /** @brief Take ownership of cuDNN object */
  void reset(dnnDropoutDescriptor_t desc=nullptr);
  /** @brief Return cuDNN object and release ownership */
  dnnDropoutDescriptor_t release() noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  dnnDropoutDescriptor_t get() const noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  operator dnnDropoutDescriptor_t() const noexcept;

  /** Create cuDNN object
   *
   *  Does nothing if already created.
   */
  void create();
  /** Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  void set(
    float dropout,
    void* states,
    size_t states_size,
    unsigned long long seed);

private:

  dnnDropoutDescriptor_t desc_ = nullptr;

};

/** @brief Wrapper around @c cudnnRNNDescriptor_t */
class RNNDescriptor {

public:

  explicit RNNDescriptor(dnnRNNDescriptor_t desc=nullptr);

  RNNDescriptor(const RNNDescriptor&) = delete;
  ~RNNDescriptor();

  // Copy-and-swap idiom
  RNNDescriptor(RNNDescriptor&&);
  RNNDescriptor& operator=(RNNDescriptor);
  friend void swap(RNNDescriptor& first, RNNDescriptor& second);

  /** @brief Take ownership of cuDNN object */
  void reset(dnnRNNDescriptor_t desc=nullptr);
  /** @brief Return cuDNN object and release ownership */
  dnnRNNDescriptor_t release() noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  dnnRNNDescriptor_t get() const noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  operator dnnRNNDescriptor_t() const noexcept;

  /** Create cuDNN object
   *
   *  Does nothing if already created.
   */
  void create();
  /** Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  void set(
    dnnRNNAlgo_t algorithm,
    dnnRNNMode_t cell_mode,
    dnnRNNBiasMode_t bias_mode,
    dnnDirectionMode_t direction_mode,
    dnnRNNInputMode_t input_mode,
    dnnDataType_t data_type,
    dnnDataType_t math_precision,
    dnnMathType_t math_type,
    size_t input_size,
    size_t hidden_size,
    size_t proj_size,
    size_t num_layers,
    dnnDropoutDescriptor_t dropout_desc,
    uint32_t aux_flags);

private:

  dnnRNNDescriptor_t desc_ = nullptr;

};

/** Wrapper around @c cudnnRNNDataDescriptor_t */
class RNNDataDescriptor {

public:

  RNNDataDescriptor(dnnRNNDataDescriptor_t desc=nullptr);

  ~RNNDataDescriptor();

  // Copy-and-swap idiom
  RNNDataDescriptor(const RNNDataDescriptor&) = delete; /// @todo Implement
  RNNDataDescriptor(RNNDataDescriptor&&);
  RNNDataDescriptor& operator=(RNNDataDescriptor);
  friend void swap(RNNDataDescriptor& first, RNNDataDescriptor& second);

  /** @brief Take ownership of cuDNN object */
  void reset(dnnRNNDataDescriptor_t desc=nullptr);
  /** @brief Return cuDNN object and release ownership */
  dnnRNNDataDescriptor_t release();
  /** @brief Return cuDNN object without releasing ownership */
  dnnRNNDataDescriptor_t get() const noexcept;
  /** @brief Return cuDNN object without releasing ownership */
  operator dnnRNNDataDescriptor_t() const noexcept;

  /** @brief Allocate a new handle.
   *
   *  Does nothing if already created.
   */
  void create();

  /** Configure cuDNN object
   *
   *  Creates cuDNN object if needed.
   */
  void set(
    dnnDataType_t data_type,
    dnnRNNDataLayout_t layout,
    size_t max_seq_length,
    size_t batch_size,
    size_t vector_size,
    const int seq_length_array[],
    void* padding_fill);

private:

  dnnRNNDataDescriptor_t desc_{nullptr};

};

/** @brief Wrapper around @c cudnnConvolutionDescriptor_t */
class ConvolutionDescriptor
{
public:

  /** @brief Descriptor handle from the implementation. */
  using DescriptorHandle_t = dnnConvolutionDescriptor_t;

public:

  /** @name Constructors and destructor */
  ///@{

  /** @brief Construct from an existing handle. */
  explicit ConvolutionDescriptor(DescriptorHandle_t desc=nullptr);

  /** @brief Any handle resources will be freed. */
  ~ConvolutionDescriptor();

  /** @brief Copy constructor.
   *
   *  Constructs a new handle with identical features.
   */
  ConvolutionDescriptor(const ConvolutionDescriptor&);
  /** @brief Move constructor */
  ConvolutionDescriptor(ConvolutionDescriptor&&);

  /** @brief Assignment operator. */
  ConvolutionDescriptor& operator=(ConvolutionDescriptor);

  ///@}
  /** @name Accessors */
  ///@{

  /** @brief Return handle object and release ownership */
  DescriptorHandle_t release() noexcept;
  /** @brief Return handle object without releasing ownership */
  DescriptorHandle_t get() const noexcept;
  /** @brief Implicit conversion to handle object without releasing
   *         ownership
   */
  operator DescriptorHandle_t() const noexcept;

  ///@}
  /** @name Modifiers */
  ///@{

  /** @brief Swap contents with another descriptor */
  void swap(ConvolutionDescriptor& other);

  /** @brief Take ownership of existing handle */
  void reset(DescriptorHandle_t desc=nullptr);

  /** @brief Allocate a new handle.
   *
   *  Does nothing if already created.
   */
  void create();

  /** @brief Configure handle properties
   *
   *  Allocates a new handle if one doesn't already exist.
   */
  void set(
    std::vector<int> const& pad,
    std::vector<int> const& stride,
    std::vector<int> const& dilation,
    dnnDataType_t data_type,
    dnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION);
  void set(
    size_t array_dim,
    int const pad[],
    int const stride[],
    int const dilation[],
    dnnDataType_t data_type,
    dnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION);

  /** @brief Set the math mode for this descriptor. */
  void set_math_mode(dnnMathType_t math_type);

  /** @brief Set the group count for this descriptor. */
  void set_group_count(int num_groups);

  ///@}

private:

  DescriptorHandle_t desc_ = nullptr;

};

/** @brief Swap two convolution descriptors. */
void swap(ConvolutionDescriptor& lhs, ConvolutionDescriptor& rhs);

/** @brief Wrapper around @c cudnnActivationDescriptor_t */
class ActivationDescriptor
{
public:

  /** @brief Descriptor handle from the implementation. */
  using DescriptorHandle_t = dnnActivationDescriptor_t;

public:

  /** @name Constructors and destructor */
  ///@{

  /** @brief Construct from an existing handle. */
  explicit ActivationDescriptor(DescriptorHandle_t desc=nullptr);

  /** @brief Any handle resources will be freed. */
  ~ActivationDescriptor();

  /** @brief Copy constructor.
   *
   *  Constructs a new handle with identical features.
   */
  ActivationDescriptor(const ActivationDescriptor&);
  /** @brief Move constructor */
  ActivationDescriptor(ActivationDescriptor&&);

  /** @brief Assignment operator. */
  ActivationDescriptor& operator=(ActivationDescriptor);

  ///@}
  /** @name Accessors */
  ///@{

  /** @brief Return handle object and release ownership */
  DescriptorHandle_t release() noexcept;
  /** @brief Return handle object without releasing ownership */
  DescriptorHandle_t get() const noexcept;
  /** @brief Implicit conversion to handle object without releasing
   *         ownership
   */
  operator DescriptorHandle_t() const noexcept;

  ///@}
  /** @name Modifiers */
  ///@{

  /** @brief Swap contents with another descriptor */
  void swap(ActivationDescriptor& other);

  /** @brief Take ownership of existing handle */
  void reset(DescriptorHandle_t desc=nullptr);

  /** @brief Allocate a new handle.
   *
   *  Does nothing if already created.
   */
  void create();
  /** @brief Configure handle properties
   *
   *  Allocates a new handle if one doesn't already exist.
   */
  void set(
    dnnActivationMode_t mode,
    dnnNanPropagation_t nan_prop,
    double coeff);

  ///@}

private:

  DescriptorHandle_t desc_ = nullptr;

};

/** @brief Swap two convolution descriptors. */
void swap(ActivationDescriptor& lhs, ActivationDescriptor& rhs);

/** @brief Wrapper around @c cudnnPoolingDescriptor_t */
class PoolingDescriptor
{
public:

  /** @brief Descriptor handle from the implementation. */
  using DescriptorHandle_t = dnnPoolingDescriptor_t;

public:

  /** @name Constructors and destructor */
  ///@{

  /** @brief Construct from an existing handle. */
  explicit PoolingDescriptor(DescriptorHandle_t desc=nullptr);

  /** @brief Any handle resources will be freed. */
  ~PoolingDescriptor();

  /** @brief Copy constructor.
   *
   *  Constructs a new handle with identical features.
   */
  PoolingDescriptor(const PoolingDescriptor&);
  /** @brief Move constructor */
  PoolingDescriptor(PoolingDescriptor&&);

  /** @brief Assignment operator. */
  PoolingDescriptor& operator=(PoolingDescriptor);

  ///@}
  /** @name Accessors */
  ///@{

  /** @brief Return handle object and release ownership */
  DescriptorHandle_t release() noexcept;
  /** @brief Return handle object without releasing ownership */
  DescriptorHandle_t get() const noexcept;
  /** @brief Implicit conversion to handle object without releasing
   *         ownership
   */
  operator DescriptorHandle_t() const noexcept;

  ///@}
  /** @name Modifiers */
  ///@{

  /** @brief Swap contents with another descriptor */
  void swap(PoolingDescriptor& other);

  /** @brief Take ownership of existing handle */
  void reset(DescriptorHandle_t desc=nullptr);

  /** @brief Allocate a new handle.
   *
   *  Does nothing if already created.
   */
  void create();
  /** @brief Configure handle properties
   *
   *  Allocates a new handle if one doesn't already exist.
   */
  void set(
    dnnPoolingMode_t mode,
    dnnNanPropagation_t maxpoolingNanOpt,
    std::vector<int> const& window_dims,
    std::vector<int> const& padding,
    std::vector<int> const& stride);
  void set(
    dnnPoolingMode_t mode,
    dnnNanPropagation_t nan_prop,
    int num_dims,
    int const window_dims[],
    int const padding[],
    int const stride[]);

  ///@}

private:

  DescriptorHandle_t desc_ = nullptr;

};

/** @brief Swap two convolution descriptors. */
void swap(PoolingDescriptor& lhs, PoolingDescriptor& rhs);

/** @brief Wrapper around @c cudnnLRNDescriptor_t */
class LRNDescriptor
{
public:

  /** @brief Descriptor handle from the implementation. */
  using DescriptorHandle_t = dnnLRNDescriptor_t;

public:

  /** @name Constructors and destructor */
  ///@{

  /** @brief Construct from an existing handle. */
  explicit LRNDescriptor(DescriptorHandle_t desc=nullptr);

  /** @brief Any handle resources will be freed. */
  ~LRNDescriptor();

  /** @brief Copy constructor.
   *
   *  Constructs a new handle with identical features.
   */
  LRNDescriptor(const LRNDescriptor&);
  /** @brief Move constructor */
  LRNDescriptor(LRNDescriptor&&);

  /** @brief Assignment operator. */
  LRNDescriptor& operator=(LRNDescriptor);

  ///@}
  /** @name Accessors */
  ///@{

  /** @brief Return handle object and release ownership */
  DescriptorHandle_t release() noexcept;
  /** @brief Return handle object without releasing ownership */
  DescriptorHandle_t get() const noexcept;
  /** @brief Implicit conversion to handle object without releasing
   *         ownership
   */
  operator DescriptorHandle_t() const noexcept;

  ///@}
  /** @name Modifiers */
  ///@{

  /** @brief Swap contents with another descriptor */
  void swap(LRNDescriptor& other);

  /** @brief Take ownership of existing handle */
  void reset(DescriptorHandle_t desc=nullptr);

  /** @brief Allocate a new handle.
   *
   *  Does nothing if already created.
   */
  void create();
  /** @brief Configure handle properties
   *
   *  Allocates a new handle if one doesn't already exist.
   */
  void set(unsigned n, double alpha, double beta, double k);

  ///@}

private:

  DescriptorHandle_t desc_ = nullptr;

};

/** @brief Swap two convolution descriptors. */
void swap(LRNDescriptor& lhs, LRNDescriptor& rhs);

////////////////////////////////////////////////////////////
// cuDNN tensor managers
////////////////////////////////////////////////////////////

/** Manager for a layer's cuDNN tensor descriptors. */
template <typename TensorDataType>
class layer_tensor_manager {
public:
  using LayerType = data_type_layer<TensorDataType>;
public:
  layer_tensor_manager(const LayerType* l = nullptr);
  virtual ~layer_tensor_manager() = default;

  /** Get the layer being managed. */
  const LayerType* get_layer() const { return m_layer; }
  /** Set the layer being managed. */
  void set_layer(const LayerType* l);

  /** Get cuDNN tensor descriptor for layer input. */
  virtual TensorDescriptor& get_prev_activations(int parent_index = 0) = 0;
  /** Get cuDNN tensor descriptor for layer output. */
  virtual TensorDescriptor& get_activations(int child_index = 0) = 0;
  /** Get cuDNN tensor descriptor for gradient w.r.t. layer output. */
  virtual TensorDescriptor& get_prev_error_signals(int child_index = 0) = 0;
  /** Get cuDNN tensor descriptor for gradient w.r.t. layer input. */
  virtual TensorDescriptor& get_error_signals(int parent_index = 0) = 0;

protected:
  layer_tensor_manager(const layer_tensor_manager&) = default;
  layer_tensor_manager& operator=(const layer_tensor_manager&) = default;
  layer_tensor_manager(layer_tensor_manager&&) = default;
  layer_tensor_manager& operator=(layer_tensor_manager&&) = default;

  /** Set number of tensor descriptors corresponding to layer inputs. */
  void set_num_parents(int num_parents);
  /** Set number of tensor descriptors corresponding to layer outputs. */
  void set_num_children(int num_children);

  /** Layer being managed. */
  const LayerType* m_layer;
  /** cuDNN tensor descriptors for layer inputs. */
  std::vector<TensorDescriptor> m_prev_activations;
  /** cuDNN tensor descriptors for layer outputs. */
  std::vector<TensorDescriptor> m_activations;
  /** cuDNN tensor descriptors for gradients w.r.t. layer outputs. */
  std::vector<TensorDescriptor> m_prev_error_signals;
  /** cuDNN tensor descriptors for gradients w.r.t. layer inputs. */
  std::vector<TensorDescriptor> m_error_signals;

};

/** Manager for a data-parallel layer's cuDNN tensor descriptors. */
template <typename TensorDataType>
class data_parallel_layer_tensor_manager
  : public layer_tensor_manager<TensorDataType> {
public:
  using LayerType = data_type_layer<TensorDataType>;
public:
  data_parallel_layer_tensor_manager(const LayerType* l = nullptr);
  data_parallel_layer_tensor_manager(
    const data_parallel_layer_tensor_manager&) = default;
  data_parallel_layer_tensor_manager&
    operator=(const data_parallel_layer_tensor_manager&) = default;
  data_parallel_layer_tensor_manager(
    data_parallel_layer_tensor_manager&&) = default;
  data_parallel_layer_tensor_manager&
    operator=(data_parallel_layer_tensor_manager&&) = default;
  ~data_parallel_layer_tensor_manager() = default;
  TensorDescriptor& get_prev_activations(int parent_index = 0) override;
  TensorDescriptor& get_activations(int child_index = 0) override;
  TensorDescriptor& get_prev_error_signals(int child_index = 0) override;
  TensorDescriptor& get_error_signals(int parent_index = 0) override;
};

/** Manager for an entry-wise layer's cuDNN tensor descriptors. */
template <typename TensorDataType>
class entrywise_layer_tensor_manager
  : public layer_tensor_manager<TensorDataType> {
public:
  using LayerType = data_type_layer<TensorDataType>;
public:
  entrywise_layer_tensor_manager(const LayerType* l = nullptr);
  entrywise_layer_tensor_manager(
    const entrywise_layer_tensor_manager&) = default;
  entrywise_layer_tensor_manager&
    operator=(const entrywise_layer_tensor_manager&) = default;
  entrywise_layer_tensor_manager(
    entrywise_layer_tensor_manager&&) = default;
  entrywise_layer_tensor_manager&
    operator=(entrywise_layer_tensor_manager&&) = default;
  ~entrywise_layer_tensor_manager() = default;
  TensorDescriptor& get_prev_activations(int parent_index = 0) override;
  TensorDescriptor& get_activations(int child_index = 0) override;
  TensorDescriptor& get_prev_error_signals(int child_index = 0) override;
  TensorDescriptor& get_error_signals(int parent_index = 0) override;
};

////////////////////////////////////////////////////////////
// cuDNN algorithm selection
////////////////////////////////////////////////////////////

/**
 * Select a forward convolution algorithm.
 *
 * If autotuning, memory for cuDNN algorithm runs is needed and should be
 * provided via the pointer arguments.
 *
 * @param autotune True to attempt all cuDNN algorithms and select the fastest.
 * @param deterministic True to require deterministic algorithms.
 */
dnnConvolutionFwdAlgo_t get_fwd_algorithm(
  bool autotune,
  bool deterministic,
  const TensorDescriptor& input_desc,
  const void* input,
  const FilterDescriptor& kernel_desc,
  const void* kernel,
  const ConvolutionDescriptor& conv_desc,
  const TensorDescriptor& output_desc,
  void* output,
  size_t ws_size,
  void* ws);

/** Select a backward data convolution algorithm.
 *
 * If autotuning, memory for cuDNN algorithm runs is needed and should be
 * provided via the pointer arguments.
 *
 * @param autotune True to attempt all cuDNN algorithms and select the fastest.
 * @param deterministic True to require deterministic algorithms.
 */
dnnConvolutionBwdDataAlgo_t get_bwd_data_algorithm(
  bool autotune,
  bool deterministic,
  const FilterDescriptor& kernel_desc,
  const void* kernel,
  const TensorDescriptor& prev_error_signal_desc,
  const void* prev_error_signal,
  const ConvolutionDescriptor& conv_desc,
  const TensorDescriptor& error_signal_desc,
  void* error_signal,
  size_t ws_size,
  void* ws);

/** Select a backward filter convolution algorithm.
 *
 * If autotuning, memory for cuDNN algorithm runs is needed and should be
 * provided via the pointer arguments.
 *
 * @param autotune True to attempt all cuDNN algorithms and select the fastest.
 * @param deterministic True to require deterministic algorithms.
 */
dnnConvolutionBwdFilterAlgo_t get_bwd_filter_algorithm(
  bool autotune,
  bool deterministic,
  const TensorDescriptor& input_desc,
  const void* input,
  const TensorDescriptor& prev_error_signal_desc,
  const void* prev_error_signal,
  const ConvolutionDescriptor& conv_desc,
  const FilterDescriptor& kernel_gradient_desc,
  void* kernel_gradient,
  size_t ws_size,
  void* ws);

/** @brief Set the default to use tensor core operations, allowing
 *         FP32->FP16 conversions.
 */
void default_to_tensor_ops() noexcept;

/** @brief Get the default math type.
 *
 *  Will query the command-line args.
 */
dnnMathType_t get_default_convolution_math_type() noexcept;

} // namespace cudnn
} // namespace lbann

#endif // LBANN_HAS_CUDNN
#endif // LBANN_UTILS_CUDNN_HPP
