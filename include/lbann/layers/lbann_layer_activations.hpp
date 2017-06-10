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

#ifndef LBANN_LAYER_ACTIVATIONS_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATIONS_HPP_INCLUDED

#include "lbann/lbann_base.hpp"
#include "lbann/utils/lbann_exception.hpp"

namespace lbann {

/** Represent the type of activation function. */
enum class activation_type {
  //if you add or change the following enums, please also edit the
  //activation_name() method in the Activation class
  SIGMOID = 1,
  TANH,
  RELU,
  ID,
  LEAKY_RELU,
  SOFTPLUS,
  SMOOTH_RELU,
  ELU
};

/** Base activation function class. */
class Activation {
 public:
  virtual ~Activation(void) {}
  /** Apply the activation function elementwise to m. */
  virtual void forwardProp(ElMat& m);
  /** Apply the activation derivative function elementwise to m. */
  virtual void backwardProp(ElMat& m);
  /**
   * Apply the activation derivative function and then multiply by the error
   * signal in one step, storing into prev_error_signal.
   */
  virtual void backwardPropError(const ElMat& m, ElMat& prev_error_signal);
  static const std::string activation_name(activation_type id);
 protected:
  /** The activation function. */
  virtual DataType act(const DataType& z) = 0;
  /** The derivative of the activation function. */
  virtual DataType act_prime(const DataType& z) = 0;
};

/**
 * Sigmoid activation function.
 * See: https://en.wikipedia.org/wiki/Sigmoid_function
 */
class sigmoid_layer : public Activation {
 protected:
  DataType act(const DataType& z) {
    return (DataType(1) / (DataType(1) + std::exp(-z)));
  }
  DataType act_prime(const DataType& z) {
    const DataType sigz = act(z);
    return sigz * (DataType(1) - sigz);
  }
};

/** Hyperbolic tangent activation function. */
class tanh_layer : public Activation {
 protected:
  DataType act(const DataType& z) {
    return std::tanh(z);
  }
  DataType act_prime(const DataType& z) {
    const DataType e = std::exp(DataType(2)*z);
    return (e - DataType(1)) / (e + DataType(1));
  }
};

/**
 * Rectified linear unit activation function.
 * See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
class reLU_layer : public Activation {
 protected:
  DataType act(const DataType& z) {
    return std::max(DataType(0), z);
  }
  DataType act_prime(const DataType& z) {
    return (z > DataType(0)) ? DataType(1) : DataType(0);
  }
};

/** Identity activation function -- does nothing. */
class id_layer : public Activation {
  void forwardProp(ElMat& m) {}
  void backwardProp(ElMat& m) {}
  void backwardPropError(const ElMat& m, ElMat& prev_error_signal) {}
 protected:
  DataType act(const DataType& z) {
    return z;
  }
  DataType act_prime(const DataType& z) {
    return z;
  }
};

/**
 * Leaky rectified linear unit activation function.
 * This is a ReLU variant that avoids the dying ReLU problem where a ReLU neuron
 * can stop updating. See:
 * Maas, Andrew L., Awni Y. Hannun, and Andrew Y. Ng. "Rectifier nonlinearities
 * improve neural network acoustic models." Proc. ICML. Vol. 30. No. 1. 2013.
 */
class leaky_reLU_layer : public Activation {
 public:
  /** Leak is the amount of signal to permit for negative values. */
  leaky_reLU_layer(DataType leak = 0.01f) : m_leak(leak) {}
 protected:
  DataType act(const DataType& z) {
    return std::max(m_leak * z, z);
  }
  DataType act_prime(const DataType& z) {
    return (z > DataType(0)) ? DataType(1) : m_leak;
  }
 private:
  DataType m_leak;
};

/**
 * Softplus activation function.
 * This is a smooth approximation of the ReLU.
 * See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
class softplus_layer : public Activation {
 protected:
  DataType act(const DataType& z) {
    // Warning: Not numerically stable.
    // Better approach is to determine a threshold so that for large z,
    // softplus(z) ~= z and for small z, softplus(z) ~= exp(z).
    return std::log1p(std::exp(z));
  }
  DataType act_prime(const DataType& z) {
    return DataType(1.0) / (DataType(1.0) + std::exp(-z));
  }
};

/**
 * Smooth Rectified linear unit activation function.
 * This is an approximation to the softplus.
 */
class smooth_reLU_layer : public Activation {
 protected:
  DataType act(const DataType& z) {
    return z / (DataType(1) + std::exp(-z));
  }
  DataType act_prime(const DataType& z) {
    const DataType sigz = DataType(1) / (DataType(1) + std::exp(-z));
    return sigz + z*sigz - z*sigz*sigz;
  }
};

/**
 * Exponential linear unit.
 * Tries to speed up learning by pushing the mean of activations more towards
 * zero by allowing negative values. Helps avoid the need for batch
 * normalization.
 * See:
 * Djork-Arne Clevert, Thomas Unterthiner, and Sepp Hochreiter
 * "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"
 * ICLR 2016.
 */
class ELU_layer : public Activation {
 public:
  /**
   * alpha controls the value to which the ELU saturates for negative inputs.
   * alpha must be >= 0.
   * If alpha = 0, this turns into a ReLU.
   * Paper uses alpha = 1.0 as a good starting point.
   */
  ELU_layer(DataType alpha = 1.0f) : m_alpha(alpha) {}
 protected:
  DataType act(const DataType& z) {
    return (z > DataType(0)) ? z : (m_alpha*std::expm1(z));
  }
  DataType act_prime(const DataType& z) {
    return (z > DataType(0)) ? DataType(1) : (m_alpha*std::expm1(z) + m_alpha);
  }
 private:
  DataType m_alpha;
};

/** Return a new Activation class of type act_fn. */
template<typename... Args>
Activation *new_activation(activation_type act_fn, Args... params) {
  switch (act_fn) {
  case activation_type::SIGMOID:
    return new sigmoid_layer();
  case activation_type::TANH:
    return new tanh_layer();
  case activation_type::RELU:
    return new reLU_layer();
  case activation_type::ID:
    return new id_layer();
  case activation_type::LEAKY_RELU:
    return new leaky_reLU_layer(params...);
  case activation_type::SOFTPLUS:
    return new softplus_layer();
  case activation_type::SMOOTH_RELU:
    return new smooth_reLU_layer();
  case activation_type::ELU:
    return new ELU_layer(params...);
  default:
    throw lbann_exception("Unsupported activation type.");
  }
  return nullptr;  // Never reached.
}

}  // namespace lbann

#endif  // LBANN_LAYER_ACTIVATIONS_HPP_INCLUDED
