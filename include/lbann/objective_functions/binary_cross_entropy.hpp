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

#ifndef LBANN_OBJECTIVE_FUNCTION_BINARY_CROSS_ENTROPY_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_BINARY_CROSS_ENTROPY_HPP_INCLUDED

#include "lbann/objective_functions/objective_function.hpp"

namespace lbann {

namespace objective_functions {

/** Mean squared error objective function. */
class binary_cross_entropy : public objective_function {

 public:
  /** Default constructor. */
  binary_cross_entropy() = default;
  /** Copy constructor. */
  binary_cross_entropy(const binary_cross_entropy& other) = default;
  /** Copy assignment operator. */
  binary_cross_entropy& operator=(const binary_cross_entropy& other) = default;
  /** Destructor. */
  ~binary_cross_entropy() = default;
  /** Copy function. */
  binary_cross_entropy* copy() const { return new binary_cross_entropy(*this); }

  /** Compute the mean squared error objective function.
   *  Given a prediction \f$y\f$ and ground truth \f$\hat{y}\f$, the
   *  mean squared error is
   *    \f[
   *    MSE(y,\hat{y}) = \frac{1}{n} \lVert y - \hat{y} \rVert^2_2
   *    \f]
   *  This function updates the objective function value with the mean
   *  value of the mean squared error across the mini-batch.
   */
  void compute_value(const AbsDistMat& predictions,
                     const AbsDistMat& ground_truth);

  /** Compute the gradient of the mean squared error objective function.
   *  Given a prediction \f$y\f$ and ground truth \f$\hat{y}\f$, the
   *  gradient of the mean squared error is
   *    \f[
   *    \nabla_y MSE (y,\hat{y}) = \frac{2}{n} (y - \hat{y})
   *    \f]
   */
  void compute_gradient(const AbsDistMat& predictions,
                        const AbsDistMat& ground_truth,
                        AbsDistMat& gradient);

  /** Get the name of the objective function. */
  std::string name() const { return "binary_cross_entropy"; }

};

}  // namespace objective_functions

}  // namespace lbann

#endif  // LBANN_OBJECTIVE_FUNCTION_BINARY_CROSS_ENTROPY_HPP_INCLUDED