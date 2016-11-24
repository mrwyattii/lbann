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

#ifndef LBANN_LAYERS_TARGET_LAYER_UNSUPERVISED_INCLUDED
#define LBANN_LAYERS_TARGET_LAYER_UNSUPERVISED_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include "lbann/layers/lbann_target_layer.hpp"

namespace lbann
{
  class target_layer_unsupervised : public target_layer{
  public:
    target_layer_unsupervised(size_t index,lbann_comm* comm,
                              Optimizer* optimizer,
                              const uint miniBatchSize,
                              Layer* original_layer,
                              weight_initialization init=weight_initialization::glorot_uniform);

    void setup(int num_prev_neurons);
    DataType forwardProp(DataType prev_WBL2NormSum);
    void backProp();
    bool update();
    void summarize(lbann_summary& summarizer, int64_t step);
    void epoch_print() const;
    void epoch_reset();
    execution_mode get_execution_mode();
    DataType reconstruction_cost(const DistMat& Y);
    void reset_cost();
    DataType average_cost() const;


  protected:
    void fp_linearity(ElMat&, ElMat&, ElMat&, ElMat&) {}
    void bp_linearity() {}

  private:
    Layer* m_original_layer;
    DataType aggregate_cost;
    long num_backprop_steps;
    /// Views of the weight matrix that allow you to separate activation weights from bias weights
    DistMat m_activation_weights_v;
    DistMat m_bias_weights_v;
    DistMat m_activation_weights_gradient_v;
    DistMat m_bias_weights_gradient_v;
    DistMat m_bias_bp_t;
    DataType m_bias_term;
    weight_initialization m_weight_initialization;
  };
}

#endif  // LBANN_LAYERS_TARGET_LAYER_UNSUPERVISED_HPP_INCLUDED
