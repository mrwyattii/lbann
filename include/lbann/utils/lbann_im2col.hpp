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

#ifndef LBANN_UTILS_IM2COL_HPP
#define LBANN_UTILS_IM2COL_HPP

#include "lbann/lbann_base.hpp"

namespace lbann {

/// Rearrange image blocks into matrix columns
/** The output matrix col is generated from the input tensor im by
 *  shifting a window across im. Each column of col is produced by
 *  positioning the window, extracting entries from im, and
 *  flattening.
 *  @param im               Input tensor, represented as a column
 *                          vector.
 *  @param col              Output matrix. Height should be equal
 *                          to window size and width equal to number
 *                          of window shifts. Data should be
 *                          contiguous.
 *  @param im_dims          Input tensor dimensions.
 *  @param im_pads          Zero pads for input tensor.
 *  @param num_im_channels  Number of channels in input tensor.
 *  @param window_dims      Dimensions of window.
 *  @param window_strides   Window shift strides.
 */
void im2col(const Mat& im,
            Mat& col,
            const std::vector<El::Int>& im_dims,
            const std::vector<El::Int>& im_pads,
            El::Int num_im_channels,
            const std::vector<El::Int>& window_dims,
            const std::vector<El::Int>& window_strides);

/// Rearrange matrix columns into image blocks
/** This is approximately the inverse of im2col. The output tensor im
 *  is produced from the input matrix col by shifting a window across
 *  im. Each column of col is matched with the corresponding window
 *  position and corresponding entries are added to im.
 *  @param col              Input matrix. Height should be equal to
 *                          window size and width equal to number of
 *                          window shifts. Data should be contiguous.
 *  @param im               Output tensor, represented as a column
 *                          vector.
 *  @param im_dims          Output tensor dimensions.
 *  @param im_pads          Zero pads for output tensor.
 *  @param num_im_channels  Number of channels in output tensor.
 *  @param window_dims      Dimensions of window.
 *  @param window_strides   Window shift strides.
 */
void col2im(const Mat& col,
            Mat& im,
            const std::vector<El::Int>& im_dims,
            const std::vector<El::Int>& im_pads,
            El::Int num_im_channels,
            const std::vector<El::Int>& window_dims,
            const std::vector<El::Int>& window_strides);

/// Rearrange 2D image blocks into matrix columns
/** This is an optimized implementation of im2col for 2D data. im2col
 *  will automatically call this routine if it detects 2D data.
 */
void im2col_2d(const DataType *__restrict__ input_buffer,
               DataType *__restrict__ output_buffer,
               int input_dim_x,
               int input_dim_y,
               int input_pad_x,
               int input_pad_y,
               int num_channels,
               int window_dim_x,
               int window_dim_y,
               int offset_stride_x,
               int offset_stride_y);

/// Rearrange matrix columns into 2D image blocks
/** This is an optimized implementation of col2im for 2D data. col2im
 *  will automatically call this routine if it detects 2D data.
 */
void col2im_2d(const DataType *__restrict__ input_buffer,
               DataType *__restrict__ output_buffer,
               int output_dim_x,
               int output_dim_y,
               int output_pad_x,
               int output_pad_y,
               int num_channels,
               int window_dim_x,
               int window_dim_y,
               int offset_stride_x,
               int offset_stride_y);

} // end namespace
#endif // LBANN_UTILS_IM2COL_HPP