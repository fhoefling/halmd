/*
 * Copyright © 2012  Michael Kopp and Felix Höfling
 *
 * This file is part of HALMD.
 *
 * HALMD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <halmd/algorithm/gpu/apply_bind_kernel.cuh>
#include <halmd/algorithm/gpu/fill_kernel.cuh>
#include <halmd/numeric/blas/fixed_vector.hpp>

using namespace halmd; // fixed_vector
using namespace halmd::algorithm::gpu; // kernel wrapper

// fill kernels
template class fill_preserve_tag_wrapper<
    fixed_vector<float, 3>              // value_type
>;
template class fill_preserve_tag_wrapper<
    fixed_vector<float, 2>              // value_type
>;

// fill kernels for high-precision part
template class fill_wrapper<
    fixed_vector<float, 3>              // value_type
  , float4                              // coalesced_value_type
>;
template class fill_wrapper<
    fixed_vector<float, 2>              // value_type
  , float4                              // coalesced_value_type
>;

// add constant value to array
template class apply_bind2nd_preserve_tag_wrapper<
    sum_                                // functor
  , fixed_vector<float, 3>              // input_type
>;
template class apply_bind2nd_preserve_tag_wrapper<
    sum_                                // functor
  , fixed_vector<float, 2>              // input_type
>;
