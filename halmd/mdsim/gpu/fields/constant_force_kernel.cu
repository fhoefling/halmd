/*
 * Copyright © 2012  Michael Kopp
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

// Explicit instantiation of algorithms must happen in their namespace.
using namespace halmd::algorithm::gpu; // wrapper
using namespace halmd; // fixed_vector

// set in 3d
template class fill_wrapper<
    float4
  , float4
>;
// set in 2d
template class fill_wrapper<
    float2
  , float2
>;

// add in 3d
// Use fixed_vector, as it has operator+ defined.
template class apply_bind2nd_wrapper<
    sum_                                // functor
  , fixed_vector<float, 4>              // input_type
  , float4                              // coalesced_input_type
  , fixed_vector<float, 4>              // output_type
  , float4                              // coalesced_output_type
>;
// add in 2d
template class apply_bind2nd_wrapper<
    sum_                                // functor
  , fixed_vector<float, 2>              // input_type
  , float2                              // coalesced_input_type
  , fixed_vector<float, 2>              // output_type
  , float2                              // coalesced_output_type
>;
