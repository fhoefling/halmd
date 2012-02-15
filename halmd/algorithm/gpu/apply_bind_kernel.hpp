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

#ifndef HALMD_ALGORITHM_GPU_APPLY_BIND_KERNEL_HPP
#define HALMD_ALGORITHM_GPU_APPLY_BIND_KERNEL_HPP

#include <cuda_wrapper/cuda_wrapper.hpp>

#include <halmd/algorithm/gpu/transform.cuh>
#include <halmd/mdsim/gpu/particle_kernel.cuh> // tie/tag

namespace halmd {
namespace algorithm {
namespace gpu {

template <
    typename functor
  , typename input_type
  , typename coalesced_input_type       = input_type
  , typename output_type                = input_type
  , typename coalesced_output_type      = output_type
>
struct apply_bind1st_wrapper
{
    cuda::function<void (coalesced_input_type const*, coalesced_output_type*, coalesced_input_type const, unsigned int)> apply;
    static apply_bind1st_wrapper const kernel;
};

template <
    typename functor
  , typename input_type
  , typename coalesced_input_type       = input_type
  , typename output_type                = input_type
  , typename coalesced_output_type      = output_type
>
struct apply_bind2nd_wrapper
{
    cuda::function<void (coalesced_input_type const*, coalesced_output_type*, coalesced_input_type const, unsigned int)> apply;
    static apply_bind2nd_wrapper const kernel;
};

// preserve tag of first (unbound) argument
template <
    typename functor
  , typename input_type
  // , typename coalesced_input_type       = input_type  --> must be float4
  , typename output_type                = input_type
  // , typename coalesced_output_type      = output_type --> must be float4
>
struct apply_bind2nd_preserve_tag_wrapper
{
    cuda::function<void (float4 const*, float4*, input_type const, unsigned int)> apply_preserve_tag;
    static apply_bind2nd_preserve_tag_wrapper const kernel;
};

} // namespace gpu
} // namespace algorithm
} // namespace halmd

#endif /* ! HALMD_ALGORITHM_GPU_APPLY_BIND_KERNEL_HPP */
