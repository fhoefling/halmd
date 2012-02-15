/*
 * Copyright © 2011  Michael Kopp
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

#ifndef HALMD_ALGORITHM_GPU_FILL_KERNEL_HPP
#define HALMD_ALGORITHM_GPU_FILL_KERNEL_HPP

#include <cuda_wrapper/cuda_wrapper.hpp>

namespace halmd {
namespace algorithm {
namespace gpu {

template <
    typename value_type
  , typename coalesced_value_type = value_type
>
struct fill_wrapper
{
    cuda::function<void (coalesced_value_type*, value_type, unsigned int)> fill;
    static fill_wrapper const kernel;
};

template<typename vector_type>
struct fill_preserve_tag_wrapper
{
    cuda::function<void (float4*, vector_type, unsigned int)> fill_preserve_tag;
    static fill_preserve_tag_wrapper const kernel;
};

} // namespace algorithm
} // namespace gpu
} // namespace halmd

#endif /* ! HALMD_ALGORITHM_GPU_FILL_KERNEL_HPP */
