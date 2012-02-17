/*
 * Copyright © 2011-2012  Michael Kopp
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

#ifndef HALMD_MDSIM_GPU_MOBILITIES_OSEEN_KERNEL_HPP
#define HALMD_MDSIM_GPU_MOBILITIES_OSEEN_KERNEL_HPP

#include <boost/mpl/if.hpp> //TODO necessary?

#include <cuda_wrapper/cuda_wrapper.hpp>
#include <halmd/mdsim/type_traits.hpp>
#include <halmd/numeric/mp/dsfloat.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace mobilities {

enum { WARP_SIZE = 32 };

// pointer for all functions
template <int dimension>
struct oseen_wrapper
{
    typedef typename type_traits<dimension, float>::vector_type vector_type;
    typedef typename type_traits<dimension, float>::gpu::coalesced_vector_type gpu_vector_type;
    cuda::function<void (const float4*, const gpu_vector_type*, float4*, const unsigned int, const vector_type, const float, const float)> compute_velocities_oseen;
    cuda::function<void (const float4*, const gpu_vector_type*, float4*, const unsigned int, const vector_type, const float, const float)> compute_velocities_rotne;
    static oseen_wrapper const wrapper;
};

} // namespace mobilities
} // namespace gpu
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_MOBILITIES_OSEEN_KERNEL_HPP */
