/*
 * Copyright © 2008-2010  Peter Colberg
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

#ifndef HALMD_MDSIM_GPU_PARTICLE_HPP
#define HALMD_MDSIM_GPU_PARTICLE_HPP

#include <boost/mpl/if.hpp>
#include <vector>

#include <cuda_wrapper.hpp>
#include <halmd/math/vector2d.hpp>
#include <halmd/math/vector3d.hpp>
#include <halmd/mdsim/particle.hpp>
#include <halmd/options.hpp>

namespace halmd { namespace mdsim { namespace gpu
{

template <unsigned int dimension, typename float_type>
class particle
  : public mdsim::particle<dimension>
{
public:
    typedef mdsim::particle<dimension> _Base;
    typedef typename boost::mpl::if_c<dimension == 3, float4, float2>::type gpu_vector_type;
    typedef vector<float_type, dimension> vector_type;

    particle(options const& vm);
    virtual ~particle() {}

    //
    // particles in global device memory
    //

    /** positions, tags */
    cuda::vector<float4> g_r;
    /** minimum image vectors */
    cuda::vector<gpu_vector_type> g_image;
    /** velocities, types */
    cuda::vector<float4> g_v;
    /** forces */
    cuda::vector<gpu_vector_type> g_f;
    /** neighbour lists */
    cuda::vector<unsigned int> g_neighbor;

    //
    // particles in page-locked host memory
    //

    /** positions, tags */
    cuda::host::vector<float4> h_r;
    /** minimum image vectors */
    cuda::host::vector<gpu_vector_type> h_image;
    /** velocities, types */
    cuda::host::vector<float4> h_v;

    /** number of particles in simulation box */
    using _Base::nbox;
    /** number of particle types */
    using _Base::ntype;
};

}}} // namespace halmd::mdsim::gpu

#endif /* ! HALMD_MDSIM_GPU_PARTICLE_HPP */
