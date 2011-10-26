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

#include <boost/mpl/if.hpp>

#include <halmd/mdsim/gpu/integrators/euler_kernel.cuh>
#include <halmd/mdsim/gpu/integrators/euler_kernel.hpp>
#include <halmd/mdsim/gpu/particle_kernel.cuh>
#include <halmd/numeric/blas/blas.hpp>
#include <halmd/numeric/mp/dsfloat.hpp>
#include <halmd/utility/gpu/thread.cuh>
#include <halmd/utility/gpu/variant.cuh>

using namespace boost::mpl;
using namespace halmd::mdsim::gpu::particle_kernel;
using namespace halmd::utility::gpu;

namespace halmd {
namespace mdsim {
namespace gpu {
namespace integrators {
namespace euler_kernel {

/** integration time-step */
static __constant__ float timestep_;
/** cuboid box edge length */
static __constant__ variant<map<pair<int_<3>, float3>, pair<int_<2>, float2> > > box_length_;

/**
 * Euler integration
 *
 * @param g_r positions (CUDA vector in global memory of gpu)
 * @param g_image number of times the particle exceeded the box margin (CUDA vector)
 * @param g_v velocities (CUDA vector in global memory of gpu)
 * @param g_f forces (CUDA vector)
 */
template <
    typename vector_type //< dsfloat-precision
  , typename vector_type_ //< float-precision
  , typename gpu_vector_type
>
__global__ void _integrate(
  float4* g_r,
  gpu_vector_type* g_image,
  float4* g_v)
{
    // get information which thread this is and thus which particles are to
    // be processed
    unsigned int const i = GTID;
    unsigned int const threads = GTDIM;
    unsigned int type, tag;
    // local copy of position and velocity
    vector_type r, v;
#ifdef USE_VERLET_DSFUN
    tie(r, type) = untagged<vector_type>(g_r[i], g_r[i + threads]);
    tie(v, tag) = untagged<vector_type>(g_v[i], g_v[i + threads]);
#else
    tie(r, type) = untagged<vector_type>(g_r[i]);
    tie(v, tag) = untagged<vector_type>(g_v[i]);
#endif
    // local copy of image
    vector_type_ image = g_image[i];
    vector_type_ L = get<vector_type::static_size>(box_length_);

    // run actual integration routine in .cuh file
    integrate(r, image, v, timestep_, L);

#ifdef USE_VERLET_DSFUN
    tie(g_r[i], g_r[i + threads]) = tagged(r, type);
#else
    g_r[i] = tagged(r, type);
#endif
    g_image[i] = image;
}

} // namespace euler_kernel

template <int dimension>
euler_wrapper<dimension> const euler_wrapper<dimension>::wrapper = {
    euler_kernel::timestep_ //timestep
  , get<dimension>(euler_kernel::box_length_) //boxlength
#ifdef USE_VERLET_DSFUN
  , euler_kernel::_integrate<fixed_vector<dsfloat, dimension>, fixed_vector<float, dimension> > //integrate function pointer
#else
  , euler_kernel::_integrate<fixed_vector<float, dimension>, fixed_vector<float, dimension> >
#endif
};

// explicit instantiation
template class euler_wrapper<3>;
template class euler_wrapper<2>;

} // namespace integrators
} // namespace gpu
} // namespace mdsim
} // namespace halmd