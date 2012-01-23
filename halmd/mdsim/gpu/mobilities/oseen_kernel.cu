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

#include <halmd/algorithm/gpu/reduction.cuh>
#include <halmd/mdsim/gpu/box_kernel.cuh> // reduce_periodic
#include <halmd/mdsim/gpu/mobilities/oseen_kernel.hpp>
#include <halmd/mdsim/gpu/particle_kernel.cuh> // tagged/untagged
#include <halmd/numeric/blas/fixed_vector/operators.hpp> //inner_product()
#include <halmd/numeric/mp/dsfloat.hpp> // sqrt
#include <halmd/utility/gpu/thread.cuh> // TID


using namespace halmd::algorithm::gpu;
using namespace halmd::mdsim::gpu::particle_kernel;

//
// Compute velocities from forces via Oseen/Rotne Prager Tensor calculus
//

namespace halmd {
namespace mdsim {
namespace gpu {
namespace mobilities {
namespace oseen_kernel {



/**
  * compute interactive-mobility
  *
  * @tparam order order of precision in (r/a) (1,2: oseen, >3: rotne-prager)
  * \note It is passed as template parameter, so that the compiler can decide
  * whether to implement oseen or rotne-prager part.
  *
  * \note This function is (normally) inlined automatically.
  */
template<
    int order
  , typename vector_type
  , typename vector_type_
>
__device__ void interaction_mobility(
    vector_type const& that_position
  , vector_type const& that_force
  , vector_type const& this_position
  , vector_type_& this_velocity
  , vector_type const& box_length
  , float const radius
)
{
    vector_type dr = this_position - that_position ;

    // apply minimum image convention
    box_kernel::reduce_periodic(dr, box_length);

    float dr2 = inner_prod(dr, dr);
    float dr_norm = sqrtf(dr2);
    float b = radius / dr_norm;

    // to the actual oseen stuff
    if( order <= 2 ) { //oseen
        this_velocity += (that_force + (inner_prod(dr,that_force) / dr2) * dr) * 0.75f * b;
    }
    else if (order <= 4) { // rotne prager
        if( dr_norm < 2*radius ) { // close branch
            this_velocity += ( 1 - (9.f / 32) * dr_norm / radius ) * that_force + ( (3.f / 32) * inner_prod(dr, that_force) / (radius * dr_norm) ) * dr;
        }
        else { // default branch
            float b2 = b * b;
            this_velocity += ((0.75f + 0.5f * b2) * b) * that_force + ((0.75f - 1.5f * b2) * b * inner_prod(dr, that_force) / dr2) * dr;
        }
    }
}


/**
  * update velocities from positions using oseen tensor calculus
  *
  * Every thread computes velocity of one single (associated) particle;
  * thread GTID is responsible for which one (g_v[GTID]).
  *
  * Positions and velocities are computed in single-precision, only for the
  * summing-up of velocities dsfun-precision is used.
  *
  * @param g_r positions in global device momory
  * @param g_f forces in global device momory
  * @param g_v velocities in global device momory -- will be updated in this function!
  * @param npart number of particles
  * @param radius hydrodynamic radius
  * @param self_mobility 1/(6*pi*eta*a) with eta being viscosity and a being radius
  * @tparam order order of precision in (r/a) (1,2: oseen, >3: rotne-prager)
  * @tparam vector_type float-vector type with appropriate dimension
  * @tparam vector_type_ dsfloat-vector type with appropriate dimension. If USE_OSEEN_DSFUN is set: dsfun. Else: float.
  * @tparam gpu_vector_type either float4 in 3D or float2 in 2D. Enables coalesced storage of forces.
  *
  */
template<
    int order
  , typename vector_type      // necessary for tagging/untagging stuff
  , typename vector_type_     // dsfun
  , typename gpu_vector_type  // forces: gpu::vector<gpu_vector_type> (float4 in 3D, float2 in 2d)
>
__global__ void _compute_velocities(
    const float4* g_r
  , const gpu_vector_type* g_f
  , float4* g_v
  , const unsigned int npart
  , const vector_type box_length
  , const float radius
  , const float self_mobility
)
{
    // get information which thread this is and thus which particles are to be processed
    unsigned int const i = GTID; // thread ID within grid
    unsigned int const threads_block = TDIM; // threads per block
    unsigned int const threads_grid = GTDIM; // threads per grid

    /* shared memory for this block
     *
     * In order to decrease the necessity for threads to read data from the
     * global memory, each block has some shared memory. It reads \e threads
     * particle positions and forces from global memory and stores them in the
     * shared memory. Then the threads in a block compute the velocities
     * resulting from this information. Only in the next timestep, information
     * from the global memory is being requested.
     *
     * \note CUDA only allows one pointer to shared memory. Yet there is the
     * special construct \code extern __shared__ type name[]; \endcode which
     * will create a (the) pointer to shared memory.  For this to work, a
     * default-shared-size must be passed to CUDA. This is done via a size_t
     * parameter in cuda::configure(..). It's the optional third parameter. So
     * make sure that configure(..) is called properly in the cpp file.
     */
    extern __shared__ char s_mem[];
    //! position of other particles in shared memory
    float4* const s_positions = reinterpret_cast<float4*>(s_mem);
    //! forces of other particles in shared memory
    gpu_vector_type* const s_forces = reinterpret_cast<gpu_vector_type*>(&s_positions[threads_block]);

    // position of particle associated with this particular thread (single precision)
    //
    // Although for particles with i >= npart the following does not make sense
    // (as there are no positions to be fetched), it does not harm though. The
    // particle module creates vectors big enough so that this operation will
    // not fail. So for each thread connected to a ghost particle there will be
    // one superfluous access to global memory. However if there was an
    // if-statement [if(i < npart)] before this, there would be one superflous
    // if statement for each single (real) particle. So as there are
    // (hopefully) much more real than ghost particles, it makes sense to
    // simply apply these operations to the ghost ones, too...
    //
    // Similar situations in this file will be denoted by a `[=*=]'-symbol.
    vector_type this_position = g_r[i];

    // velocity of particle associated with this particular thread
    vector_type_ this_velocity;
    unsigned int this_tag;
    // [=*=]
#ifdef USE_OSEEN_DSFUN
    tie(this_velocity, this_tag) = untagged<vector_type_>(g_v[i], g_v[i + threads_grid]);
#else
    tie(this_velocity, this_tag) = untagged<vector_type_>(g_v[i]);
#endif
    // reset velocity to zero //TODO this must be moved to a particle member function
    // Since we're in an overdamped regime, the velocity should consist solely
    // of external velocities (a `global' velocity). The velocity from the
    // previous timestep must not enter here.
    this_velocity = 0;

    // loop over every particle and consecutively add up velocity of this particle
    for(unsigned int tile_offset = 0; tile_offset < GTDIM; tile_offset+=TDIM) {
        // transfer positions and forces from global to shared memory
        s_positions[TID] = g_r[tile_offset + TID];
        s_forces[TID] = g_f[tile_offset + TID];
        __syncthreads(); //IMPORTANT: sync after reading. Otherwise a thread could request information not yet stored in shared memory.

        if( i < npart ) { //this could be removed [=*=]
            // loop over threads in this tile (= block)
            for(unsigned int k = 0; k < TDIM; ++k ) {
                if( tile_offset+k < npart ) { //IMPORTANT: this must not be removed!
                    // force on other particle
                    vector_type that_force = s_forces[k];

                    if( i == tile_offset+k ) { // self mobility
                        this_velocity += that_force;
                    }
                    else { // interaction
                        // position of other particle
                        vector_type that_position = s_positions[k];

                        // compute interaction of `this' and `that'
                        interaction_mobility<order>(that_position, that_force, this_position, this_velocity, box_length, radius);
                    }
                }
            }
        }
        __syncthreads(); //IMPORTANT: sync after computations
    }

    this_velocity *= self_mobility; // this has been factorized in previous computations

    // store final velocity for this particle [=*=]
#ifdef USE_OSEEN_DSFUN
    tie(g_v[i], g_v[i + threads_grid]) = tagged(this_velocity, this_tag);
#else
    g_v[i] = tagged(this_velocity, this_tag);
#endif

}


} // namespace oseen_kernel

template <int dimension>
oseen_wrapper<dimension> const oseen_wrapper<dimension>::wrapper = {
    /* gpu_vector_type does not have to be passed (in < >) as it's the type of
     * one argument and thus the compiler can identify is. On the other hand,
     * vector_type must be passed -- even though it's in the parameter list --,
     * because it's the first parameter and the _second_ (vector_type_) is
     * _not_ in the parameter list and thus has to be passed explicitly.
     *
     * This could be changed by simply using the order vector_type_,
     * vector_type, gpu_vector_type.
     */
#ifdef USE_OSEEN_DSFUN
    oseen_kernel::_compute_velocities<1, fixed_vector<float, dimension>, fixed_vector<dsfloat, dimension> > // _oseen
  , oseen_kernel::_compute_velocities<3, fixed_vector<float, dimension>, fixed_vector<dsfloat, dimension> > // _rotne
#else
    oseen_kernel::_compute_velocities<1, fixed_vector<float, dimension>, fixed_vector<float, dimension> > // _oseen
  , oseen_kernel::_compute_velocities<3, fixed_vector<float, dimension>, fixed_vector<float, dimension> > // _rotne
#endif
};

template class oseen_wrapper<3>;
template class oseen_wrapper<2>;

} // namespace mobilities
} // namespace gpu
} // namespace mdsim
} // namespace halmd
