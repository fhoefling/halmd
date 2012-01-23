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

#ifndef HALMD_MDSIM_GPU_MOBILITIES_OSEEN_HPP
#define HALMD_MDSIM_GPU_MOBILITIES_OSEEN_HPP

#include <lua.hpp>
#include <utility>

#include <cuda_wrapper/cuda_wrapper.hpp> // cuda::vector
#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/gpu/particle.hpp>
#include <halmd/mdsim/gpu/mobilities/oseen_kernel.hpp>
#include <halmd/mdsim/mobility.hpp>
#include <halmd/numeric/mp/dsfloat.hpp>
#include <halmd/random/gpu/random.hpp> // necessary?
#include <halmd/utility/profiler.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace mobilities {

template <int dimension, typename float_type>
class oseen
  : public mdsim::mobility<dimension>
{
public:
    typedef mdsim::box<dimension> box_type;
    typedef mdsim::mobility<dimension> _Base;
    typedef gpu::particle<dimension, float_type> particle_type;
    typedef typename particle_type::vector_type vector_type;
    typedef typename particle_type::gpu_vector_type gpu_vector_type;
    typedef halmd::mdsim::gpu::mobilities::oseen_wrapper<dimension> wrapper_type;

    static char const* module_name() { return "oseen"; }

    boost::shared_ptr<particle_type> particle;
    boost::shared_ptr<box_type> box;

    static void luaopen(lua_State* L);

    oseen(
        boost::shared_ptr<particle_type> particle
      , boost::shared_ptr<box_type> box
      , float radius
      , float viscosity
      , int order
    );

    // inherited functions
    virtual void compute();
    virtual void compute_velocities();

    //! returns radius
    float radius() const
    {
        return radius_;
    }

    //! returns dynamic viscosity of fluid
    float viscosity() const
    {
        return viscosity_;
    }

    //! returns self mobility of particle
    float self_mobility() const
    {
        return self_mobility_ ;
    }

    //! returns order of integration
    int order() const
    {
        return order_;
    }

protected:
    typedef utility::profiler profiler_type;
    typedef typename profiler_type::accumulator_type accumulator_type;

    struct runtime
    {
        accumulator_type compute_velocities;
        accumulator_type compute;
    };

    //! hydrodynamic radius
    float radius_;
    //! dynamic viscosity of fluid
    float viscosity_;
    //! self mobility (1/6 pi eta a), a==radius_, eta==viscosity_
    float self_mobility_;
    //! order of accuracy of hydrodynamic interaction in (a/r)
    int order_;
    //! store box length as float-vector here (it's passed from box as double-vector)
    vector_type box_length_;
    /** profiling runtime accumulators */
    runtime runtime_;
};

} // namespace velocities
} // namespace gpu
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_MOBILITIES_OSEEN_HPP */
