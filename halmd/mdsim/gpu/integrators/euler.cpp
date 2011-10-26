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

#include <algorithm>
#include <cmath>
#include <string>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/gpu/integrators/euler.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <halmd/utility/scoped_timer.hpp>
#include <halmd/utility/timer.hpp>

using namespace boost;
using namespace std;

namespace halmd {
namespace mdsim {
namespace gpu {
namespace integrators {

template <int dimension, typename float_type>
euler<dimension, float_type>::euler(
    shared_ptr<particle_type> particle
  , shared_ptr<box_type const> box
  , double timestep
  , shared_ptr<logger_type> logger
)
  // dependency injection
  : particle_(particle)
  , box_(box)
  , logger_(logger)
  // reference CUDA C++ euler_wrapper
  , wrapper_(&euler_wrapper<dimension>::wrapper)
{
    this->timestep(timestep);

    try {
        cuda::copy(static_cast<vector_type>(box_->length()), wrapper_->box_length);
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to initialize Euler integrator symbols");
        throw;
    }
}

/**
 * set integration time-step
 */
template <int dimension, typename float_type>
void euler<dimension, float_type>::timestep(double timestep)
{
    timestep_ = timestep;
    timestep_half_ = 0.5 * timestep_;

    try {
        cuda::copy(timestep_, wrapper_->timestep);
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to initialize Euler integrator symbols");
        throw;
    }

    LOG("integration timestep: " << timestep_);
}

/**
 * Perform (Euler) integration: If v is set, update r
 */
template <int dimension, typename float_type>
void euler<dimension, float_type>::integrate()
{
    try {
        scoped_timer_type timer(runtime_.integrate); //< start timer -- runs 'till destruction
        cuda::configure(particle_->dim.grid, particle_->dim.block);
        wrapper_->integrate(
            particle_->g_r, particle_->g_image, particle_->g_v ) ;
        cuda::thread::synchronize();
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to stream euler integration on GPU");
        throw;
    }
}

/**
 * Finalize Euler (do nothing). Euler does not need finalisation.
 */
template <int dimension, typename float_type>
void euler<dimension, float_type>::finalize() { }

template <int dimension, typename float_type>
static char const* module_name_wrapper(euler<dimension, float_type> const&)
{
    return euler<dimension, float_type>::module_name();
}

template <int dimension, typename float_type>
void euler<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name(module_name() + ("_" + lexical_cast<string>(dimension) + "_"));
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("gpu")
            [
                namespace_("integrators")
                [
                    class_<euler, shared_ptr<_Base>, _Base>(class_name.c_str())
                        .def(constructor<
                            shared_ptr<particle_type>
                          , shared_ptr<box_type const>
                          , double
                          , shared_ptr<logger_type>
                        >())
                        .property("module_name", &module_name_wrapper<dimension, float_type>)
                        .scope
                        [
                            class_<runtime>("runtime")
                                .def_readonly("integrate", &runtime::integrate)
                        ]
                        .def_readonly("runtime", &euler::runtime_)
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_gpu_integrators_euler(lua_State* L)
{
    euler<3, float>::luaopen(L);
    euler<2, float>::luaopen(L);
    return 0;
}

// explicit instantiation
template class euler<3, float>;
template class euler<2, float>;

} // namespace integrators
} // namespace gpu
} // namespace mdsim
} // namespace halmd