/*
 * Copyright © 2008-2010  Peter Colberg and Felix Höfling
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
#include <halmd/mdsim/gpu/integrators/verlet.hpp>
#include <halmd/utility/lua_wrapper/lua_wrapper.hpp>
#include <halmd/utility/scoped_timer.hpp>
#include <halmd/utility/timer.hpp>

using namespace boost;
using namespace std;

namespace halmd
{
namespace mdsim { namespace gpu { namespace integrators
{

template <int dimension, typename float_type>
verlet<dimension, float_type>::verlet(
    shared_ptr<particle_type> particle
  , shared_ptr<box_type> box
  , double timestep
)
  // dependency injection
  : particle(particle)
  , box(box)
  // reference CUDA C++ verlet_wrapper
  , wrapper(&verlet_wrapper<dimension>::wrapper)
{
    this->timestep(timestep);

#ifdef USE_VERLET_DSFUN
    //
    // Double-single precision requires two single precision
    // "words" per coordinate. We use the first part of a GPU
    // vector for the higher (most significant) words of all
    // particle positions or velocities, and the second part for
    // the lower (least significant) words.
    //
    // The additional memory is allocated using reserve(), which
    // increases the capacity() without changing the size().
    //
    // Take care to pass capacity() as an argument to cuda::copy
    // or cuda::memset calls if needed, as the lower words will
    // be ignored in the operation.
    //
    LOG("using velocity-Verlet integration in double-single precision");
    particle->g_r.reserve(2 * particle->dim.threads());
    // particle images remain in single precision as they
    // contain integer values (and otherwise would not matter
    // for the long-time stability of the Verlet integrator)
    particle->g_v.reserve(2 * particle->dim.threads());
#else
    LOG_WARNING("using velocity-Verlet integration in single precision");
#endif

    try {
        cuda::copy(static_cast<vector_type>(box->length()), wrapper->box_length);
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to initialize Verlet integrator symbols");
        throw;
    }
}

/**
 * set integration time-step
 */
template <int dimension, typename float_type>
void verlet<dimension, float_type>::timestep(double timestep)
{
    timestep_ = timestep;
    timestep_half_ = 0.5 * timestep_;

    try {
        cuda::copy(timestep_, wrapper->timestep);
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to initialize Verlet integrator symbols");
        throw;
    }

    LOG("integration timestep: " << timestep_);
}

/**
 * register module runtime accumulators
 */
template <int dimension, typename float_type>
void verlet<dimension, float_type>::register_runtimes(profiler_type& profiler)
{
    profiler.register_runtime(runtime_.integrate, "integrate", "first half-step of velocity-Verlet");
    profiler.register_runtime(runtime_.finalize, "finalize", "second half-step of velocity-Verlet");
}

/**
 * First leapfrog half-step of velocity-Verlet algorithm
 */
template <int dimension, typename float_type>
void verlet<dimension, float_type>::integrate()
{
    try {
        scoped_timer<timer> timer_(runtime_.integrate);
        cuda::configure(particle->dim.grid, particle->dim.block);
        wrapper->integrate(
            particle->g_r, particle->g_image, particle->g_v, particle->g_f);
        cuda::thread::synchronize();
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to stream first leapfrog step on GPU");
        throw;
    }
}

/**
 * Second leapfrog half-step of velocity-Verlet algorithm
 */
template <int dimension, typename float_type>
void verlet<dimension, float_type>::finalize()
{
    // TODO: possibly a performance critical issue:
    // the old implementation had this loop included in update_forces(),
    // which saves one additional read of the forces plus the additional kernel execution
    // and scheduling
    try {
        scoped_timer<timer> timer_(runtime_.finalize);
        cuda::configure(particle->dim.grid, particle->dim.block);
        wrapper->finalize(particle->g_v, particle->g_f);
        cuda::thread::synchronize();
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to stream second leapfrog step on GPU");
        throw;
    }
}

template <int dimension, typename float_type>
static char const* module_name_wrapper(verlet<dimension, float_type> const&)
{
    return verlet<dimension, float_type>::module_name();
}

template <int dimension, typename float_type>
void verlet<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name(module_name() + ("_" + lexical_cast<string>(dimension) + "_"));
    module(L)
    [
        namespace_("halmd_wrapper")
        [
            namespace_("mdsim")
            [
                namespace_("gpu")
                [
                    namespace_("integrators")
                    [
                        class_<verlet, shared_ptr<_Base>, bases<_Base> >(class_name.c_str())
                            .def(constructor<shared_ptr<particle_type>, shared_ptr<box_type>, double>())
                            .def("register_runtimes", &verlet::register_runtimes)
                            .property("module_name", &module_name_wrapper<dimension, float_type>)
                    ]
                ]
            ]
        ]
    ];
}

namespace // limit symbols to translation unit
{

__attribute__((constructor)) void register_lua()
{
    lua_wrapper::register_(1) //< distance of derived to base class
    [
        &verlet<3, float>::luaopen
    ]
    [
        &verlet<2, float>::luaopen
    ];
}

} // namespace

// explicit instantiation
template class verlet<3, float>;
template class verlet<2, float>;

}}} // namespace mdsim::gpu::integrators

} // namespace halmd