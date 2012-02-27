/*
 * Copyright © 2012  Michael Kopp and Felix Höfling
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

#include <halmd/mdsim/gpu/fields/constant_velocity.hpp>

#include <halmd/algorithm/gpu/apply_bind_kernel.hpp>

#include <halmd/utility/lua/lua.hpp>

using namespace boost;
using namespace std;

namespace halmd {
namespace mdsim {
namespace gpu {
namespace fields {

template <int dimension, typename float_type>
constant_velocity<dimension, float_type>::constant_velocity(
    boost::shared_ptr<particle_type> particle
  , vector_type value
  , boost::shared_ptr<logger_type> logger
)
    // dependency injection
  : particle_(particle)
  , logger_(logger)
    // parameters
  , value_(value)
{
    LOG("apply constant velocity field: " << value_);
}

template <int dimension, typename float_type>
void constant_velocity<dimension, float_type>::set()
{
    LOG_TRACE("set constant velocity field: " << value_);

    try{
        cuda::configure(particle_->dim.grid, particle_->dim.block);
        fill_preserve_tag_wrapper::kernel.fill_preserve_tag(particle_->g_v, value_, particle_->nbox);
        cuda::thread::synchronize();
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to set velocities");
        throw;
    }

#ifdef USE_VERLET_DSFUN
    // Set high precision bits of dsfloat to zero.  The external fields are
    // (per definition) not given to such a high precision.
    // FIXME use cuda::memset
    try{
        cuda::configure(particle_->dim.grid, particle_->dim.block);
        fill_wrapper::kernel.fill(
                particle_->g_v + particle_->g_v.capacity() / 2
              , 0
              , particle_->g_v.capacity() / 2
        );
        cuda::thread::synchronize();
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to set high precision bits of velocities to zero");
        throw;
    }
#endif // USE_VERLET_DSFUN
}

template <int dimension, typename float_type>
void constant_velocity<dimension, float_type>::add()
{
    LOG_TRACE("add constant velocity field: " << value_);

    // Only treat the low precision bits of dsfloat.  As the external field
    // is given with float precision (see above), the float part is added
    // here, and zeros would be added to the high precision bits.
    try {
        cuda::configure(particle_->dim.grid, particle_->dim.block);
        add_wrapper::kernel.apply_preserve_tag(particle_->g_v, particle_->g_v, value_, particle_->nbox);
        cuda::thread::synchronize();
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to add external velocity field");
        throw;
    }
}

// Wrapper to connect set with slots.
template <typename constant_velocity_type>
typename signal<void ()>::slot_function_type
wrap_set(shared_ptr<constant_velocity_type> self)
{
    return bind(&constant_velocity_type::set, self);
}

template <typename constant_velocity_type>
typename signal<void ()>::slot_function_type
wrap_add(shared_ptr<constant_velocity_type> self)
{
    return bind(&constant_velocity_type::add, self);
}

template <int dimension, typename float_type>
void constant_velocity<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name("constant_velocity_" + lexical_cast<string>(dimension) + "_");
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("gpu")
            [
                namespace_("fields")
                [
                    class_<constant_velocity>(class_name.c_str())
                        .def(constructor<
                            shared_ptr<particle_type>
                          , vector_type
                          , shared_ptr<logger_type>
                         >())
                    .property("add", &wrap_add<constant_velocity>)
                    .property("set", &wrap_set<constant_velocity>)
                    // the second function is exposed to lua as setter value()
                    .property("value", &constant_velocity::value, &constant_velocity::set_value)
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_gpu_fields_constant_velocity(lua_State* L)
{
    constant_velocity<3, float>::luaopen(L);
    constant_velocity<2, float>::luaopen(L);
    return 0;
}

// explicite instantiation
template class constant_velocity<3, float>;
template class constant_velocity<2, float>;

} // namespace fields
} // namespace gpu
} // namespace mdsim
} // namespace halmd
