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

#include <halmd/mdsim/gpu/fields/constant_force.hpp>

#include <halmd/algorithm/gpu/apply_bind_kernel.hpp>

#include <halmd/utility/lua/lua.hpp>

using namespace boost;
using namespace std;

namespace halmd {
namespace mdsim {
namespace gpu {
namespace fields {

template <int dimension, typename float_type>
constant_force<dimension, float_type>::constant_force(
    boost::shared_ptr<particle_type> particle
  , vector_type value
  , boost::shared_ptr<logger_type> logger
)
    // dependency injection
  : particle_(particle)
  , logger_(logger)
    // parameters
  , value_(value) // conversion from fixed_vector to float4/float2
{
    zero_ = value_.x == 0 and value_.y == 0;
    if (dimension == 3) {
        // zero_ = zero_ and value_.z == 0;
        zero_ = zero_ and value[2] == 0;
    }
    LOG("module initialized with field " << value);
}

template <int dimension, typename float_type>
void constant_force<dimension, float_type>::set()
{
    LOG_ONCE("set constant force for all particles");
    if (zero_) {
        try {
            cuda::configure(particle_->dim.grid, particle_->dim.block);
            cuda::memset(particle_->g_f, 0, particle_->g_f.capacity());
            cuda::thread::synchronize();
        }
        catch (cuda::error const&) {
            LOG_ERROR("failed to set all forces to zero (due to external force field)");
            throw;
        }
    }
    else {
        try{
            cuda::configure(particle_->dim.grid, particle_->dim.block);
            fill_wrapper::kernel.fill(particle_->g_f, value_, particle_->nbox);
            cuda::thread::synchronize();
        }
        catch (cuda::error const&) {
            LOG_ERROR("failed to set forces according to external force field");
            throw;
        }
    }
}

template <int dimension, typename float_type>
void constant_force<dimension, float_type>::add()
{
    LOG_ONCE("add external force to all internal forces");
    if (zero_) {
        LOG_ONCE("Addition of a zero force field was requested.");
    }
    else {
        try {
            cuda::configure(particle_->dim.grid, particle_->dim.block);
            add_wrapper::kernel.apply(particle_->g_f, particle_->g_f, value_, particle_->nbox);
            cuda::thread::synchronize();
        }
        catch (cuda::error const&) {
            LOG_ERROR("failed to add external force field to internal forces");
            throw;
        }
    }
}

// Wrapper to connect set with slots.
template <typename constant_force_type>
typename signal<void ()>::slot_function_type
wrap_set(shared_ptr<constant_force_type> self)
{
    return bind(&constant_force_type::set, self);
}

template <typename constant_force_type>
typename signal<void ()>::slot_function_type
wrap_add(shared_ptr<constant_force_type> self)
{
    return bind(&constant_force_type::add, self);
}

template <int dimension, typename float_type>
void constant_force<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name(module_name() + ("_" + lexical_cast<string>(dimension) + "_"));
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("gpu")
            [
                namespace_("fields")
                [
                    class_<constant_force>(class_name.c_str())
                        .def(constructor<
                            shared_ptr<particle_type>
                          , vector_type
                          , shared_ptr<logger_type>
                         >())
                    .property("add", &wrap_add<constant_force>)
                    .property("set", &wrap_set<constant_force>)
                    .property("value", &constant_force::value)
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_gpu_fields_constant_force(lua_State* L)
{
    constant_force<3, float>::luaopen(L);
    constant_force<2, float>::luaopen(L);
    return 0;
}

// explicite instantiation
template class constant_force<3, float>;
template class constant_force<2, float>;

} // namespace fields
} // namespace gpu
} // namespace mdsim
} // namespace halmd
