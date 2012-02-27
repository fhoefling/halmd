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

#include <algorithm>    // transform, fill
#include <functional>   // plus, bind1st
#include <string>

#include <halmd/mdsim/host/fields/constant_velocity.hpp>
#include <halmd/utility/lua/lua.hpp>

using namespace boost;
using namespace std;

namespace halmd {
namespace mdsim {
namespace host {
namespace fields {

template <int dimension, typename float_type>
constant_velocity<dimension, float_type>::constant_velocity(
    shared_ptr<particle_type> particle
  , vector_type value
  , boost::shared_ptr<logger_type> logger
)
    // dependency injection
  : particle(particle)
  , logger_(logger)
    // set parameters
  , value_(value)
{
    LOG("apply constant force field: " << value_);
}

template <int dimension, typename float_type>
void constant_velocity<dimension, float_type>::add()
{
    LOG_TRACE("add constant velocity field: " << value_);

    transform(
        particle->v.begin(), particle->v.end(), particle->v.begin()
      , std::bind1st(plus<vector_type>(), value_)
    );
}

template <int dimension, typename float_type>
void constant_velocity<dimension, float_type>::set()
{
    LOG_TRACE("set constant velocity field: " << value_);

    fill(particle->v.begin(), particle->v.end(), value_);
}

// Wrappers expose signal-functions which can passed to a signal.
template <typename constant_velocity_type>
typename signal<void ()>::slot_function_type
wrap_add(shared_ptr<constant_velocity_type> self)
{
    return bind(&constant_velocity_type::add, self);
}

template <typename constant_velocity_type>
typename signal<void ()>::slot_function_type
wrap_set(shared_ptr<constant_velocity_type> self)
{
    return bind(&constant_velocity_type::set, self);
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
            namespace_("host")
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
                    .property("value", &constant_velocity::value)
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_host_fields_constant_velocity(lua_State* L)
{
#ifndef USE_HOST_SINGLE_PRECISION
    constant_velocity<3, double>::luaopen(L);
    constant_velocity<2, double>::luaopen(L);
#else
    constant_velocity<3, float>::luaopen(L);
    constant_velocity<2, float>::luaopen(L);
#endif
    return 0;
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class constant_velocity<3, double>;
template class constant_velocity<2, double>;
#else
template class constant_velocity<3, float>;
template class constant_velocity<2, float>;
#endif

} // namespace fields
} // namespace host
} // namespace mdsim
} // namespace halmd
