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

#include <halmd/mdsim/host/fields/constant_force.hpp>
#include <halmd/utility/lua/lua.hpp>

using namespace boost;
using namespace std;

namespace halmd {
namespace mdsim {
namespace host {
namespace fields {

template <int dimension, typename float_type>
constant_force<dimension, float_type>::constant_force(
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
void constant_force<dimension, float_type>::add()
{
    LOG_TRACE("add constant force field: " << value_);

    transform(
        particle->f.begin(), particle->f.end(), particle->f.begin()
      , std::bind1st(plus<vector_type>(), value_)
    );
}

template <int dimension, typename float_type>
void constant_force<dimension, float_type>::set()
{
    LOG_TRACE("set constant force field: " << value_);

    fill(particle->f.begin(), particle->f.end(), value_);
}

// Wrappers expose signal-functions which can passed to a signal.
template <typename constant_force_type>
typename signal<void ()>::slot_function_type
wrap_add(shared_ptr<constant_force_type> self)
{
    return bind(&constant_force_type::add, self);
}

template <typename constant_force_type>
typename signal<void ()>::slot_function_type
wrap_set(shared_ptr<constant_force_type> self)
{
    return bind(&constant_force_type::set, self);
}

template <int dimension, typename float_type>
void constant_force<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name("constant_force_" + lexical_cast<string>(dimension) + "_");
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("host")
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

HALMD_LUA_API int luaopen_libhalmd_mdsim_host_fields_constant_force(lua_State* L)
{
#ifndef USE_HOST_SINGLE_PRECISION
    constant_force<3, double>::luaopen(L);
    constant_force<2, double>::luaopen(L);
#else
    constant_force<3, float>::luaopen(L);
    constant_force<2, float>::luaopen(L);
#endif
    return 0;
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class constant_force<3, double>;
template class constant_force<2, double>;
#else
template class constant_force<3, float>;
template class constant_force<2, float>;
#endif

} // namespace fields
} // namespace host
} // namespace mdsim
} // namespace halmd
