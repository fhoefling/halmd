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

#ifndef HALMD_MDSIM_HOST_FIELDS_CONSTANT_FORCE_HPP
#define HALMD_MDSIM_HOST_FIELDS_CONSTANT_FORCE_HPP

#include <boost/shared_ptr.hpp>
#include <lua.hpp>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/host/particle.hpp>

namespace halmd {
namespace mdsim {
namespace host {
namespace fields {

/**
 * \brief Add or set a constant force for all particles.
 *
 * This Module can be used to simulate external fields.
 */
template <int dimension, typename float_type>
class constant_force
{
public:
    typedef host::particle<dimension, float_type> particle_type;
    typedef typename particle_type::vector_type vector_type; // fixed_vector
    typedef logger logger_type;

    boost::shared_ptr<particle_type> particle;

    static void luaopen(lua_State* L);

    /**
     * @param particle host::particle instance.
     * @param value external force field to add/set
     */
    constant_force(
        boost::shared_ptr<particle_type> particle
      , vector_type value
      , boost::shared_ptr<logger_type> logger = boost::make_shared<logger_type>()
    );

    //! Add the external force field to all particles.
    void add();

    //! Set forces of all particles to the given value.
    void set();

    //! return constant force
    vector_type value() const
    {
        return value_;
    }

    //! set new value for constant field
    void set_value(vector_type const value)
    {
        value_ = value;
    }

private:
    /** module logger */
    boost::shared_ptr<logger_type> logger_;
    //! The external force field.
    vector_type value_;
};

} // namespace fields
} // namespace host
} // namespace forces
} // namespace halmd

#endif /* ! HALMD_MDSIM_HOST_FIELDS_CONSTANT_FORCE_HPP */
