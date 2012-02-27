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

#ifndef HALMD_MDSIM_GPU_FIELDS_CONSTANT_FORCE
#define HALMD_MDSIM_GPU_FIELDS_CONSTANT_FORCE

#include <halmd/mdsim/gpu/particle.hpp>
#include <halmd/io/logger.hpp>
#include <boost/shared_ptr.hpp>
#include <halmd/algorithm/gpu/apply_bind_kernel.hpp>
#include <halmd/algorithm/gpu/fill_kernel.hpp>
#include <halmd/numeric/blas/fixed_vector.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace fields {

template <int dimension, typename float_type>
class constant_force
{
public:
    typedef gpu::particle<dimension, float_type> particle_type;
    typedef typename particle_type::gpu_vector_type gpu_vector_type; // 3D: float4, 2D: float2
    typedef typename halmd::fixed_vector<float, (dimension == 3 ? 4 : 2)> fixed_vector_type; // convertible to/from gpu_vector_type
    typedef typename particle_type::vector_type vector_type; // fixed_vector
    typedef logger logger_type;

    // Wrapper for transform kernels.
    typedef typename halmd::algorithm::gpu::fill_wrapper<
        gpu_vector_type
      , gpu_vector_type
    > fill_wrapper;
    typedef typename halmd::algorithm::gpu::apply_bind2nd_wrapper<
        halmd::algorithm::gpu::sum_
      , fixed_vector_type
      , gpu_vector_type
      , fixed_vector_type
      , gpu_vector_type
    > add_wrapper;

    static char const* module_name() { return "constant_force"; }

    static void luaopen(lua_State* L);

    /**
     * @param particle gpu::particle instance.
     * @param f_ext external force field to add/set
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

    vector_type value() const
    {
        return value_;
    }

    // Don't use a reference, as this should be called from lua.
    void set_value(vector_type const value)
    {
        LOG("value reset to " << value);
        value_ = value;
    }

private:
    //! particle instance
    boost::shared_ptr<particle_type> particle_;
    //! module logger
    boost::shared_ptr<logger_type> logger_;
    //! Value of the field to add/set.
    gpu_vector_type value_;
    //! Whether the field to add/set is zero.
    bool zero_;
};

} // namespace fields
} // namespace gpu
} // namespace mdsim
} // namespace halmd


#endif  // HALMD_MDSIM_GPU_FIELDS_CONSTANT_FORCE
