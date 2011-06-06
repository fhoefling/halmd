/*
 * Copyright © 2008-2011  Peter Colberg and Felix Höfling
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

#include <cuda_wrapper/cuda_wrapper.hpp>
#include <string>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/gpu/forces/lennard_jones_simple.hpp>
#include <halmd/mdsim/gpu/forces/lennard_jones_simple_kernel.hpp>
#include <halmd/mdsim/gpu/forces/pair_trunc_kernel.hpp>
#include <halmd/utility/lua/lua.hpp>

using namespace boost;
using namespace std;

namespace halmd
{
namespace mdsim { namespace gpu { namespace forces
{

/**
 * Initialise Lennard-Jones potential parameters
 */
template <typename float_type>
lennard_jones_simple<float_type>::lennard_jones_simple(float_type cutoff)
  // allocate potential parameters
  : r_cut_(1, 1)
{
    r_cut_(0, 0) = cutoff;
    rr_cut_ = cutoff * cutoff;

    // energy shift due to truncation at cutoff length
    float_type rri_cut = 1 / rr_cut_;
    float_type r6i_cut = rri_cut * rri_cut * rri_cut;
    en_cut_ = 4 * r6i_cut * (r6i_cut - 1);

    LOG("potential cutoff length: r_c = " << r_cut_(0, 0));
    LOG("potential cutoff energy: U = " << en_cut_);

    cuda::copy(rr_cut_, lennard_jones_simple_wrapper::rr_cut);
    cuda::copy(en_cut_, lennard_jones_simple_wrapper::en_cut);
}

template <typename float_type>
void lennard_jones_simple<float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("gpu")
            [
                namespace_("forces")
                [
                    class_<lennard_jones_simple, shared_ptr<lennard_jones_simple> >(module_name())
                        .def(constructor<float_type>())
                        .property("r_cut", (matrix_type const& (lennard_jones_simple::*)() const) &lennard_jones_simple::r_cut)
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_gpu_forces_lennard_jones_simple(lua_State* L)
{
    lennard_jones_simple<float>::luaopen(L);
    pair_trunc<3, float, lennard_jones_simple<float> >::luaopen(L);
    pair_trunc<2, float, lennard_jones_simple<float> >::luaopen(L);
    return 0;
}

// explicit instantiation
template class lennard_jones_simple<float>;
template class pair_trunc<3, float, lennard_jones_simple<float> >;
template class pair_trunc<2, float, lennard_jones_simple<float> >;

}}} // namespace mdsim::gpu::forces

} // namespace halmd