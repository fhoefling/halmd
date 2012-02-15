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

#include <boost/preprocessor/repetition/enum_params.hpp>

#include <halmd/algorithm/gpu/fill_kernel.hpp>
#include <halmd/utility/gpu/thread.cuh>
#include <halmd/mdsim/gpu/particle_kernel.cuh> // tie/tag

namespace halmd {
namespace algorithm {
namespace gpu {
namespace fill_kernel {

/**
 * set one value for a whole array
 */
template <
    typename value_type // providing this explicitely can be used for conversions
  , typename coalesced_value_type
>
__global__ void fill(
    coalesced_value_type* g_data
  , value_type value
  , unsigned int n
)
{
    for (unsigned int i = GTID; i < n; i += GTDIM) {
        g_data[i] = value;
    }
}

/**
 * set one value for a whole array of float4s but preserve the w component
 *
 * This should be useful for tagged float4 fields.
 */
template<typename vector_type>
__global__ void fill_preserve_tag(
    float4* g_data
  , vector_type value // must have ::static_size=2 or =3
  , unsigned int n
)
{
    using halmd::algorithm::gpu::get;
    using namespace halmd::mdsim::gpu::particle_kernel;
    for (unsigned int i = GTID; i < n; i += GTDIM) {
        // unsigned int tag = halmd::mdsim::gpu::particle_kernel::untagged<vector_type>(g_data[i]).get<1>(); // boost notation
        unsigned int tag = get<1>(untagged<vector_type>(g_data[i]));
        g_data[i] = tagged(value, tag);
    }
}

} // namespace fill_kernel

// bind function to wrapper

//
// To avoid repeating template arguments and at the same time not
// define an ugly macro, we use BOOST_PP_ENUM_PARAMS to generate
// the template arguments. The meaningless names (T0, T1, …) will
// never show up in compile error messages, as the compiler uses
// the template argument names of the *declaration*.
//

template <BOOST_PP_ENUM_PARAMS(2, typename T)>
fill_wrapper<BOOST_PP_ENUM_PARAMS(2, T)> const fill_wrapper<BOOST_PP_ENUM_PARAMS(2, T)>::kernel = {
    fill_kernel::fill<BOOST_PP_ENUM_PARAMS(2, T)>
};

template<typename vector_type>
fill_preserve_tag_wrapper<vector_type> const fill_preserve_tag_wrapper<vector_type>::kernel = { fill_kernel::fill_preserve_tag<vector_type> };

} // namespace algorithm
} // namespace gpu
} // namespace halmd
