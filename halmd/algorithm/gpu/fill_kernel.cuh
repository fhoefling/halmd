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

#include <boost/preprocessor/repetition/enum_params.hpp>

#include <halmd/algorithm/gpu/fill_kernel.hpp>
#include <halmd/utility/gpu/thread.cuh>

namespace halmd {
namespace algorithm {
namespace gpu {
namespace fill_kernel {

/**
 * fill array with value
 */
template <
    typename value_type
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
 * fill array of float4 with value, but preserve the w component
 *
 * This should be useful for tagged float4 fields.
 */
template<typename value_type>
__global__ void fill_preserve_tag(
    float4* g_data
  , value_type value
  , unsigned int n
)
{
    for (unsigned int i = GTID; i < n; i += GTDIM) {
        float w = g_data[i].w;
        float4 a = value; a.w = w;
        g_data[i] = a;
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

template<typename value_type>
fill_preserve_tag_wrapper<value_type> const fill_preserve_tag_wrapper<value_type>::kernel = {
    fill_kernel::fill_preserve_tag<value_type>
};

} // namespace algorithm
} // namespace gpu
} // namespace halmd
