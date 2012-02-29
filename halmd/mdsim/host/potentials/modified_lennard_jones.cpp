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

#include <boost/numeric/ublas/io.hpp>
#include <cmath>
#include <exception>
#include <string>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/host/potentials/modified_lennard_jones.hpp>
#include <halmd/utility/lua/lua.hpp>

using namespace boost;
using namespace boost::assign;
using namespace boost::numeric::ublas;
using namespace std;

namespace halmd {
namespace mdsim {
namespace host {
namespace potentials {

/**
 * Initialise Lennard-Jones potential parameters
 */
template <typename float_type>
modified_lennard_jones<float_type>::modified_lennard_jones(
    unsigned ntype1
  , unsigned ntype2
  , matrix_type const& cutoff
  , matrix_type const& epsilon
  , matrix_type const& sigma
  , uint_matrix_type const& index_m
  , uint_matrix_type const& index_n
  , shared_ptr<logger_type> logger
)
  // allocate potential parameters
  : epsilon_(epsilon)
  , sigma_(sigma)
  , index_m_(index_m)
  , index_m_2_(index_m_ / 2)
  , index_n_(index_n)
  , index_n_2_(index_n_ / 2)
  , r_cut_sigma_(cutoff)
  , r_cut_(element_prod(sigma_, r_cut_sigma_))
  , rr_cut_(element_prod(r_cut_, r_cut_))
  , sigma2_(element_prod(sigma_, sigma_))
  , en_cut_(ntype1, ntype2)
  , logger_(logger)
{
    // energy shift due to truncation at cutoff length
    for (unsigned i = 0; i < ntype1; ++i) {
        for (unsigned j = 0; j < ntype2; ++j) {
            en_cut_(i, j) = (*this)(rr_cut_(i, j), i, j).get<1>();
        }
    }

    LOG("potential well depths: ε = " << epsilon_);
    LOG("interaction range: σ = " << sigma_);
    LOG("index of repulsion: m = " << index_m_);
    LOG("index of attraction: n = " << index_n_);
    LOG("cutoff length: r_c = " << r_cut_sigma_);
    LOG("cutoff energy: U = " << en_cut_);

    // check conditions on power law indices (after logging output)
    for (unsigned i = 0; i < ntype1; ++i) {
        for (unsigned j = 0; j < ntype2; ++j) {
            // indices must be even
            if (index_m_(i, j) & 1 || index_n_(i, j) & 1) {
                throw std::logic_error("power law indices of potential must be even");
            }
            if (index_m_(i, j) <= index_n_(i, j)) {
                throw std::logic_error("repulsive part of potential must be stronger than attraction");
            }
        }
    }
}

template <typename float_type>
void modified_lennard_jones<float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("host")
            [
                namespace_("potentials")
                [
                    class_<modified_lennard_jones, shared_ptr<modified_lennard_jones> >(module_name())
                        .def(constructor<
                            unsigned
                          , unsigned
                          , matrix_type const&
                          , matrix_type const&
                          , matrix_type const&
                          , uint_matrix_type const&
                          , uint_matrix_type const&
                          , shared_ptr<logger_type>
                        >())
                        .property("r_cut", (matrix_type const& (modified_lennard_jones::*)() const) &modified_lennard_jones::r_cut)
                        .property("r_cut_sigma", &modified_lennard_jones::r_cut_sigma)
                        .property("epsilon", &modified_lennard_jones::epsilon)
                        .property("sigma", &modified_lennard_jones::sigma)
                        .property("index_m", &modified_lennard_jones::index_m)
                        .property("index_n", &modified_lennard_jones::index_n)
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_host_potentials_modified_lennard_jones(lua_State* L)
{
#ifndef USE_HOST_SINGLE_PRECISION
    modified_lennard_jones<double>::luaopen(L);
#else
    modified_lennard_jones<float>::luaopen(L);
#endif
    return 0;
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class modified_lennard_jones<double>;
#else
template class modified_lennard_jones<float>;
#endif

} // namespace potentials
} // namespace host
} // namespace mdsim
} // namespace halmd