/*
 * Copyright © 2010-2013 Felix Höfling
 * Copyright © 2008-2012 Peter Colberg
 *
 * This file is part of HALMD.
 *
 * HALMD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General
 * Public License along with this program. If not, see
 * <http://www.gnu.org/licenses/>.
 */

#include <boost/numeric/ublas/io.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

#include <halmd/mdsim/host/forces/pair_full.hpp>
#include <halmd/mdsim/host/forces/pair_trunc.hpp>
#include <halmd/mdsim/host/potentials/pair/adapters/hard_core.hpp>
#include <halmd/mdsim/host/potentials/pair/lennard_jones.hpp>
#include <halmd/mdsim/host/potentials/pair/truncations/truncations.hpp>
#include <halmd/utility/lua/lua.hpp>

namespace halmd {
namespace mdsim {
namespace host {
namespace potentials {
namespace pair {

/**
 * Initialise Lennard-Jones potential parameters
 */
template <typename float_type>
lennard_jones<float_type>::lennard_jones(
    matrix_type const& epsilon
  , matrix_type const& sigma
  , std::shared_ptr<logger> logger
)
  // allocate potential parameters
  : epsilon_(epsilon)
  , sigma_(check_shape(sigma, epsilon))
  , sigma2_(element_prod(sigma_, sigma_))
  , logger_(logger)
{
    LOG("potential well depths: ε = " << epsilon_);
    LOG("potential core width: σ = " << sigma_);
}

template <typename float_type>
void lennard_jones<float_type>::luaopen(lua_State* L)
{
    using namespace luaponte;
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("host")
            [
                namespace_("potentials")
                [
                    namespace_("pair")
                    [
                        class_<lennard_jones, std::shared_ptr<lennard_jones> >("lennard_jones")
                            .def(constructor<
                                matrix_type const&
                              , matrix_type const&
                              , std::shared_ptr<logger>
                            >())
                            .property("epsilon", &lennard_jones::epsilon)
                            .property("sigma", &lennard_jones::sigma)
                    ]
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_host_potentials_pair_lennard_jones(lua_State* L)
{
#ifndef USE_HOST_SINGLE_PRECISION
    lennard_jones<double>::luaopen(L);
    forces::pair_full<3, double, lennard_jones<double> >::luaopen(L);
    forces::pair_full<2, double, lennard_jones<double> >::luaopen(L);
    truncations::truncations_luaopen<double, lennard_jones<double> >(L);

    adapters::hard_core<lennard_jones<double> >::luaopen(L);
    forces::pair_full<3, double, adapters::hard_core<lennard_jones<double> > >::luaopen(L);
    forces::pair_full<2, double, adapters::hard_core<lennard_jones<double> > >::luaopen(L);
    truncations::truncations_luaopen<double, adapters::hard_core<lennard_jones<double> > >(L);
#else
    lennard_jones<float>::luaopen(L);
    forces::pair_full<3, float, lennard_jones<float> >::luaopen(L);
    forces::pair_full<2, float, lennard_jones<float> >::luaopen(L);
    truncations::truncations_luaopen<float, lennard_jones<float> >(L);

    adapters::hard_core<lennard_jones<float> >::luaopen(L);
    forces::pair_full<3, float, adapters::hard_core<lennard_jones<float> > >::luaopen(L);
    forces::pair_full<2, float, adapters::hard_core<lennard_jones<float> > >::luaopen(L);
    truncations::truncations_luaopen<float, adapters::hard_core<lennard_jones<float> > >(L);
#endif
    return 0;
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class lennard_jones<double>;
HALMD_MDSIM_HOST_POTENTIALS_PAIR_TRUNCATIONS_INSTANTIATE(lennard_jones<double>)

template class adapters::hard_core<lennard_jones<double> >;
HALMD_MDSIM_HOST_POTENTIALS_PAIR_TRUNCATIONS_INSTANTIATE(adapters::hard_core<lennard_jones<double> >)
#else
template class lennard_jones<float>;
HALMD_MDSIM_HOST_POTENTIALS_PAIR_TRUNCATIONS_INSTANTIATE(lennard_jones<float>)

template class adapters::hard_core<lennard_jones<float> >;
HALMD_MDSIM_HOST_POTENTIALS_PAIR_TRUNCATIONS_INSTANTIATE(adapters::hard_core<lennard_jones<float> >)
#endif

} // namespace pair
} // namespace potentials

namespace forces {

// explicit instantiation of force modules
#ifndef USE_HOST_SINGLE_PRECISION
template class pair_full<3, double, potentials::pair::lennard_jones<double> >;
template class pair_full<2, double, potentials::pair::lennard_jones<double> >;
HALMD_MDSIM_HOST_POTENTIALS_PAIR_TRUNCATIONS_INSTANTIATE_FORCES(double, potentials::pair::lennard_jones<double>)

template class pair_full<3, double, potentials::pair::adapters::hard_core<potentials::pair::lennard_jones<double> > >;
template class pair_full<2, double, potentials::pair::adapters::hard_core<potentials::pair::lennard_jones<double> > >;
HALMD_MDSIM_HOST_POTENTIALS_PAIR_TRUNCATIONS_INSTANTIATE_FORCES(
    double
  , potentials::pair::adapters::hard_core<potentials::pair::lennard_jones<double> >
  )
#else
template class pair_full<3, float, potentials::pair::lennard_jones<float> >;
template class pair_full<2, float, potentials::pair::lennard_jones<float> >;
HALMD_MDSIM_HOST_POTENTIALS_PAIR_TRUNCATIONS_INSTANTIATE_FORCES(float, potentials::pair::lennard_jones<float>)

template class pair_full<3, float, potentials::pair::adapters::hard_core<potentials::pair::lennard_jones<float> > >;
template class pair_full<2, float, potentials::pair::adapters::hard_core<potentials::pair::lennard_jones<float> > >;
HALMD_MDSIM_HOST_POTENTIALS_PAIR_TRUNCATIONS_INSTANTIATE_FORCES(
    float
  , potentials::pair::adapters::hard_core<potentials::pair::lennard_jones<float> >
  )
#endif

} // namespace forces
} // namespace host
} // namespace mdsim
} // namespace halmd
