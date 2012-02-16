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

#include <boost/foreach.hpp>

#include <cmath> // sqrt(), pow(), M_PI
#include <algorithm> // fill

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/host/mobilities/oseen.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <halmd/numeric/blas/fixed_vector/operators.hpp>
#include <halmd/utility/scoped_timer.hpp>
#include <halmd/utility/timer.hpp>

using namespace boost;
using namespace std;

namespace halmd {
namespace mdsim {
namespace host {
namespace mobilities {

template <int dimension, typename float_type>
oseen<dimension, float_type>::oseen(
    shared_ptr<particle_type> particle
  , shared_ptr<box_type> box
  , float radius
  , float viscosity
  , int order
)
  : particle(particle) // dependency injection
  , box(box) // dependency injection
  , radius_(radius)
  , viscosity_(viscosity)
  , order_(order)
  , self_mobility_(1 / (6 * M_PI * viscosity * radius))
{
    LOG("Particle radii: a = " << radius_ );
    LOG("Dynamic viscosity of fluid: eta = " << viscosity_ );
    LOG("Order of accurancy of hydrodynamic interaction in (a/r): " << order_ );
    if( order_ <= 2 ) LOG( "Using Oseen Tensor for hydrodynamic interaction" );
    if( order_ >= 3 ) LOG( "Using Rotne-Prager Tensor for hydrodynamic interaction" );
}

/**
 * \brief compute velocities from forces using Oseen Tensor calculus
 *
 * \note This algorithm exploits the fact that the Oseen Tensor is even in \f$
 * \vec r\f$ meaning that it computes to the same velocity regardless whether
 * \f$ \vec r\f$ or \f$ -\vec r\f$ is used.
 * This way \f$ r = \| \vec r \| \f$ needs only be computed \f$ N (N-1) \f$
 * times.
 *
 * \note The \a optimized code for interaction is taken from the GPU Module
 */
template <int dimension, typename float_type>
void oseen<dimension, float_type>::compute_velocities()
{
    scoped_timer<timer> timer_(runtime_.compute_velocities); // measure time 'till destruction

    // set all velocities to zero  //TODO this must be moved to a particle member function
    // Since we're in an overdamped regime, the velocity should consist solely
    // of external velocities (a `global' velocity). The velocity from the
    // previous timestpe must not enter here.
    fill(particle->v.begin(), particle->v.end(), 0);

    for(unsigned int i = 0; i < particle->nbox; ++i)
    {
        // self mobility
        particle->v[i] += particle->f[i];

        // interaction
        for(unsigned int j = i + 1; j < particle->nbox; ++j)
        {
            // vector connecting the two particles i and j
            vector_type dr =  particle->r[i] - particle->r[j];
            // apply minimum image convention in PBC
            box->reduce_periodic( dr );
            // distance between particles
            float_type dist2 = inner_prod(dr, dr);
            float_type dist = sqrt( dist2 );
            float_type b = radius_ / dist; //< to simplify following calculations

            if( order_ <= 2 ) { //oseen
                particle->v[i] += (particle->f[j] + (inner_prod(dr,particle->f[j]) / dist2) * dr) * 0.75f * b;
                particle->v[j] += (particle->f[i] + (inner_prod(dr,particle->f[i]) / dist2) * dr) * 0.75f * b;
            }
            else if (order_ <= 4) { // rotne prager
                if( dist < 2*radius_ ) { // close branch
                    LOG_ONCE( "Particles are at distance " << dist << " -- using close branch" );
                    particle->v[i] += ( 1 - (9.f / 32) * dist / radius_ ) * particle->f[j] + ( (3.f / 32) * inner_prod(dr, particle->f[j]) / (radius_ * dist) ) * dr;
                    particle->v[j] += ( 1 - (9.f / 32) * dist / radius_ ) * particle->f[i] + ( (3.f / 32) * inner_prod(dr, particle->f[i]) / (radius_ * dist) ) * dr;
                }
                else { // default branch
                    float_type b2 = b * b;
                    particle->v[i] += ((0.75f + 0.5f * b2) * b) * particle->f[j] + ((0.75f - 1.5f * b2) * b * inner_prod(dr, particle->f[j]) / dist2) * dr;
                    particle->v[j] += ((0.75f + 0.5f * b2) * b) * particle->f[i] + ((0.75f - 1.5f * b2) * b * inner_prod(dr, particle->f[i]) / dist2) * dr;
                }
            }
        }
        particle->v[i] *= self_mobility_; //< this has been factorized in previous computations
    }

}

//! compute oseen tensor -- NOT YET IMPLEMENTED
template <int dimension, typename float_type>
void oseen<dimension, float_type>::compute() {
    scoped_timer<timer> timer_(runtime_.compute); // measure time 'till destruction
}


//! output module name
template <int dimension, typename float_type>
static char const* module_name_wrapper(oseen<dimension, float_type> const&)
{
    return oseen<dimension, float_type>::module_name();
}

//! register class in luabind
template <int dimension, typename float_type>
void oseen<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name(module_name() + ("_" + lexical_cast<string>(dimension) + "_"));
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("host")
            [
                namespace_("mobilities")
                [
                    class_<oseen, shared_ptr<_Base>, _Base>(class_name.c_str())
                        .def(constructor<shared_ptr<particle_type>, shared_ptr<box_type>, double, double, int>())
                        .property("radius", &oseen::radius)
                        .property("viscosity", &oseen::viscosity)
                        .property("order", &oseen::order)
                        // .property("self_mobility", &oseen::self_mobility)
                        .property("module_name", &module_name_wrapper<dimension, float_type>)
                        // register runtime accumulators
                        .scope
                        [
                            class_<runtime>("runtime")
                                .def_readonly("compute", &runtime::compute)
                                .def_readonly("compute_velocities", &runtime::compute_velocities)
                        ]
                        .def_readonly("runtime", &oseen::runtime_)
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_host_mobilities_oseen(lua_State* L)
{
#ifndef USE_HOST_SINGLE_PRECISION
    oseen<3, double>::luaopen(L);
    oseen<2, double>::luaopen(L);
#else
    oseen<3, float>::luaopen(L);
    oseen<2, float>::luaopen(L);
#endif
    return 0;
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class oseen<3, double>;
template class oseen<2, double>;
#else
template class oseen<3, float>;
template class oseen<2, float>;
#endif

} // namespace mobilities
} // namespace host
} // namespace mdsim
} // namespace halmd
