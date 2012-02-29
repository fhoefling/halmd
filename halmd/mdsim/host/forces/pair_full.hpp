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

#ifndef HALMD_MDSIM_HOST_FORCES_PAIR_FULL_HPP
#define HALMD_MDSIM_HOST_FORCES_PAIR_FULL_HPP

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <lua.hpp>
#include <string>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/force_kernel.hpp>
#include <halmd/mdsim/host/force.hpp>
#include <halmd/mdsim/host/forces/smooth.hpp>
#include <halmd/mdsim/host/particle.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <halmd/utility/profiler.hpp>

namespace halmd {
namespace mdsim {
namespace host {
namespace forces {

/**
 * template class for modules implementing short ranged potential forces
 */
template <int dimension, typename float_type, typename potential_type>
class pair_full
  : public mdsim::host::force<dimension, float_type>
{
public:
    typedef mdsim::host::force<dimension, float_type> _Base;
    typedef typename _Base::vector_type vector_type;
    typedef typename _Base::matrix_type matrix_type;
    typedef typename _Base::stress_tensor_type stress_tensor_type;

    typedef host::particle<dimension, float_type> particle_type;
    typedef mdsim::box<dimension> box_type;
    typedef host::forces::smooth<dimension, float_type> smooth_type;

    inline static void luaopen(lua_State* L);

    inline pair_full(
        boost::shared_ptr<potential_type> potential
      , boost::shared_ptr<particle_type> particle
      , boost::shared_ptr<box_type> box
      // FIXME , boost::shared_ptr<smooth_type> smooth
    );
    inline virtual void compute();

    //! return potential cutoffs
    virtual matrix_type const& r_cut()
    {
        return potential_->r_cut();
    }

    /**
     * enable computation of auxiliary variables
     *
     * The flag is reset by the next call to compute().
     */
    virtual void aux_enable()
    {
        LOG_TRACE("enable computation of auxiliary variables");
        aux_flag_ = true;
    }

    //! return average potential energy per particle
    virtual double potential_energy() const
    {
        assert_aux_valid();
        return en_pot_;
    }

    //! potential part of stress tensor
    virtual stress_tensor_type stress_tensor_pot() const
    {
        assert_aux_valid();
        return stress_pot_;
    }

    //! return average potential energy per particle
    virtual double hypervirial() const
    {
        assert_aux_valid();
        return hypervirial_;
    }

private:
    typedef utility::profiler profiler_type;
    typedef typename profiler_type::accumulator_type accumulator_type;
    typedef typename profiler_type::scoped_timer_type scoped_timer_type;

    struct runtime
    {
        accumulator_type compute;
    };

    void assert_aux_valid() const
    {
        if (!aux_valid_) {
            throw std::logic_error("Auxiliary variables were not enabled in force module.");
        }
    }

    boost::shared_ptr<potential_type> potential_;
    boost::shared_ptr<particle_type> particle_;
    boost::shared_ptr<box_type> box_;
    boost::shared_ptr<smooth_type> smooth_;

    /** flag for switching the computation of auxiliary variables in function compute() */
    bool aux_flag_;
    /** flag indicates that the auxiliary variables were updated by the last call to compute() */
    bool aux_valid_;
    /** average potential energy per particle */
    double en_pot_;
    /** potential part of stress tensor */
    stress_tensor_type stress_pot_;
    /** hyper virial for each particle */
    double hypervirial_;
    /** profiling runtime accumulators */
    runtime runtime_;

    template <bool do_aux>
    inline void compute_impl_();
};

template <int dimension, typename float_type, typename potential_type>
pair_full<dimension, float_type, potential_type>::pair_full(
    boost::shared_ptr<potential_type> potential
  , boost::shared_ptr<particle_type> particle
  , boost::shared_ptr<box_type> box
  // FIXME , boost::shared_ptr<smooth_type> smooth
)
  // dependency injection
  : potential_(potential)
  , particle_(particle)
  , box_(box)
  // member initialisation
  , aux_flag_(false)          //< disable auxiliary variables by default
  , aux_valid_(false)
{}

/**
 * Compute pair forces and, if enabled, auxiliary variables,
 * i.e., potential energy, potential part of stress tensor
 *
 * Reset flag for auxiliary variables.
 */
template <int dimension, typename float_type, typename potential_type>
void pair_full<dimension, float_type, potential_type>::compute()
{
    scoped_timer_type timer(runtime_.compute);

    // call implementation which fits to current value of aux_flag_
    aux_valid_ = aux_flag_;
    if (!aux_flag_) {
        compute_impl_<false>();
    }
    else {
        compute_impl_<true>();
        aux_flag_ = false;
    }
}

template <int dimension, typename float_type, typename potential_type>
template <bool do_aux>
void pair_full<dimension, float_type, potential_type>::compute_impl_()
{
    // initialise particle forces to zero
    std::fill(particle_->f.begin(), particle_->f.end(), 0);

    // initialise potential energy and potential part of stress tensor
    if (do_aux) {
        en_pot_ = 0;
        stress_pot_ = 0;
        hypervirial_ = 0;
    }

    for (size_t i = 0; i < particle_->nbox; ++i) {
        // calculate untruncated pairwise Lennard-Jones force with all other particles
        for (size_t j = 0; j < particle_->nbox; ++j) {
            // skip self-interaction
            if (i == j ) {
                continue;
            }
            // particle distance vector
            vector_type r = particle_->r[i] - particle_->r[j];
            box_->reduce_periodic(r);
            // particle types
            unsigned a = particle_->type[i];
            unsigned b = particle_->type[j];
            // squared particle distance
            float_type rr = inner_prod(r, r);

            float_type fval, en_pot, hvir;
            boost::tie(fval, en_pot, hvir) = (*potential_)(rr, a, b);

            // optionally smooth potential yielding continuous 2nd derivative
            // FIXME test performance of template versus runtime bool
            if (smooth_) {
                smooth_->compute(std::sqrt(rr), potential_->r_cut(a, b), fval, en_pot);
            }

            // add force contribution to both particles
            particle_->f[i] += r * fval;
            particle_->f[j] -= r * fval;

            if (do_aux) {
                // add contribution to potential energy
                en_pot_ += en_pot;

                // ... and potential part of stress tensor
                stress_pot_ += fval * make_stress_tensor(rr, r);

                // compute contribution to hypervirial
                hypervirial_ += hvir / (dimension * dimension);
            }
        }
    }

    if (do_aux) {
        en_pot_ /= particle_->nbox;
        stress_pot_ /= particle_->nbox;
        hypervirial_ /= particle_->nbox;

        // ensure that system is still in valid state
        if (isinf(en_pot_)) {
            throw std::runtime_error("Potential energy diverged");
        }
    }
}

template <int dimension, typename float_type, typename potential_type>
static char const* module_name_wrapper(pair_full<dimension, float_type, potential_type> const&)
{
    return potential_type::module_name();
}

template <int dimension, typename float_type, typename potential_type>
void pair_full<dimension, float_type, potential_type>::luaopen(lua_State* L)
{
    typedef typename _Base::_Base _Base_Base;
    using namespace luabind;
    static std::string class_name("pair_full_" + boost::lexical_cast<std::string>(dimension) + "_");
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("host")
            [
                namespace_("forces")
                [
                    namespace_(class_name.c_str())
                    [
                        class_<pair_full, boost::shared_ptr<_Base_Base>, bases<_Base_Base, _Base> >(potential_type::module_name())
                            .def(constructor<
                                boost::shared_ptr<potential_type>
                              , boost::shared_ptr<particle_type>
                              , boost::shared_ptr<box_type>
                            >())
                            .property("r_cut", &pair_full::r_cut)
                            .property("module_name", &module_name_wrapper<dimension, float_type, potential_type>)
                            .scope
                            [
                                class_<runtime>("runtime")
                                    .def_readonly("compute", &runtime::compute)
                            ]
                            .def_readonly("runtime", &pair_full::runtime_)
                    ]
                ]
            ]

          , namespace_("forces")
            [
                def("pair_full", &boost::make_shared<pair_full,
                    boost::shared_ptr<potential_type>
                  , boost::shared_ptr<particle_type>
                  , boost::shared_ptr<box_type>
                >)
            ]
        ]
    ];
}

} // namespace mdsim
} // namespace host
} // namespace forces
} // namespace halmd

#endif /* ! HALMD_MDSIM_HOST_FORCES_PAIR_FULL_HPP */