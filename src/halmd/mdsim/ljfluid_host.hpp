/* Lennard-Jones fluid simulation
 *
 * Copyright © 2008-2010  Peter Colberg
 *                        Felix Höfling
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

#ifndef HALMD_MDSIM_LJFLUID_HOST_HPP
#define HALMD_MDSIM_LJFLUID_HOST_HPP

#include <algorithm>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/multi_array.hpp>
#include <boost/ref.hpp>
#include <cmath>
#include <iostream>
#include <list>
#include <vector>

#include <halmd/mdsim/hilbert.hpp>
#include <halmd/mdsim/ljfluid_base.hpp>
#include <halmd/rng/gsl_rng.hpp>
#include <halmd/util/timer.hpp>

#define foreach BOOST_FOREACH
#define range boost::make_iterator_range

namespace halmd
{

template <typename ljfluid_impl, int dimension>
class ljfluid;

template <int dimension>
class ljfluid<ljfluid_impl_host, dimension>
    : public ljfluid_base<ljfluid_impl_host, dimension>
{
public:
    typedef ljfluid_base<ljfluid_impl_host, dimension> _Base;
    typedef typename _Base::float_type float_type;
    typedef typename _Base::vector_type vector_type;
    typedef typename _Base::host_sample_type host_sample_type;
    typedef typename _Base::energy_sample_type energy_sample_type;
    typedef typename _Base::virial_tensor virial_tensor;

    /**
     * MD simulation particle
     */
    struct particle
    {
        typedef boost::reference_wrapper<particle> ref;
        enum types { A = 0, B = 1 };

        particle(unsigned int tag, types type = A) : tag(tag), type(type) {}

        /** periodically reduced particle position */
        vector_type r;
        /** periodic box traversal vector */
        vector_type R;
        /** particle velocity */
        vector_type v;
        /** particle force */
        vector_type f;
        /** particle number */
        unsigned int tag;
        /** particle type */
        types type;
        /** particle neighbours list */
        std::vector<ref> neighbour;
    };

    typedef typename std::vector<typename particle::ref> cell_list;
    typedef typename cell_list::iterator cell_list_iterator;
    typedef typename cell_list::const_iterator cell_list_const_iterator;
    typedef boost::array<int, dimension> cell_index;
    typedef boost::multi_array<cell_list, dimension> cell_lists;

public:
    /** set number of particles */
    template <typename T>
    void particles(T const& value);
    /** set neighbour list skin */
    void nbl_skin(float value);

    /** set system state from phase space sample */
    void state(host_sample_type& sample, float_type box);
    /** rescale particle velocities */
    void rescale_velocities(double coeff);
    /** initialize random number generator with seed */
    void rng(unsigned int seed);
    /** initialize random number generator from state */
    void rng(gsl::gfsr4::state_type const& state);
    /** place particles on a face-centered cubic (fcc) lattice */
    void lattice();
    /** set system temperature according to Maxwell-Boltzmann distribution */
    void temperature(double value);

    /** returns number of particles */
    unsigned int particles() const { return npart; }
    /** returns number of cells per dimension */
    int cells() const { return ncell; }
    /** returns cell length */
    double cell_length() const { return cell_length_; }

    /** MD integration step */
    void mdstep();
    /** sample phase space on host */
    void sample(host_sample_type& sample) const;
    /** sample thermodynamic equilibrium properties */
    void sample(energy_sample_type& sample) const;

    /** write parameters to HDF5 parameter group */
    void param(H5param& param) const;

private:
    /** initialise velocities from Maxwell-Boltzmann distribution */
    void boltzmann(double temp);
    /** update cell lists */
    void update_cells();
    /** returns cell list which a particle belongs to */
    cell_list& compute_cell(vector_type const& r);
    /** update neighbour lists */
    template <bool binary>
    void update_neighbours();
    /** update neighbour lists for a single cell */
    template <bool binary>
    void update_cell_neighbours(cell_index const& i);
    /** update neighbour list of particle */
    template <bool same_cell, bool binary>
    void compute_cell_neighbours(particle& p, cell_list& c);
    /** compute Lennard-Jones forces */
    template <bool binary>
    void compute_forces();
    /** compute C²-smooth potential */
    template <bool binary>
    void compute_smooth_potential(float_type r, float_type& fval, float_type& pot, unsigned int type);
    /** compute kinetic part of virial stress tensor */
    void compute_virial_kinetic();
    /** first leapfrog step of integration of equations of motion */
    void leapfrog_half();
    /** second leapfrog step of integration of equations of motion */
    void leapfrog_full();
#ifdef USE_HILBERT_ORDER
    /** order particles after Hilbert space-filling curve */
    void hilbert_order();
#endif

private:
    using _Base::npart;
    using _Base::mpart;
    using _Base::density_;
    using _Base::box_;
    using _Base::timestep_;
    using _Base::r_cut;
    using _Base::rr_cut;
    using _Base::en_cut;
    using _Base::sigma_;
    using _Base::sigma2_;
    using _Base::epsilon_;
    using _Base::r_smooth;
    using _Base::rri_smooth;
    using _Base::thermostat_steps;
    using _Base::thermostat_count;
    using _Base::thermostat_temp;

    using _Base::m_times;

    using _Base::mixture_;
    using _Base::potential_;

    /** particles */
    std::vector<particle> part;
    /** cell lists */
    cell_lists cell;
    /** random number generator */
    gsl::gfsr4 rng_;
#ifdef USE_HILBERT_ORDER
    /** 1-dimensional Hilbert curve mapping of cell lists */
    std::vector<cell_list*> hilbert_cell;
    /** particles buffer */
    std::vector<particle> part_buf;
#endif

    /** number of cells per dimension */
    int ncell;
    /** cell length */
    float_type cell_length_;
    /** neighbour list skin */
    float_type r_skin;
    /** cutoff radii with neighbour list skin */
    boost::array<float_type, 3> r_cut_skin;
    /** squared cutoff radii with neighbour list skin */
    boost::array<float_type, 3> rr_cut_skin;

    /** potential energy per particle */
    double en_pot;
    /** virial equation sum per particle */
    std::vector<virial_tensor> virial;
    /** time integral of virial stress tensor to calculate Helfand moment */
    std::vector<virial_tensor> helfand;
    /** sum over maximum velocity magnitudes since last neighbour lists update */
    float_type v_max_sum;
};

/**
 * set number of particles in system
 */
template <int dimension>
template <typename T>
void ljfluid<ljfluid_impl_host, dimension>::particles(T const& value)
{
    _Base::particles(value);

    try {
        part.reserve(npart);
#ifdef USE_HILBERT_ORDER
        part_buf.reserve(npart);
#endif
    }
    catch (std::bad_alloc const& e) {
        throw exception("failed to allocate phase space state");
    }
}

/**
 * set system state from phase space sample
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::state(host_sample_type& sample, float_type box)
{
    typedef typename host_sample_type::value_type sample_type;

    _Base::state(sample, box);

    using namespace boost::assign;
    boost::array<typename particle::types, 2> const types = list_of(particle::A)(particle::B);
    typename sample_type::position_sample_vector::const_iterator r;
    typename sample_type::velocity_sample_vector::const_iterator v;

    part.clear();
    for (size_t i = 0, n = 0; n < npart; ++i) {
        for (r = sample[i].r->begin(), v = sample[i].v->begin(); r != sample[i].r->end(); ++r, ++v, ++n) {
            particle p(n, types[i]);
            p.r = *r;
            p.R = 0;
            p.v = *v;
            part.push_back(p);
        }
    }

    // update cell lists
    update_cells();
#ifdef USE_HILBERT_ORDER
    // Hilbert space-filling curve particle sort
    hilbert_order();
#endif

    // initialize virial tensor and compute `kinetic part'
    compute_virial_kinetic();
    helfand.assign(mixture_ == BINARY ? 2 : 1, 0);

    if (mixture_ == BINARY) {
        // update Verlet neighbour lists
        update_neighbours<true>();
        // calculate forces, potential energy and virial equation sum
        compute_forces<true>();
    }
    else {
        update_neighbours<false>();
        compute_forces<false>();
    }

    // reset sum over maximum velocity magnitudes to zero
    v_max_sum = 0;
}

template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::nbl_skin(float value)
{
    r_skin = value;
    LOG("neighbour list skin: " << r_skin);

    for (size_t i = 0; i < sigma_.size(); ++i) {
        r_cut_skin[i] = r_cut[i] + r_skin;
        rr_cut_skin[i] = std::pow(r_cut_skin[i], 2);
    }

    // number of cells per dimension
    ncell = static_cast<int>(box_ / *std::max_element(r_cut_skin.begin(), r_cut_skin.end()));
    LOG("number of cells per dimension: " << ncell);

    if (ncell < 3) {
        throw exception("less than least 3 cells per dimension");
    }

    // create empty cell lists
    cell_index size;
    std::fill(size.begin(), size.end(), ncell);
    cell.resize(size);

    // derive cell length from integer number of cells per dimension
    cell_length_ = box_ / ncell;
    LOG("cell length: " << cell_length_);

#ifdef USE_HILBERT_ORDER
    // set Hilbert space-filling curve recursion depth
    unsigned int depth = static_cast<unsigned int>(std::ceil(std::log(static_cast<float_type>(ncell)) / M_LN2));
    // 32-bit integer for 2D Hilbert code allows a maximum of 16/10 levels
    depth = std::min((dimension == 3) ? 10U : 16U, depth);

    LOG("Hilbert space-filling curve recursion depth: " << depth);

    // generate 1-dimensional Hilbert curve mapping of cell lists
    hilbert_sfc<float_type, dimension> sfc(box_, depth);
    typedef std::pair<cell_list*, unsigned int> hilbert_pair;
    std::vector<hilbert_pair> hilbert_pairs;
    vector<int, dimension> x;
    for (x[0] = 0; x[0] < ncell; ++x[0]) {
        for (x[1] = 0; x[1] < ncell; ++x[1]) {
            if (dimension == 3) {
                for (x[2] = 0; x[2] < ncell; ++x[2]) {
                    vector<float_type, dimension> r(x);
                    r = (r + 0.5) * cell_length_;
                    hilbert_pairs.push_back(std::make_pair(&cell(x), sfc(r)));
                }
            }
            else {
                vector<float_type, dimension> r(x);
                r = (r + 0.5) * cell_length_;
                hilbert_pairs.push_back(std::make_pair(&cell(x), sfc(r)));
            }
        }
    }
    std::stable_sort(hilbert_pairs.begin(), hilbert_pairs.end(),
                     boost::bind(&hilbert_pair::second, _1) <
                     boost::bind(&hilbert_pair::second, _2));
    hilbert_cell.clear();
    hilbert_cell.reserve(cell.size());
    std::transform(hilbert_pairs.begin(), hilbert_pairs.end(),
                   std::back_inserter(hilbert_cell),
                   boost::bind(&hilbert_pair::first, _1));
#endif
}

template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::rescale_velocities(double coeff)
{
    LOG("rescaling velocities with coefficient: " << coeff);
    foreach (particle& p, part) {
        p.v *= coeff;
    }
}

/**
 * initialize random number generator with seed
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::rng(unsigned int seed)
{
    rng_.set(seed);
    LOG("initializing random number generator with seed: " << seed);
}

/**
 * initialize random number generator from state
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::rng(gsl::gfsr4::state_type const& state)
{
    rng_.restore(state);
    LOG("restoring random number generator from state");
}

/**
 * place particles on a face-centered cubic (fcc) lattice
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::lattice()
{
    std::vector<typename particle::types> types;
    if (mixture_ == BINARY) {
        LOG("randomly placing A and B particles on fcc lattice");
        types.resize(mpart[0], particle::A);
        types.resize(npart, particle::B);
        rng_.shuffle(types);
    }
    else {
        LOG("placing particles on fcc lattice");
        types.resize(npart, particle::A);
    }

    // particles per 2- or 3-dimensional unit cell
    const unsigned int m = 2 * (dimension - 1);
    // lower boundary for number of particles per lattice dimension
    unsigned int n = static_cast<unsigned int>(std::pow(npart / m, 1. / dimension));
    // lower boundary for total number of lattice sites
    unsigned int N = m * static_cast<unsigned int>(pow(n, dimension));

    if (N < npart) {
        n += 1;
        N = m * static_cast<unsigned int>(pow(n, dimension));
    }
    if (N > npart) {
        LOG_WARNING("lattice not fully occupied (" << N << " sites)");
    }

    // lattice distance
    float_type a = box_ / n;
    // minimum distance in 2- or 3-dimensional fcc lattice
    LOG("minimum lattice distance: " << a / std::sqrt(2.));

    boost::array<unsigned int, 2> tag = boost::assign::list_of(0)(mpart[0]);
    part.clear();
    for (unsigned int i = 0; i < npart; ++tag[types[i]], ++i) {
        particle p(tag[types[i]], types[i]);
        vector_type& r = p.r;
        // compose primitive vectors from 1-dimensional index
        if (dimension == 3) {
            r[0] = ((i >> 2) % n) + ((i ^ (i >> 1)) & 1) / 2.;
            r[1] = ((i >> 2) / n % n) + (i & 1) / 2.;
            r[2] = ((i >> 2) / n / n) + (i & 2) / 4.;
        }
        else {
            r[0] = ((i >> 1) % n) + (i & 1) / 2.;
            r[1] = ((i >> 1) / n) + (i & 1) / 2.;
        }
        r *= a;
        p.R = 0;
        part.push_back(p);
    }

    // sort particles after binary mixture species for trajectory output
    struct compare
    {
        static bool _(particle const& p1, particle const& p2)
        {
            return (p1.type < p2.type);
        }
    };
    std::stable_sort(this->part.begin(), this->part.end(), compare::_);

    // update cell lists
    update_cells();
#ifdef USE_HILBERT_ORDER
    // Hilbert space-filling curve particle sort
    hilbert_order();
#endif

    // initialize virial tensor and compute `kinetic part'
    compute_virial_kinetic();
    helfand.assign(mixture_ == BINARY ? 2 : 1, 0);

    if (mixture_ == BINARY) {
        // update Verlet neighbour lists
        update_neighbours<true>();
        // calculate forces, potential energy and virial equation sum
        compute_forces<true>();
    }
    else {
        update_neighbours<false>();
        compute_forces<false>();
    }

    // reset sum over maximum velocity magnitudes to zero
    v_max_sum = 0;
}

/**
 * initialise velocities from Maxwell-Boltzmann distribution
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::temperature(double value)
{
    LOG("initializing velocities from Maxwell-Boltzmann distribution at temperature: " << value);

    // initialize force to zero for first leapfrog half step
    foreach (particle& p, part) {
        p.f = 0;
    }
    // initialize sum over maximum velocity magnitudes since last neighbour lists update
    v_max_sum = 0;
    // and re-compute virial tensor
    compute_virial_kinetic();

    boltzmann(value);
}

/**
 * set system temperature according to Maxwell-Boltzmann distribution
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::boltzmann(double temp)
{
    // center of mass velocity
    vector_type v_cm = 0;
    // mean squared velocity
    double vv = 0;

    // generate random Maxwell-Boltzmann distributed velocity
    foreach (particle& p, part) {
        rng_.gaussian(p.v, static_cast<float_type>(temp));
        v_cm += p.v;
    }
    v_cm /= npart;

    // set center of mass velocity to zero
    foreach (particle& p, part) {
        p.v -= v_cm;
        vv += p.v * p.v;
    }
    vv /= npart;

    // rescale velocities to accurate temperature
    double s = std::sqrt(temp * dimension / vv);
    foreach (particle& p, part) {
        p.v *= s;
    }
}

/**
 * update cell lists
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::update_cells()
{
    // empty cell lists without memory reallocation
    foreach (cell_list& c, range(cell.data(), cell.data() + cell.num_elements())) {
        c.clear();
    }
    // add particles to cells
    foreach (particle& p, part) {
        compute_cell(p.r).push_back(boost::ref(p));
    }
}

/**
 * returns cell list which a particle belongs to
 */
template <int dimension>
typename ljfluid<ljfluid_impl_host, dimension>::cell_list&
ljfluid<ljfluid_impl_host, dimension>::compute_cell(vector_type const& r)
{
    cell_index index;
    for (int i = 0; i < dimension; ++i) {
        index[i] = (unsigned int)(r[i] / cell_length_) % ncell;
    }
    return cell(index);
}

/**
 * update neighbour lists
 */
template <int dimension>
template <bool binary>
void ljfluid<ljfluid_impl_host, dimension>::update_neighbours()
{
    cell_index i;
    for (i[0] = 0; i[0] < ncell; ++i[0]) {
        for (i[1] = 0; i[1] < ncell; ++i[1]) {
            if (dimension == 3) {
                for (i[2] = 0; i[2] < ncell; ++i[2]) {
                    update_cell_neighbours<binary>(i);
                }
            }
            else {
                update_cell_neighbours<binary>(i);
            }
        }
    }
}

/**
 * update neighbour lists for a single cell
 */
template <int dimension>
template <bool binary>
void ljfluid<ljfluid_impl_host, dimension>::update_cell_neighbours(cell_index const& i)
{
    foreach (particle& p, cell(i)) {
        // empty neighbour list of particle
        p.neighbour.clear();

        cell_index j;
        for (j[0] = -1; j[0] <= 1; ++j[0]) {
            for (j[1] = -1; j[1] <= 1; ++j[1]) {
                if (dimension == 3) {
                    for (j[2] = -1; j[2] <= 1; ++j[2]) {
                        // visit half of 26 neighbour cells due to pair potential
                        if (j[0] == 0 && j[1] == 0 && j[2] == 0) {
                            goto out;
                        }
                        // update neighbour list of particle
                        cell_index k;
                        for (int n = 0; n < dimension; ++n) {
                            k[n] = (i[n] + ncell + j[n]) % ncell;
                        }
                        compute_cell_neighbours<false, binary>(p, cell(k));
                    }
                }
                else {
                    // visit half of 8 neighbour cells due to pair potential
                    if (j[0] == 0 && j[1] == 0) {
                        goto out;
                    }
                    // update neighbour list of particle
                    cell_index k;
                    for (int n = 0; n < dimension; ++n) {
                        k[n] = (i[n] + ncell + j[n]) % ncell;
                    }
                    compute_cell_neighbours<false, binary>(p, cell(k));
                }
            }
        }
out:
        // visit this cell
        compute_cell_neighbours<true, binary>(p, cell(i));
    }
}

/**
 * update neighbour list of particle
 */
template <int dimension>
template <bool same_cell, bool binary>
void ljfluid<ljfluid_impl_host, dimension>::compute_cell_neighbours(particle& p1, cell_list& c)
{
    // half periodic box length for nearest mirror-image particle
    float_type box_half = 0.5 * box_;

    foreach (particle& p2, c) {
        // skip identical particle and particle pair permutations if same cell
        if (same_cell && p2.tag <= p1.tag)
            continue;

        // particle distance vector
        vector_type r = p1.r - p2.r;
        // binary particles type
        unsigned int type = (binary ? (p1.type + p2.type) : 0);
        // enforce periodic boundary conditions
        for (int i = 0; i < dimension; ++i) {
            if (r[i] > box_half) {
                r[i] -= box_;
            }
            else if (r[i] < -box_half) {
                r[i] += box_;
            }
        }
        // squared particle distance
        float_type rr = r * r;

        // enforce cutoff radius with neighbour list skin
        if (rr >= static_cast<float_type>(rr_cut_skin[type]))
            continue;

        // add particle to neighbour list
        p1.neighbour.push_back(boost::ref(p2));
    }
}

/**
 * compute Lennard-Jones forces
 */
template <int dimension>
template <bool binary>
void ljfluid<ljfluid_impl_host, dimension>::compute_forces()
{
    // initialize particle forces to zero
    foreach (particle& p, part) {
        p.f = 0;
    }

    // potential energy
    en_pot = 0;
    // half periodic box length for nearest mirror-image particle
    float_type box_half = 0.5 * box_;

    foreach (particle& p1, part) {
        // calculate pairwise Lennard-Jones force with neighbour particles
        foreach (particle& p2, p1.neighbour) {
            // particle distance vector
            vector_type r = p1.r - p2.r;
            // binary particles type
            unsigned int type = (binary ? (p1.type + p2.type) : 0);
            // enforce periodic boundary conditions
            for (int i = 0; i < dimension; ++i) {
                if (r[i] > box_half) {
                    r[i] -= box_;
                }
                else if (r[i] < -box_half) {
                    r[i] += box_;
                }
            }
            // squared particle distance
            float_type rr = r * r;

            // enforce cutoff radius
            if (rr >= rr_cut[type])
                continue;

            // compute Lennard-Jones force in reduced units
            float_type sigma2 = (binary ? sigma2_[type] : 1);
            float_type eps = (binary ? epsilon_[type] : 1);
            float_type rri = sigma2 / rr;
            float_type r6i = rri * rri * rri;
            float_type fval = 48 * rri * r6i * (r6i - 0.5) * (eps / sigma2);
            float_type pot = (4 * r6i * (r6i - 1) - en_cut[type]) * eps;

            if (potential_ == C2POT) {
                compute_smooth_potential<binary>(std::sqrt(rr), fval, pot, type);
            }

            // add force contribution to both particles
            p1.f += r * fval;
            p2.f -= r * fval;

            // add contribution to potential energy
            en_pot += pot;

            // add contribution to virial equation sum
            float_type vir = 0.5 * rr * fval;
            virial[p1.type][0] += vir;
            virial[p2.type][0] += vir;

            // compute off-diagonal virial stress tensor elements
            if (dimension == 3) {
                vir = 0.5 * r[1] * r[2] * fval;
                virial[p1.type][1] += vir;
                virial[p2.type][1] += vir;

                vir = 0.5 * r[2] * r[0] * fval;
                virial[p1.type][2] += vir;
                virial[p2.type][2] += vir;

                vir = 0.5 * r[0] * r[1] * fval;
                virial[p1.type][3] += vir;
                virial[p2.type][3] += vir;
            }
            else {
                vir = 0.5 * r[0] * r[1] * fval;
                virial[p1.type][1] += vir;
                virial[p2.type][1] += vir;
            }
        }
    }

    // finalise averages
    en_pot /= npart;
    for (size_t i = 0; i < virial.size(); ++i) {
        virial[i] /= mpart[i];
    }

    // ensure that system is still in valid state
    if (std::isinf(en_pot)) {
        throw potential_energy_divergence();
    }
}

/**
 * compute C²-smooth potential
 */
template <int dimension>
template <bool binary>
void ljfluid<ljfluid_impl_host, dimension>::compute_smooth_potential(float_type r, float_type& fval, float_type& pot, unsigned int type)
{
    float_type y = r - r_cut[binary ? type : 0];
    float_type x2 = y * y * rri_smooth;
    float_type x4 = x2 * x2;
    float_type x4i = 1 / (1 + x4);
    // smoothing function
    float_type h0_r = x4 * x4i;
    // first derivative times (r_smooth)^(-1) [sic!]
    float_type h1_r = 4 * y * rri_smooth * x2 * x4i * x4i;
    // apply smoothing function to obtain C¹ force function
    fval = h0_r * fval - h1_r * (pot / r);
    // apply smoothing function to obtain C² potential function
    pot = h0_r * pot;
}

/**
 * compute kinetic part of virial stress tensor
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::compute_virial_kinetic()
{
    virial.assign(mixture_ == BINARY ? 2 : 1, 0);

    foreach (particle& p, part) {
        virial_tensor& vir = virial[p.type];
        vector_type& v = p.v;
        vir[0] += v * v;
        if (dimension == 3) {
            vir[1] += v[1] * v[2];
            vir[2] += v[2] * v[0];
            vir[3] += v[0] * v[1];
        }
        else {
            vir[1] += v[0] * v[1];
        }
    }
}

/**
 * first leapfrog step of integration of equations of motion
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::leapfrog_half()
{
    float_type vv_max = 0;

    foreach (particle& p, part) {
        // half step velocity
        p.v += p.f * (static_cast<float_type>(timestep_) / 2);
        // full step position
        p.r += p.v * static_cast<float_type>(timestep_);
        // enforce periodic boundary conditions
        for (int i = 0; i < dimension; ++i) {
            // assumes that particle position wraps at most once per time-step
            if (p.r[i] > box_) {
                p.r[i] -= box_;
                p.R[i] += 1;
            }
            else if (p.r[i] < 0) {
                p.r[i] += box_;
                p.R[i] -= 1;
            }
        }
        // maximum squared velocity
        vv_max = std::max(vv_max, p.v * p.v);
    }

    v_max_sum += std::sqrt(vv_max);
}

/**
 * second leapfrog step of integration of equations of motion
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::leapfrog_full()
{
    foreach (particle& p, part) {
        // full step velocity
        p.v += p.f * (static_cast<float_type>(timestep_) / 2);
    }
}

#ifdef USE_HILBERT_ORDER
/**
 * order particles after Hilbert space-filling curve
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::hilbert_order()
{
    part_buf.clear();
    foreach (cell_list* c, hilbert_cell) {
        foreach (typename particle::ref& p, *c) {
            part_buf.push_back(p);
            p = boost::ref(part_buf.back());
        }
    }
    part.swap(part_buf);
}
#endif /* USE_HILBERT_ORDER */

/**
 * MD integration step
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::mdstep()
{
    // nanosecond resolution process times
    boost::array<high_resolution_timer, 5> t;

    // compute kinetic part of virial tensor with initial velocities,
    // the "static" part is added in compute_forces()
    compute_virial_kinetic();

    // calculate particle positions
    t[0].record();
    leapfrog_half();
    t[1].record();

    if (v_max_sum * timestep_ > r_skin / 2) {
        // update cell lists
        update_cells();
        t[2].record();
#ifdef USE_HILBERT_ORDER
        // Hilbert space-filling curve particle sort
        hilbert_order();
#endif
        t[3].record();
        // update Verlet neighbour lists
        if (mixture_ == BINARY) {
            update_neighbours<true>();
        }
        else {
            update_neighbours<false>();
        }
        t[4].record();
        // reset sum over maximum velocity magnitudes to zero
        v_max_sum = 0;

        m_times["update_cells"] += t[2] - t[1];
#ifdef USE_HILBERT_ORDER
        m_times["hilbert_sort"] += t[3] - t[2];
#endif
        m_times["update_neighbours"] += t[4] - t[3];
    }

    // calculate forces, potential energy and virial equation sum
    t[2].record();
    if (mixture_ == BINARY) {
        compute_forces<true>();
    }
    else {
        compute_forces<false>();
    }
    // calculate velocities
    t[3].record();
    if (thermostat_steps && ++thermostat_count >= thermostat_steps) {
        boltzmann(thermostat_temp);
    }
    else {
        leapfrog_full();
    }
    t[4].record();

    // integrate virial tensor for each component
    for (size_t i = 0; i < virial.size(); ++i) {
        helfand[i] += virial[i] * static_cast<typename virial_tensor::value_type>(timestep_);
    }

    if (thermostat_steps && thermostat_count >= thermostat_steps) {
        // reset MD steps since last heatbath coupling
        thermostat_count = 0;
        m_times["boltzmann"] += t[4] - t[3];
        m_times["velocity_verlet"] += t[1] - t[0];
    }
    else {
        m_times["velocity_verlet"] += (t[1] - t[0]) + (t[4] - t[3]);
    }
    m_times["update_forces"] += t[3] - t[2];
    m_times["mdstep"] += t[4] - t[0];
}

template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::sample(host_sample_type& sample) const
{
    typedef typename host_sample_type::value_type sample_type;
    typedef typename sample_type::position_sample_vector position_sample_vector;
    typedef typename sample_type::position_sample_ptr position_sample_ptr;
    typedef typename sample_type::velocity_sample_vector velocity_sample_vector;
    typedef typename sample_type::velocity_sample_ptr velocity_sample_ptr;

    for (size_t i = 0, m = 0; m < npart; m += mpart[i], ++i) {
        // allocate memory for trajectory sample
        position_sample_ptr r(new position_sample_vector(mpart[i]));
        velocity_sample_ptr v(new velocity_sample_vector(mpart[i]));
        sample.push_back(sample_type(r, v));
        // assign particle positions and velocities of homogenous type
        foreach (particle const& p, part) {
            if (p.type == (int) i) {
                assert(p.tag >= m && p.tag - m < mpart[p.type]);
                // periodically extended particle position
                (*r)[p.tag - m] = p.r + p.R * box_;
                // particle velocity
                (*v)[p.tag - m] = p.v;
            }
        }
    }
}

template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::sample(energy_sample_type& sample) const
{
    typedef typename std::vector<particle>::const_iterator iterator;

    // virial tensor trace and off-diagonal elements for particle species
    sample.virial = virial;
    sample.helfand = helfand;

    sample.vv = 0;
    sample.v_cm = 0;

    for (iterator p = part.begin(); p != part.end(); ++p) {
        sample.vv += p->v * p->v;
        sample.v_cm += p->v;
    }

    // mean potential energy per particle
    sample.en_pot = en_pot;
    // mean squared velocity per particle
    sample.vv /= npart;
    // mean velocity per particle
    sample.v_cm /= npart;
}

/**
 * write parameters to HDF5 parameter group
 */
template <int dimension>
void ljfluid<ljfluid_impl_host, dimension>::param(H5param& param) const
{
    _Base::param(param);

    H5xx::group node(param["mdsim"]);
    node["cells"] = ncell;
    node["cell_length"] = cell_length_;
    node["neighbour_skin"] = r_skin;
}

} // namespace halmd

#undef foreach
#undef range

#endif /* ! HALMD_MDSIM_LJFLUID_HOST_HPP */