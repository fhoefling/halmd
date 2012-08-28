/*
 * Copyright © 2012  Peter Colberg
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

#ifndef HALMD_OBSERVABLES_GPU_THERMODYNAMICS_KERNEL_HPP
#define HALMD_OBSERVABLES_GPU_THERMODYNAMICS_KERNEL_HPP

#include <cuda_wrapper/cuda_wrapper.hpp>
#include <halmd/config.hpp>
#include <halmd/numeric/blas/fixed_vector.hpp>
#include <halmd/mdsim/type_traits.hpp>

namespace halmd {
namespace observables {
namespace gpu {

/**
 * Compute total kinetic energy.
 */
template <int dimension, typename float_type>
class kinetic_energy
{
private:
    typedef unsigned int size_type;

public:
    /** element pointer type of input array */
    typedef size_type const* iterator;

    /**
     * Initialise kinetic energy to zero.
     */
    kinetic_energy() : mv2_(0) {}

    /**
     * Accumulate kinetic energy of a particle.
     */
    inline HALMD_GPU_ENABLED void operator()(size_type i);

    /**
     * Accumulate kinetic energy of another accumulator.
     */
    HALMD_GPU_ENABLED void operator()(kinetic_energy const& acc)
    {
        mv2_ += acc.mv2_;
    }

    /**
     * Returns total kinetic energy.
     */
    double operator()() const
    {
        return 0.5 * mv2_;
    }

    /**
     * Returns reference to texture with velocities and masses.
     */
    static cuda::texture<float4> const& get()
    {
        return texture_;
    }

private:
    /** sum over mass × square of velocity vector */
    float_type mv2_;
    /** texture with velocities and masses */
    static cuda::texture<float4> const texture_;
};

/**
 * Compute velocity of centre of mass.
 */
template <int dimension, typename float_type>
class velocity_of_centre_of_mass
{
private:
    typedef unsigned int size_type;
    typedef fixed_vector<float_type, dimension> vector_type;

public:
    /** element pointer type of input array */
    typedef size_type const* iterator;

    /**
     * Initialise momentan and mass to zero.
     */
    velocity_of_centre_of_mass() : mv_(0), m_(0) {}

    /**
     * Accumulate momentum and mass of a particle.
     */
    inline HALMD_GPU_ENABLED void operator()(size_type i);

    /**
     * Accumulate velocity centre of mass of another accumulator.
     */
    HALMD_GPU_ENABLED void operator()(velocity_of_centre_of_mass const& acc)
    {
        mv_ += acc.mv_;
        m_ += acc.m_;
    }

    /**
     * Returns velocity centre of mass.
     */
    fixed_vector<double, dimension> operator()() const
    {
        return fixed_vector<double, dimension>(mv_ / m_);
    }

    /**
     * Returns reference to texture with velocities and masses.
     */
    static cuda::texture<float4> const& get()
    {
        return texture_;
    }

private:
    /** sum over momentum vector */
    vector_type mv_;
    /** sum over mass */
    float_type m_;
    /** texture with velocities and masses */
    static cuda::texture<float4> const texture_;
};

/**
 * Compute total potential energy.
 */
template <typename float_type>
class potential_energy
{
private:
    typedef unsigned int size_type;

public:
    typedef size_type const* iterator;

    /**
     * Accumulate potential energy of a particle.
     */
    inline HALMD_GPU_ENABLED void operator()(size_type i);

    /**
     * Accumulate potential energy of another accumulator.
     */
    HALMD_GPU_ENABLED void operator()(potential_energy const& acc)
    {
        en_pot_ += acc.en_pot_;
    }

    /**
     * Returns total potential energy.
     */
    double operator()() const
    {
        return en_pot_;
    }

    /**
     * Returns reference to texture with potential energies.
     */
    static cuda::texture<float> const& get()
    {
        return texture_;
    }

private:
    /** total potential energy */
    float_type en_pot_;
    /** texture with potential energies */
    static cuda::texture<float> const texture_;
};

/**
 * Compute total virial sum.
 */
template <int dimension, typename float_type>
class virial
{
private:
    typedef unsigned int size_type;
    typedef typename mdsim::type_traits<dimension, float>::stress_tensor_type stress_pot_type;
    typedef typename mdsim::type_traits<dimension, float>::gpu::stress_tensor_type coalesced_stress_pot_type;

public:
    /** element pointer type of input array */
    typedef size_type const* iterator;

    /**
     * Accumulate stress tensor diagonal of a particle.
     */
    inline HALMD_GPU_ENABLED void operator()(size_type i);

    /**
     * Accumulate virial sum of another accumulator.
     */
    HALMD_GPU_ENABLED void operator()(virial const& acc)
    {
        virial_ += acc.virial_;
    }

    /**
     * Returns total virial sum.
     */
    double operator()() const
    {
        return virial_;
    }

    /**
     * Returns reference to texture with stress tensors.
     */
    static cuda::texture<coalesced_stress_pot_type> const& get()
    {
        return texture_;
    }

private:
    /** total virial sum */
    float_type virial_;
    /** texture with stress tensors */
    static cuda::texture<coalesced_stress_pot_type> const texture_;
};

} // namespace observables
} // namespace gpu
} // namespace halmd

#endif /* ! HALMD_OBSERVABLES_THERMODYNAMICS_KERNEL_HPP */