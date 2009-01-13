/* Molecular Dynamics simulation
 *
 * Copyright © 2008-2009  Peter Colberg
 *
 * This program is free software: you can redistribute it and/or modify
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

#ifndef LJGPU_MDSIM_BASE_HPP
#define LJGPU_MDSIM_BASE_HPP

#include <ljgpu/mdsim/impl.hpp>
#include <ljgpu/mdsim/traits.hpp>

namespace ljgpu
{

template <typename mdsim_impl>
class mdsim_base
{
public:
    typedef mdsim_impl impl_type;

protected:
    /** apply periodic boundary conditions to given coordinates */
    template <typename T>
    T make_periodic(T const& r, typename T::value_type box)
    {
	return r - floor(r / box) * box;
    }
};


} // namespace ljgpu

#endif /* ! LJGPU_MDSIM_BASE_HPP */
