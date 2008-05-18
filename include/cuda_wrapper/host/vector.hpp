/* CUDA host memory vector
 *
 * Copyright (C) 2007  Peter Colberg
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

#ifndef CUDA_HOST_VECTOR_HPP
#define CUDA_HOST_VECTOR_HPP

#include <cuda_wrapper/host/allocator.hpp>
#include <cuda_wrapper/function.hpp>
#include <vector>


namespace cuda { namespace host
{

/**
 * CUDA host memory vector
 */
template <typename T, typename Alloc = allocator<T> >
class vector : public std::vector<T, Alloc>
{
public:
    typedef Alloc _Alloc;
    typedef std::vector<T, Alloc> _Base;
    typedef vector<T, Alloc> vector_type;
    typedef T value_type;
    typedef size_t size_type;

public:
    vector() {}

    /**
     * initialize host vector with copies of value
     */
    vector(size_type size, value_type const& value = value_type()) : _Base(size, value)
    {
    }

    /**
     * initialize host vector with copies of value
     */
    vector(config const& dim, value_type const& value = value_type()) : _Base(dim.threads(), value)
    {
    }
};

}} // namespace cuda::host

#endif /* ! CUDA_HOST_VECTOR_HPP */
