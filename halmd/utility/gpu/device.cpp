/*
 * Copyright © 2008-2010  Peter Colberg
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

#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <exception>
#include <fstream>

#include <halmd/io/logger.hpp>
#include <halmd/utility/gpu/device.hpp>
#include <halmd/utility/gpu/device_kernel.hpp>
#include <halmd/utility/lua_wrapper/lua_wrapper.hpp>
#include <halmd/utility/multi_array.hpp>

using namespace boost;
using namespace boost::algorithm;
using namespace std;

namespace halmd
{
namespace utility { namespace gpu
{

/**
 * Initialize CUDA device
 */
device::device(vector<int> devices, unsigned int threads)
  : threads_(threads)
{
    try {
        LOG("NVIDIA driver version: " << device::nvidia_driver_version());
    }
    catch (runtime_error const& e) {
        LOG_WARNING(e.what());
    }
# if CUDA_VERSION >= 2020
    LOG("CUDA driver version: " << device::cuda_driver_version());
# endif
# if (CUDART_VERSION >= 2020)
    LOG("CUDA runtime version: " << device::cuda_runtime_version());
# endif

    // default to list of available CUDA devices
    if (devices.empty()) {
        copy(
            counting_iterator<int>(0)
          , counting_iterator<int>(cuda::device::count())
          , std::back_inserter(devices)
        );
    }

    // choose first available CUDA device
    BOOST_FOREACH (int i, devices) {
        try {
            // create CUDA context and associate it with this thread
            context_.reset(new cuda::driver::context(i));
            break;
        }
        catch (cuda::driver::error const&) {
            // device is compute-exlusive mode and in use
        }
    }

    LOG("CUDA device: " << cuda::driver::context::device());
    cuda::device::properties prop(cuda::driver::context::device());

    LOG("CUDA device name: " << prop.name());
    LOG("CUDA device total global memory: " << prop.total_global_mem() << " bytes");
    LOG("CUDA device shared memory per block: " << prop.shared_mem_per_block() << " bytes");
    LOG("CUDA device registers per block: " << prop.regs_per_block());
    LOG("CUDA device warp size: " << prop.warp_size());
    LOG("CUDA device maximum number of threads per block: " << prop.max_threads_per_block());
    LOG("CUDA device total constant memory: " << prop.total_const_mem());
    LOG("CUDA device major revision: " << prop.major());
    LOG("CUDA device minor revision: " << prop.minor());
    LOG("CUDA device clock frequency: " << prop.clock_rate() << " kHz");
    LOG("CUDA device compute capability: " << prop.major() << "." << prop.minor());

    LOG("CUDA compute version: " << device::compute_version());

    if (threads_ < 1) {
        throw runtime_error("invalid number of CUDA threads");
    }
    if (threads_ > prop.max_threads_per_block()) {
        throw runtime_error("number of CUDA threads exceeds maximum number of threads per block");
    }
    if (threads_ & (threads_ - 1)) {
        LOG_WARNING("number of CUDA threads not a power of 2");
    }
    if (threads_ % prop.warp_size()) {
        LOG_WARNING("number of CUDA threads not a multiple of warp size");
    }

    LOG("number of CUDA threads: " << threads_);
}

/**
 * Detach CUDA runtime from CUDA device context
 *
 * This explicit clean-up is needed with CUDA < 3.0.
 */
device::~device()
{
    cuda::thread::exit();
}

/**
 * Query NVIDIA driver version
 */
string device::nvidia_driver_version()
{
    string s;
    try {
        stringbuf buf;
        ifstream ifs("/proc/driver/nvidia/version");
        ifs.exceptions(std::ifstream::failbit|std::ifstream::badbit);
        ifs.get(buf, '\n');
        ifs.close();
        s = buf.str();
    }
    catch (ifstream::failure&) {
        throw runtime_error("failed to query NVIDIA driver version");
    }
    size_t pos = s.find(": ");
    if (pos != string::npos) {
        s = s.substr(pos + 2);
    }
    trim(s);
    return s;
}

/**
 * Query CUDA compute version
 */
string device::compute_version()
{
    cuda::vector<int> g_arch(1);
    cuda::host::vector<int> h_arch(1);
    cuda::configure(1, 1);
    device_wrapper::arch(g_arch);
    cuda::copy(g_arch, h_arch);
    int major = h_arch.front() / 100;
    int minor = h_arch.front() / 10 % 10;
    return lexical_cast<string>(major) + "." + lexical_cast<string>(minor);
}

#if CUDA_VERSION >= 2020

/**
 * Query CUDA driver version
 */
string device::cuda_driver_version()
{
    int major = cuda::driver::version() / 1000;
    int minor = cuda::driver::version() / 10 % 10;
    return lexical_cast<string>(major) + "." + lexical_cast<string>(minor);
}

/**
 * Query CUDA runtime version
 */
string device::cuda_runtime_version()
{
    int major = cuda::version() / 1000;
    int minor = cuda::version() / 10 % 10;
    return lexical_cast<string>(major) + "." + lexical_cast<string>(minor);
}

#endif /* CUDART_VERSION >= 2020 */

/**
 * Translate CUDA exception to Lua error message
 */
static void translate_cuda_error(lua_State* L, cuda::error const& e)
{
    lua_pushliteral(L, "[CUDA] ");
    lua_pushstring(L, e.what());
    lua_concat(L, 2);
}

void device::luaopen(lua_State* L)
{
    using namespace luabind;
    module(L)
    [
        namespace_("halmd_wrapper")
        [
            namespace_("utility")
            [
                namespace_("gpu")
                [
                    class_<device, shared_ptr<device> >("device")
                        .def(constructor<vector<int>, unsigned int>())
                        .property("threads", &device::threads)
                        .scope
                        [
                            def("nvidia_driver_version", &device::nvidia_driver_version)
                          , def("compute_version", &device::compute_version)
                          , def("cuda_driver_version", &device::cuda_driver_version)
                          , def("cuda_runtime_version", &device::cuda_runtime_version)
                        ]
                ]
            ]
        ]
    ];
    register_exception_handler<cuda::error>(&translate_cuda_error);
}

namespace // limit symbols to translation unit
{

__attribute__((constructor)) void register_lua()
{
    lua_wrapper::register_(0) //< distance of derived to base class
    [
        &device::luaopen
    ];
}

} // namespace

}} // namespace utility::gpu

} // namespace halmd