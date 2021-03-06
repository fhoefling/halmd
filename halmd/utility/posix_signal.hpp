/*
 * Copyright © 2008-2012  Peter Colberg
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

#ifndef HALMD_UTILITY_POSIX_SIGNAL_HPP
#define HALMD_UTILITY_POSIX_SIGNAL_HPP

#include <halmd/utility/signal.hpp>

#include <lua.hpp>

#include <csignal>
#include <string>
#include <unordered_map>

namespace halmd {

/**
 * POSIX signal handler
 */
class posix_signal
{
private:
    typedef halmd::signal<void (int)> handler_type;

public:
    typedef handler_type::slot_function_type slot_function_type;

    connection on_signal(int signum, slot_function_type const& slot);
    void wait() const;
    bool poll() const;
    static std::string name(int signum);
    static void luaopen(lua_State* L);

private:
    typedef std::unordered_map<int, handler_type> handler_map_type;

    static sigset_t block_signals();
    static sigset_t set_;

    void handle(int signum) const;

    handler_map_type handler_;
};

} // namespace halmd

#endif /* ! HALMD_UTILITY_POSIX_SIGNAL_HPP */
