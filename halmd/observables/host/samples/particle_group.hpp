/*
 * Copyright © 2012  Felix Höfling
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

#ifndef HALMD_OBSERVABLES_HOST_SAMPLES_PARTICLE_GROUP_HPP
#define HALMD_OBSERVABLES_HOST_SAMPLES_PARTICLE_GROUP_HPP

#include <lua.hpp>
#include <utility>

#include <halmd/mdsim/host/particle.hpp>

namespace halmd {
namespace observables {
namespace host {
namespace samples {

/**
 * A particle group represents a subset of particles, which is defined here by
 * an instance of host::particle together with either a range of tags or by
 * selecting all.
 *
 * A tag range is a contiguous range of particle tags, specified in terms of
 * begin and end tags in analogy to iterator ranges, the particle with tag
 * 'begin' is included, while tag 'end' is not.
 *
 * The group represents a fixed order of the particles according to their tags
 * and starts with the smallest tag in the set.
 *
 */

template <int dimension, typename float_type>
class particle_group
{
public:
    typedef mdsim::host::particle<dimension, float_type> particle_type;
    typedef std::vector<unsigned int>::const_iterator map_iterator;

    static void luaopen(lua_State* L);

    particle_group() {}

    //! returns underlying particle instance
    virtual boost::shared_ptr<particle_type const> particle() const = 0;

    /**
     * returns iterator to an index array mapping particle tags to array
     * indices in host::particle
     */
    virtual map_iterator map() const = 0;

    //! returns the size of the group, i.e., the number of particles
    virtual unsigned int size() const = 0;

    //! returns true if the group is the empty set
    bool empty() const
    {
        return size() == 0;
    }
};

template <int dimension, typename float_type>
class particle_group_all
  : public particle_group<dimension, float_type>
{
public:
    typedef particle_group<dimension, float_type> _Base;
    typedef typename _Base::particle_type particle_type;
    typedef typename _Base::map_iterator map_iterator;

    static void luaopen(lua_State* L);

    particle_group_all(
        boost::shared_ptr<particle_type const> particle
    )
      : particle_(particle) {}

    virtual boost::shared_ptr<particle_type const> particle() const
    {
        return particle_;
    }

    virtual map_iterator map() const
    {
        return particle_->tag.begin(); // FIXME host::particle doesn't provide reverse_tag
    }

    //! returns size of the group, i.e., the number of particles
    virtual unsigned int size() const
    {
        return particle_->nbox;
    }

private:
    /** host::particle instance */
    boost::shared_ptr<particle_type const> particle_;
};

template <int dimension, typename float_type>
class particle_group_from_range
  : public particle_group<dimension, float_type>
{
public:
    typedef particle_group<dimension, float_type> _Base;
    typedef typename _Base::particle_type particle_type;
    typedef typename _Base::map_iterator map_iterator;

    static void luaopen(lua_State* L);

    particle_group_from_range(
        boost::shared_ptr<particle_type const> particle
      , unsigned int begin
      , unsigned int end
    );

    virtual boost::shared_ptr<particle_type const> particle() const
    {
        return particle_;
    }

    virtual map_iterator map() const
    {
        return particle_->tag.begin() + begin_; // FIXME host::particle doesn't provide reverse_tag
    }

    //! returns size of the group, i.e., the number of particles
    virtual unsigned int size() const
    {
        return end_ - begin_;
    }

private:
    /** host::particle instance */
    boost::shared_ptr<particle_type const> particle_;
    /** tag range [begin, end) */
    unsigned int begin_;
    unsigned int end_;
};
} // namespace samples
} // namespace host
} // namespace observables
} // namespace halmd

#endif /* ! HALMD_OBSERVABLES_HOST_SAMPLES_PARTICLE_GROUP_HPP */