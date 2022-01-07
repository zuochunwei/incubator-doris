// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <ext/pb_ds/priority_queue.hpp>

#include "olap/olap_define.h"
#include "olap/reader.h"
#include "olap/rowset/rowset_reader.h"
#include "vec/core/block.h"

namespace doris {

class TabletSchema;

namespace vectorized {

struct IteratorRowRef {
    const Block* block;
    uint16_t row_pos;
    bool is_same;
};

class VCollectIterator {
public:
    ~VCollectIterator() { delete _inner_iter; }

    // Hold reader point to get reader params
    void init(Reader* reader);

    OLAPStatus add_child(RowsetReaderSharedPtr rs_reader);

    void build_heap(std::vector<RowsetReaderSharedPtr>& rs_readers);
    // Get top row of the heap, nullptr if reach end.
    OLAPStatus current_row(IteratorRowRef* ref) const {
        if (LIKELY(_inner_iter)) {
            *ref = *_inner_iter->current_row_ref();
            if (ref->row_pos == -1) {
                return OLAP_ERR_DATA_EOF;
            } else {
                return OLAP_SUCCESS;
            }
        }
        return OLAP_ERR_DATA_ROW_BLOCK_ERROR;
    }

    // Read nest order row in Block.
    // Returns
    //      OLAP_SUCCESS when read successfully.
    //      OLAP_ERR_DATA_EOF and set *row to nullptr when EOF is reached.
    //      Others when error happens
    OLAPStatus next(IteratorRowRef* ref) {
        if (LIKELY(_inner_iter)) {
            return _inner_iter->next(ref);
        } else {
            return OLAP_ERR_DATA_EOF;
        }
    }

    OLAPStatus next(Block* block) {
        if (LIKELY(_inner_iter)) {
            return _inner_iter->next(block);
        } else {
            return OLAP_ERR_DATA_EOF;
        }
    }

    bool is_merge() const { return _merge; }

private:
    // This interface is the actual implementation of the new version of iterator.
    // It currently contains two implementations, one is Level0Iterator,
    // which only reads data from the rowset reader, and the other is Level1Iterator,
    // which can read merged data from multiple LevelIterators through MergeHeap.
    // By using Level1Iterator, some rowset readers can be merged in advance and
    // then merged with other rowset readers.
    class LevelIterator {
    public:
        LevelIterator(Reader* reader) : num_key_columns(reader->tablet()->tablet_schema().num_key_columns()) {}

        virtual OLAPStatus init() = 0;

        virtual int64_t version() const = 0;

        const IteratorRowRef* current_row_ref() const { return &_ref; }

        virtual OLAPStatus next(IteratorRowRef* ref) = 0;

        virtual OLAPStatus next(Block* block) = 0;

        void set_same(bool same) { _ref.is_same = same; }

        bool is_same() const { return _ref.is_same; }

        virtual ~LevelIterator() = default;

        IteratorRowRef _ref;
        uint32_t num_key_columns = 0;
    };

    // Compare row cursors between multiple merge elements,
    // if row cursors equal, compare data version.
    class LevelIteratorComparator {
    public:
        LevelIteratorComparator(int sequence = -1) : _sequence(sequence) {}

        bool operator()(LevelIterator* lhs, LevelIterator* rhs) const;

    private:
        int _sequence;
    };

    using MergeHeap = __gnu_pbds::priority_queue<LevelIterator*, LevelIteratorComparator,
                                                 __gnu_pbds::pairing_heap_tag>;

    // Iterate from rowset reader. This Iterator usually like a leaf node
    class Level0Iterator : public LevelIterator {
    public:
        Level0Iterator(RowsetReaderSharedPtr rs_reader, Reader* reader);

        OLAPStatus init() override;

        int64_t version() const override { return _version; }

        OLAPStatus next(IteratorRowRef* ref) override {
            _ref.row_pos++;
            RETURN_NOT_OK(_refresh_current_row());
            *ref = _ref;
            return OLAP_SUCCESS;
        }

        OLAPStatus next(Block* block) override {
            return _rs_reader->next_block(block);
        }

    private:
        OLAPStatus _refresh_current_row();

        RowsetReaderSharedPtr _rs_reader;
        Reader* _reader = nullptr;
        Block _block;
        int64_t _version = 0;
        uint16_t _block_rows = 0;
    };

    // Iterate from LevelIterators (maybe Level0Iterators or Level1Iterator or mixed)
    class Level1Iterator : public LevelIterator {
    public:
        Level1Iterator(std::list<LevelIterator*>&& children, Reader* reader, bool merge,
                       bool skip_same);

        ~Level1Iterator() { 
            for (auto child : _children) 
                delete child; 
            delete _heap;
        }

        OLAPStatus init() override;

        int64_t version() const override {
            if (_cur_child != nullptr) {
                return _cur_child->version();
            }
            return -1;
        }

        OLAPStatus next(IteratorRowRef* ref) override {
            if (UNLIKELY(_cur_child == nullptr)) {
                _ref.row_pos = -1;
                return OLAP_ERR_DATA_EOF;
            }
            if (_merge) {
                return _merge_next(ref);
            } else {
                return _normal_next(ref);
            }
        }

        OLAPStatus next(Block* block) override {
            if (UNLIKELY(_cur_child == nullptr)) {
                return OLAP_ERR_DATA_EOF;
            }
            return _normal_next(block);
        }

    private:
        inline OLAPStatus _merge_next(IteratorRowRef* ref);

        inline OLAPStatus _normal_next(IteratorRowRef* ref);

        inline OLAPStatus _normal_next(Block* block);

        // Each LevelIterator corresponds to a rowset reader,
        // it will be cleared after '_heap' has been initilized when '_merge == true'.
        std::list<LevelIterator*> _children;
        // point to the Level0Iterator containing the next output row.
        // null when VCollectIterator hasn't been initialized or reaches EOF.
        LevelIterator* _cur_child = nullptr;
        Reader* _reader = nullptr;

        // when `_merge == true`, rowset reader returns ordered rows and VCollectIterator uses a priority queue to merge
        // sort them. The output of VCollectIterator is also ordered.
        // When `_merge == false`, rowset reader returns *partial* ordered rows. VCollectIterator simply returns all rows
        // from the first rowset, the second rowset, .., the last rowset. The output of CollectorIterator is also
        // *partially* ordered.
        bool _merge = true;

        bool _skip_same = false;
        // used when `_merge == true`
        MergeHeap* _heap = nullptr;
        // used when `_merge == false`
        int _child_idx = 0;
    };

    LevelIterator* _inner_iter = nullptr;

    // Each LevelIterator corresponds to a rowset reader,
    // it will be cleared after '_inner_iter' has been initilized.
    std::list<LevelIterator*> _children;

    bool _merge = true;
    // Hold reader point to access read params, such as fetch conditions.
    Reader* _reader = nullptr;

    bool _skip_same = false;
};

} // namespace vectorized
} // namespace doris
