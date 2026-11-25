// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_REFERENT_RESOLVER_H_
#define SPATIALBLEND_SPACE_REFERENT_RESOLVER_H_

#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/element.h"
#include "spatialblend/model/element/element_tree.h"

namespace spatialblend {

// Resolves an incoming payload's shared referent id to a local element.
// Default: exact shared_id match. Future extensions — semantic-compatibility
// matching and virtual-object materialization — plug in here by
// overriding Resolve, without touching the solver or Space call sites.
class ReferentResolver {
 public:
  virtual ~ReferentResolver() = default;

  virtual const Element* Resolve(shared_element_id_t referent_shared_id,
                                 const ElementTree& tree) const {
    return tree.GetElementBySharedId(referent_shared_id);
  }
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_REFERENT_RESOLVER_H_
