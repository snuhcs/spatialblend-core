// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_NETWORK_PROTO_UTIL_H_
#define SPATIALBLEND_NETWORK_PROTO_UTIL_H_

#include <optional>

#include "spatialblend/net/proto/spatialblend.pb.h"
#include "spatialblend/model/relation/support.h"
#include "spatialblend/space/relation_event.h"
#include "spatialblend/model/realization/remote_update.h"

// The protobuf package keeps its pre-1.0 name because it is on the wire (gRPC method paths, Any type
// URLs): spatialblend_v2.proto. C++ code names it through this alias.
namespace spatialblend_v2::proto {}
namespace spatialblend {
namespace proto = ::spatialblend_v2::proto;
}  // namespace spatialblend

namespace spatialblend {

// RemoteRelationUpdate <-> RelationUpdateProto
proto::RelationUpdateProto ToProto(const RemoteRelationUpdate& update);
RemoteRelationUpdate FromProto(const proto::RelationUpdateProto& proto);

// ElementEvent <-> ElementEventProto
proto::ElementEventProto ToProto(const ElementEvent& event);
ElementEvent FromProto(const proto::ElementEventProto& proto);

// Support <-> SupportProto
proto::SupportProto ToProto(const Support& support);
Support SupportFromProto(const proto::SupportProto& proto);

// RelationPayload <-> RelationPayloadProto
// Legacy peers cannot represent native disable or observed-point lifetimes.
// Rejection is explicit; never encode either as usable ordinary geometry.
std::optional<proto::RelationPayloadProto> TryToProtoLegacy(
    const RelationPayload& payload);
RelationPayload PayloadFromProto(const proto::RelationPayloadProto& proto);

// RelationEvent <-> RelationEventProto
proto::RelationEventProto ToProto(const RelationEvent& event);
RelationEvent FromProto(const proto::RelationEventProto& proto);

}  // namespace spatialblend

#endif  // SPATIALBLEND_NETWORK_PROTO_UTIL_H_
