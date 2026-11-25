// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// T1.1: the public C API signal Unity will use for "interacting".
//
// Two sessions in one process (one sb_session_t may host or connect, never
// both). The host shares a room, an object, and an avatar element. The
// connected space receives that avatar, binds Spacing on it, and listens with
// sb_register_on_relation_event. As the source avatar approaches the shared
// object the phase must go DORMANT -> ACTIVE (or higher). DRIVING then
// persisting while the avatar moves away is by design: StepRelationPhase has
// no proximity path from DRIVING or CONTACT back to DORMANT, and a lone
// referent keeps weight 1, so falling R only steps CONTACT -> DRIVING.
// sb_get_relation_states_v2 on the received actor must still show that lower
// R and agree with the source. Events and queries both report
// SB_KERNEL_SPACING.
//
// The source also carries a Spacing binding: that is what computes the
// relation and sends it. The receiver binding is the one this test queries.

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "spatialblend/api/c/c_test_embodied.h"
#include "spatialblend/api/c/spatialblend.h"

namespace {

constexpr int64_t kRoomShared = 100;
constexpr int64_t kObjectShared = 200;
constexpr int64_t kActorShared = 300;
// Local table the receiver already owns. LinkToExisting maps the source
// object onto this element, so the two shared ids differ.
constexpr int64_t kTableShared = 900;

// 100 m square. Diagonal is 100*sqrt(2) ~= 141.42 m, which is the scope
// diameter Spacing normalizes by. The object sits in the far corner.
// (1, 1) is ~135.8 m from that corner: R ~= 0.040, below the 0.05 release.
// (98.2, 98.2) is inside the object footprint: R = 1.
constexpr double kFarX = 1.0;
constexpr double kFarY = 1.0;
constexpr double kNearX = 98.2;
constexpr double kNearY = 98.2;
constexpr double kOutsideX = -20.0;
constexpr double kOutsideY = -20.0;

const char* PhaseName(sb_relation_phase_t phase) {
  switch (phase) {
    case SB_PHASE_DORMANT:
      return "DORMANT";
    case SB_PHASE_ACTIVE:
      return "ACTIVE";
    case SB_PHASE_DRIVING:
      return "DRIVING";
    case SB_PHASE_CONTACT:
      return "CONTACT";
  }
  return "UNKNOWN";
}

int32_t PickFreePort() {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return 0;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(0);
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return 0;
  }
  sockaddr_in bound{};
  socklen_t len = sizeof(bound);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &len) != 0) {
    close(fd);
    return 0;
  }
  const int32_t port = ntohs(bound.sin_port);
  close(fd);
  return port;
}

void SetSquare(sb_space_t space, sb_element_t elem, sb_geometry_role_t role,
               double x0, double y0, double x1, double y1) {
  const double poly[] = {x0, y0, x1, y0, x1, y1, x0, y1};
  sb_set_element_polygon(space, elem, role, poly, 4);
}

struct AddedLog {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::pair<int64_t, sb_element_t>> items;
};

void OnElementAdded(sb_space_t, sb_element_t local, int64_t shared_id,
                    const char*, const void*, int64_t, void* user) {
  auto* log = static_cast<AddedLog*>(user);
  std::lock_guard<std::mutex> lock(log->mu);
  log->items.emplace_back(shared_id, local);
  log->cv.notify_all();
}

sb_element_t WaitForShared(AddedLog* log, int64_t shared_id, int timeout_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  std::unique_lock<std::mutex> lock(log->mu);
  while (std::chrono::steady_clock::now() < deadline) {
    for (const auto& item : log->items) {
      if (item.first == shared_id) return item.second;
    }
    log->cv.wait_until(lock, deadline);
  }
  for (const auto& item : log->items) {
    if (item.first == shared_id) return item.second;
  }
  return 0;
}

struct QueryRow {
  int32_t count = -1;
  int32_t v1_count = -1;
  bool found = false;
  bool v1_found = false;
  sb_relation_phase_t phase = SB_PHASE_DORMANT;
  int32_t active = 0;
  double proximity = 0;
  int64_t referent_shared_id = 0;
  sb_kernel_type_t kernel = SB_KERNEL_FACING;
  sb_kernel_type_t v1_kernel = SB_KERNEL_FACING;
};

// The object row, whatever kernel the API attached. Filtering on
// SB_KERNEL_SPACING here would hide a kernel-id regression.
QueryRow QuerySpacing(sb_space_t space, sb_actor_t actor) {
  QueryRow row;
  sb_relation_state_v2_t states[8];
  std::memset(states, 0, sizeof(states));
  for (auto& state : states) {
    state.struct_size = sizeof(state);
  }
  row.count =
      sb_get_relation_states_v2(space, actor, states, 8, sizeof(states[0]));
  sb_relation_state_t v1[8];
  std::memset(v1, 0, sizeof(v1));
  row.v1_count = sb_get_relation_states(space, actor, v1, 8);
  const auto take = [](int64_t referent, bool already) {
    return referent == kObjectShared || (!already && referent == 0);
  };
  const int32_t n = row.count < 8 ? row.count : 8;
  for (int32_t i = 0; i < n; ++i) {
    if (!take(states[i].referent_shared_id, row.found)) continue;
    row.found = true;
    row.phase = states[i].phase;
    row.active = states[i].active;
    row.proximity = states[i].proximity;
    row.referent_shared_id = states[i].referent_shared_id;
    row.kernel = states[i].kernel_type;
    if (states[i].referent_shared_id == kObjectShared) break;
  }
  const int32_t n1 = row.v1_count < 8 ? row.v1_count : 8;
  for (int32_t i = 0; i < n1; ++i) {
    if (!take(v1[i].referent_shared_id, row.v1_found)) continue;
    row.v1_found = true;
    row.v1_kernel = v1[i].kernel_type;
    if (v1[i].referent_shared_id == kObjectShared) break;
  }
  return row;
}

struct ObservedEvent {
  std::string where;
  sb_relation_phase_t from = SB_PHASE_DORMANT;
  sb_relation_phase_t to = SB_PHASE_DORMANT;
  sb_kernel_type_t kernel = SB_KERNEL_SPACING;
  int64_t actor_shared_id = 0;
  int64_t referent_shared_id = 0;
  QueryRow query;
};

struct EventLog {
  std::string where;
  sb_space_t space = 0;
  sb_actor_t actor = 0;
  mutable std::mutex mu;
  std::vector<ObservedEvent> events;
};

void OnRelation(const sb_relation_event_t* event, void* user) {
  auto* log = static_cast<EventLog*>(user);
  ObservedEvent seen;
  seen.where = log->where;
  seen.from = event->from_phase;
  seen.to = event->to_phase;
  seen.kernel = event->kernel_type;
  seen.actor_shared_id = event->actor_shared_id;
  seen.referent_shared_id = event->referent_shared_id;
  if (log->actor != 0) seen.query = QuerySpacing(log->space, log->actor);
  std::lock_guard<std::mutex> lock(log->mu);
  log->events.push_back(seen);
}

std::string Format(const EventLog& log) {
  std::lock_guard<std::mutex> lock(log.mu);
  std::ostringstream out;
  out << log.where << " events (" << log.events.size() << "):\n";
  for (const auto& event : log.events) {
    out << "  " << PhaseName(event.from) << " -> " << PhaseName(event.to)
        << " kernel=" << static_cast<int>(event.kernel)
        << " actor=" << event.actor_shared_id
        << " referent=" << event.referent_shared_id
        << " query_count=" << event.query.count
        << " v1_count=" << event.query.v1_count
        << " query_phase=" << PhaseName(event.query.phase)
        << " active=" << event.query.active << " R=" << event.query.proximity
        << "\n";
  }
  return out.str();
}

std::string FormatRow(const char* label, const QueryRow& row) {
  std::ostringstream out;
  out << label << " count=" << row.count << " v1=" << row.v1_count
      << " found=" << row.found << " phase=" << PhaseName(row.phase)
      << " active=" << row.active << " R=" << row.proximity
      << " referent=" << row.referent_shared_id
      << " kernel=" << static_cast<int>(row.kernel)
      << " v1_kernel=" << static_cast<int>(row.v1_kernel);
  return out.str();
}

bool WaitApplied(sb_space_t receiver, sb_space_t source, sb_actor_t actor,
                 int64_t sequence, int timeout_ms) {
  if (sequence <= 0) return false;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (sb_get_last_applied_remote_sequence(receiver, source, actor) >=
        sequence) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return sb_get_last_applied_remote_sequence(receiver, source, actor) >=
         sequence;
}

// Propagate one source pose and wait until the receiver applies it, when the
// source actually emits. A dormant-only sample is allowed to emit nothing.
void Drive(sb_space_t source, sb_actor_t source_actor, sb_space_t receiver,
           double x, double y) {
  const int64_t before =
      sb_get_last_emitted_update_sequence(source, source_actor);
  sb_update_actor_state(source, source_actor, x, y, 0.0);
  const int64_t after =
      sb_get_last_emitted_update_sequence(source, source_actor);
  if (after > before) {
    WaitApplied(receiver, source, source_actor, after, 2000);
  }
}

struct Sessions {
  sb_session_t host = 0;
  sb_session_t client = 0;
  sb_space_t host_space = 0;
  sb_space_t client_space = 0;
  ~Sessions() {
    if (client_space) {
      sb_register_on_relation_event(client_space, nullptr, nullptr);
      sb_register_on_element_added(client_space, nullptr, nullptr);
    }
    if (host_space) sb_register_on_relation_event(host_space, nullptr, nullptr);
    if (client) sb_destroy_session(client);
    if (host) sb_destroy_session(host);
  }
};

sb_element_t ShareAndReceive(sb_space_t host, sb_element_t elem, AddedLog* log,
                             int64_t shared_id) {
  sb_share_element(host, elem, nullptr, 0);
  return WaitForShared(log, shared_id, 2000);
}

struct LinkedActors {
  sb_actor_t source = 0;
  sb_actor_t remote = 0;
  sb_element_t object = 0;
  sb_element_t table = 0;
};

// Host shares an object. The client links that share onto a pre-existing
// table whose shared id is not the source object's, and binds Spacing on
// the table. Resolve mode is forced to linker so an env override cannot
// hide the source-id / local-id split.
::testing::AssertionResult PrepareLinkedTable(Sessions* sessions,
                                              AddedLog* added,
                                              LinkedActors* out) {
  sessions->host = sb_create_session();
  sessions->client = sb_create_session();
  if (!sessions->host || !sessions->client) {
    return ::testing::AssertionFailure() << "session create failed";
  }
  sessions->host_space = sb_add_space(sessions->host);
  sessions->client_space = sb_add_space(sessions->client);
  if (!sessions->host_space || !sessions->client_space) {
    return ::testing::AssertionFailure() << "space create failed";
  }

  const int32_t port = PickFreePort();
  if (port <= 0) return ::testing::AssertionFailure() << "no free port";
  const sb_handle_t host_handle = sb_host_space(sessions->host_space, port);
  if (!host_handle || !sb_try_wait(host_handle, 2000)) {
    return ::testing::AssertionFailure() << "host failed on port " << port;
  }
  const sb_handle_t connect_handle =
      sb_connect_space(sessions->client_space, "127.0.0.1", port);
  if (!connect_handle || !sb_try_wait(connect_handle, 5000)) {
    return ::testing::AssertionFailure() << "connect failed on port " << port;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  sb_register_on_element_added(sessions->client_space, OnElementAdded, added);
  sb_set_referent_resolve_mode(sessions->client_space,
                               SB_REFERENT_RESOLVE_LINKER);

  const sb_element_t room = sb_add_element(sessions->host_space, 0);
  sb_set_element_shared_id(sessions->host_space, room, kRoomShared);
  SetSquare(sessions->host_space, room, SB_GEOMETRY_NAVIGABLE, 0, 0, 100, 100);

  out->object = sb_add_element(sessions->host_space, room);
  sb_set_element_shared_id(sessions->host_space, out->object, kObjectShared);
  sb_set_element_tag(sessions->host_space, out->object, "Object");
  SetSquare(sessions->host_space, out->object, SB_GEOMETRY_CONTAINMENT, 97, 97,
            99.5, 99.5);

  const sb_element_t actor_elem = sb_add_element(sessions->host_space, room);
  sb_set_element_shared_id(sessions->host_space, actor_elem, kActorShared);
  out->source = sb_create_actor(sessions->host_space);
  if (!out->source) return ::testing::AssertionFailure() << "no source actor";
  sb_add_anchor(sessions->host_space, out->source, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(sessions->host_space, out->source, actor_elem);
  if (sb_bind(sessions->host_space, out->source, "root",
                     SB_KERNEL_SPACING) != 1) {
    return ::testing::AssertionFailure() << "source bind failed";
  }
  sb_set_min_move(sessions->host_space, 0.0);
  sb_set_min_yaw(sessions->host_space, 0.0);

  const sb_element_t rx_room =
      ShareAndReceive(sessions->host_space, room, added, kRoomShared);
  if (!rx_room) return ::testing::AssertionFailure() << "room was not received";
  sb_resolve_shared_element(sessions->client_space, rx_room, rx_room);
  SetSquare(sessions->client_space, rx_room, SB_GEOMETRY_NAVIGABLE, 0, 0, 100,
            100);

  out->table = sb_add_element(sessions->client_space, rx_room);
  if (!out->table) return ::testing::AssertionFailure() << "no local table";
  sb_set_element_shared_id(sessions->client_space, out->table, kTableShared);
  sb_set_element_tag(sessions->client_space, out->table, "Table");
  SetSquare(sessions->client_space, out->table, SB_GEOMETRY_CONTAINMENT, 97, 97,
            99.5, 99.5);

  const sb_element_t rx_object =
      ShareAndReceive(sessions->host_space, out->object, added, kObjectShared);
  if (!rx_object || rx_object == out->table) {
    return ::testing::AssertionFailure() << "object was not received as a temp";
  }
  sb_resolve_shared_element(sessions->client_space, rx_object, out->table);

  const sb_element_t rx_actor_elem =
      ShareAndReceive(sessions->host_space, actor_elem, added, kActorShared);
  if (!rx_actor_elem) {
    return ::testing::AssertionFailure() << "avatar element was not received";
  }
  sb_resolve_shared_element(sessions->client_space, rx_actor_elem,
                            rx_actor_elem);

  out->remote = sb_create_actor(sessions->client_space);
  if (!out->remote) return ::testing::AssertionFailure() << "no remote actor";
  sb_add_anchor(sessions->client_space, out->remote, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(sessions->client_space, out->remote, rx_actor_elem);
  if (sb_bind(sessions->client_space, out->remote, "root",
                     SB_KERNEL_SPACING) != 1) {
    return ::testing::AssertionFailure() << "receiver bind failed";
  }
  return ::testing::AssertionSuccess();
}

void Approach(sb_space_t source_space, sb_actor_t source_actor,
              sb_space_t receiver) {
  Drive(source_space, source_actor, receiver, kFarX, kFarY);
  for (int i = 1; i <= 8; ++i) {
    const double a = i / 8.0;
    Drive(source_space, source_actor, receiver, kFarX * (1.0 - a) + kNearX * a,
          kFarY * (1.0 - a) + kNearY * a);
  }
  for (int i = 0; i < 3; ++i) {
    Drive(source_space, source_actor, receiver, kNearX, kNearY);
  }
}

}  // namespace

TEST(RelationEventCTest, ReceivedActorSpacingApproachesAndLeaves) {
  // Logs outlive the sessions: a callback can still be in flight until the
  // session destructor clears it.
  EventLog receiver_log;
  EventLog source_log;
  AddedLog added;
  receiver_log.where = "receiver";
  source_log.where = "source";
  Sessions sessions;

  sessions.host = sb_create_session();
  sessions.client = sb_create_session();
  ASSERT_NE(sessions.host, 0);
  ASSERT_NE(sessions.client, 0);
  sessions.host_space = sb_add_space(sessions.host);
  sessions.client_space = sb_add_space(sessions.client);
  ASSERT_NE(sessions.host_space, 0);
  ASSERT_NE(sessions.client_space, 0);

  // Host and connect while both spaces are blank, so bootstrap does not
  // clone a scene the client will author itself.
  const int32_t port = PickFreePort();
  ASSERT_GT(port, 0);
  const sb_handle_t host_handle = sb_host_space(sessions.host_space, port);
  ASSERT_NE(host_handle, 0);
  ASSERT_NE(sb_try_wait(host_handle, 2000), 0);
  const sb_handle_t connect_handle =
      sb_connect_space(sessions.client_space, "127.0.0.1", port);
  ASSERT_NE(connect_handle, 0) << "sb_connect_space failed on port " << port;
  ASSERT_NE(sb_try_wait(connect_handle, 5000), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  sb_register_on_element_added(sessions.client_space, OnElementAdded, &added);

  const sb_element_t room = sb_add_element(sessions.host_space, 0);
  sb_set_element_shared_id(sessions.host_space, room, kRoomShared);
  SetSquare(sessions.host_space, room, SB_GEOMETRY_NAVIGABLE, 0, 0, 100, 100);

  const sb_element_t object = sb_add_element(sessions.host_space, room);
  sb_set_element_shared_id(sessions.host_space, object, kObjectShared);
  sb_set_element_tag(sessions.host_space, object, "Object");
  SetSquare(sessions.host_space, object, SB_GEOMETRY_CONTAINMENT, 97, 97, 99.5,
            99.5);

  const sb_element_t actor_elem = sb_add_element(sessions.host_space, room);
  sb_set_element_shared_id(sessions.host_space, actor_elem, kActorShared);
  const sb_actor_t source = sb_create_actor(sessions.host_space);
  spatialblend_c_test::PublishStartPose(sessions.host_space, source);
  ASSERT_NE(source, 0);
  sb_add_anchor(sessions.host_space, source, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(sessions.host_space, source, actor_elem);
  ASSERT_EQ(sb_bind(sessions.host_space, source, "root",
                           SB_KERNEL_SPACING),
            1);
  sb_set_min_move(sessions.host_space, 0.0);
  sb_set_min_yaw(sessions.host_space, 0.0);

  const sb_element_t rx_room =
      ShareAndReceive(sessions.host_space, room, &added, kRoomShared);
  ASSERT_NE(rx_room, 0) << "room was not received";
  sb_resolve_shared_element(sessions.client_space, rx_room, rx_room);
  SetSquare(sessions.client_space, rx_room, SB_GEOMETRY_NAVIGABLE, 0, 0, 100,
            100);

  const sb_element_t rx_object =
      ShareAndReceive(sessions.host_space, object, &added, kObjectShared);
  ASSERT_NE(rx_object, 0) << "object was not received";
  sb_resolve_shared_element(sessions.client_space, rx_object, rx_object);
  SetSquare(sessions.client_space, rx_object, SB_GEOMETRY_CONTAINMENT, 97, 97,
            99.5, 99.5);
  sb_set_element_tag(sessions.client_space, rx_object, "Object");

  const sb_element_t rx_actor_elem =
      ShareAndReceive(sessions.host_space, actor_elem, &added, kActorShared);
  ASSERT_NE(rx_actor_elem, 0) << "avatar element was not received";
  sb_resolve_shared_element(sessions.client_space, rx_actor_elem,
                            rx_actor_elem);

  const sb_actor_t remote = sb_create_actor(sessions.client_space);
  spatialblend_c_test::PublishStartPose(sessions.client_space, remote);
  ASSERT_NE(remote, 0);
  sb_add_anchor(sessions.client_space, remote, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(sessions.client_space, remote, rx_actor_elem);
  ASSERT_EQ(sb_find_actor_by_shared_id(sessions.client_space, kActorShared),
            remote);
  ASSERT_EQ(sb_bind(sessions.client_space, remote, "root",
                           SB_KERNEL_SPACING),
            1);

  receiver_log.space = sessions.client_space;
  receiver_log.actor = remote;
  source_log.space = sessions.host_space;
  source_log.actor = source;
  sb_register_on_relation_event(sessions.client_space, OnRelation,
                                &receiver_log);
  sb_register_on_relation_event(sessions.host_space, OnRelation, &source_log);

  Drive(sessions.host_space, source, sessions.client_space, kFarX, kFarY);
  for (int i = 1; i <= 8; ++i) {
    const double a = i / 8.0;
    Drive(sessions.host_space, source, sessions.client_space,
          kFarX * (1.0 - a) + kNearX * a, kFarY * (1.0 - a) + kNearY * a);
  }
  // Hold inside the object so the phase chain can take its one step per
  // sample (ACTIVE may jump to CONTACT when R >= 0.95) before withdrawal.
  for (int i = 0; i < 3; ++i) {
    Drive(sessions.host_space, source, sessions.client_space, kNearX, kNearY);
  }
  const QueryRow after_approach = QuerySpacing(sessions.client_space, remote);
  const QueryRow source_after_approach =
      QuerySpacing(sessions.host_space, source);

  for (int i = 1; i <= 8; ++i) {
    const double a = i / 8.0;
    Drive(sessions.host_space, source, sessions.client_space,
          kNearX * (1.0 - a) + kFarX * a, kNearY * (1.0 - a) + kFarY * a);
  }
  for (int i = 0; i < 4; ++i) {
    Drive(sessions.host_space, source, sessions.client_space, kOutsideX,
          kOutsideY);
  }
  const QueryRow after_leave = QuerySpacing(sessions.client_space, remote);
  const QueryRow source_after_leave = QuerySpacing(sessions.host_space, source);

  std::vector<ObservedEvent> receiver_events;
  {
    std::lock_guard<std::mutex> lock(receiver_log.mu);
    receiver_events = receiver_log.events;
  }

  // The binding under test is Spacing on shared object 200. Kernel id is
  // checked separately from the phase, on every object event and on both
  // query APIs, so a raw C++ enum cast cannot pass as SB_KERNEL_SPACING.
  bool saw_approach = false;
  bool approach_query_agreed = false;
  bool event_kernel_is_spacing = true;
  bool event_query_kernel_is_spacing = true;
  int object_events = 0;
  for (int i = 0; i < static_cast<int>(receiver_events.size()); ++i) {
    const auto& event = receiver_events[i];
    if (event.referent_shared_id != kObjectShared) continue;
    ++object_events;
    if (event.kernel != SB_KERNEL_SPACING) event_kernel_is_spacing = false;
    if (!event.query.found || event.query.kernel != SB_KERNEL_SPACING ||
        !event.query.v1_found || event.query.v1_kernel != SB_KERNEL_SPACING) {
      event_query_kernel_is_spacing = false;
    }
    if (!saw_approach && event.from == SB_PHASE_DORMANT &&
        event.to >= SB_PHASE_ACTIVE) {
      saw_approach = true;
      approach_query_agreed =
          event.query.found && event.query.phase == event.to &&
          event.query.phase >= SB_PHASE_ACTIVE && event.query.active != 0 &&
          event.query.referent_shared_id == kObjectShared;
    }
  }

  std::ostringstream report;
  report << "object events=" << object_events << "\n"
         << Format(receiver_log) << Format(source_log)
         << FormatRow("receiver after approach", after_approach) << "\n"
         << FormatRow("source after approach", source_after_approach) << "\n"
         << FormatRow("receiver after leave", after_leave) << "\n"
         << FormatRow("source after leave", source_after_leave) << "\n";
  SCOPED_TRACE(report.str());

  EXPECT_GT(object_events, 0);
  EXPECT_TRUE(saw_approach)
      << "receiver Spacing goes DORMANT -> ACTIVE (or higher) as the source "
         "avatar approaches the shared object";
  EXPECT_TRUE(event_kernel_is_spacing)
      << "sb_relation_event_t.kernel_type is SB_KERNEL_SPACING";
  EXPECT_TRUE(event_query_kernel_is_spacing)
      << "sb_get_relation_states and sb_get_relation_states_v2 report "
         "SB_KERNEL_SPACING for the object row";
  EXPECT_TRUE(approach_query_agreed)
      << "sb_get_relation_states_v2 on the received actor matches the "
         "approach event";
  EXPECT_EQ(after_approach.kernel, SB_KERNEL_SPACING);
  EXPECT_EQ(after_approach.v1_kernel, SB_KERNEL_SPACING);
  EXPECT_EQ(after_leave.kernel, SB_KERNEL_SPACING);
  EXPECT_EQ(after_leave.v1_kernel, SB_KERNEL_SPACING);

  // DRIVING persisting after the avatar leaves is by design. StepRelationPhase
  // has no proximity path from DRIVING or CONTACT back to DORMANT, and a lone
  // referent keeps weight 1, so falling R steps CONTACT -> DRIVING and stays
  // there. The receiver query must show that lower R and agree with the source.
  EXPECT_TRUE(after_leave.found);
  EXPECT_TRUE(source_after_leave.found);
  EXPECT_LT(after_leave.proximity, after_approach.proximity)
      << "receiver R falls after the avatar leaves the object";
  EXPECT_NEAR(after_leave.proximity, source_after_leave.proximity, 1e-6);
  EXPECT_NEAR(after_approach.proximity, source_after_approach.proximity, 1e-6);
}

TEST(RelationEventCTest, LinkedReferentQueryUsesSourceSharedId) {
  // The table is a pre-existing element (shared id 900). LinkToExisting maps
  // the source object (shared id 200) onto it. Events key the phase on 200.
  // The query must use that same id, or it reports DORMANT.
  AddedLog added;
  Sessions sessions;
  LinkedActors actors;
  ASSERT_TRUE(PrepareLinkedTable(&sessions, &added, &actors));

  EventLog receiver_log;
  receiver_log.where = "receiver";
  receiver_log.space = sessions.client_space;
  receiver_log.actor = actors.remote;
  sb_register_on_relation_event(sessions.client_space, OnRelation,
                                &receiver_log);

  Approach(sessions.host_space, actors.source, sessions.client_space);

  std::vector<ObservedEvent> events;
  {
    std::lock_guard<std::mutex> lock(receiver_log.mu);
    events = receiver_log.events;
  }
  bool agreed = false;
  for (const auto& event : events) {
    if (event.referent_shared_id != kObjectShared) continue;
    if (event.kernel != SB_KERNEL_SPACING) continue;
    if (event.from != SB_PHASE_DORMANT || event.to < SB_PHASE_ACTIVE) continue;
    agreed = event.query.found && event.query.phase == event.to &&
             event.query.phase != SB_PHASE_DORMANT && event.query.active != 0 &&
             event.query.referent_shared_id == kObjectShared;
    if (agreed) break;
  }
  const QueryRow after = QuerySpacing(sessions.client_space, actors.remote);
  SCOPED_TRACE(Format(receiver_log) + "\n" + FormatRow("after approach", after));
  EXPECT_TRUE(agreed)
      << "receiver query phase matches the ACTIVE event for a linked referent";
  EXPECT_TRUE(after.found);
  EXPECT_NE(after.phase, SB_PHASE_DORMANT);
  EXPECT_EQ(after.referent_shared_id, kObjectShared);
  EXPECT_NE(after.referent_shared_id, kTableShared);
}

TEST(RelationEventCTest, UnsharedLinkedReferentClearsRows) {
  AddedLog added;
  Sessions sessions;
  LinkedActors actors;
  ASSERT_TRUE(PrepareLinkedTable(&sessions, &added, &actors));

  Approach(sessions.host_space, actors.source, sessions.client_space);
  const QueryRow before = QuerySpacing(sessions.client_space, actors.remote);
  ASSERT_TRUE(before.found) << FormatRow("before unshare", before);
  ASSERT_GT(before.count, 0);

  sb_unshare_element(sessions.host_space, actors.object);
  QueryRow after;
  bool cleared = false;
  for (int i = 0; i < 40 && !cleared; ++i) {
    Drive(sessions.host_space, actors.source, sessions.client_space, kNearX,
          kNearY);
    after = QuerySpacing(sessions.client_space, actors.remote);
    cleared = after.count == 0 && after.v1_count == 0;
    if (!cleared) {
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
  }
  SCOPED_TRACE(FormatRow("before unshare", before) + "\n" +
               FormatRow("after unshare", after));
  EXPECT_TRUE(cleared)
      << "an unshared linked referent leaves the receiver with 0 rows";
}

// ABI 4.9: once a link ends (the receiver unlinks, or the sharer
// unshares) relations from the retired source never resolve onto the receiver's
// element again. The element deliberately carries the SAME shared id as the
// source, so the shared-id fallback would find it if the retired source were
// allowed to fall back.
namespace {

enum class EndOfLink { kReceiverUnlinks, kSharerUnshares };

std::atomic<int> g_object_events{0};
void CountObjectEvents(const sb_relation_event_t* event, void*) {
  if (event->referent_shared_id == kObjectShared) g_object_events.fetch_add(1);
}

void ExpectRelationsStopResolving(EndOfLink how) {
  AddedLog added;
  Sessions sessions;
  LinkedActors actors;
  ASSERT_TRUE(PrepareLinkedTable(&sessions, &added, &actors));
  sb_set_element_shared_id(sessions.client_space, actors.table, kObjectShared);
  // An element with no shared id: the lookup for "no shared id" (-1) would find
  // it, so a retired referent must be dropped rather than looked up.
  ASSERT_NE(sb_add_element(sessions.client_space, actors.table), 0);
  g_object_events = 0;
  sb_register_on_relation_event(sessions.client_space, CountObjectEvents,
                                nullptr);

  Approach(sessions.host_space, actors.source, sessions.client_space);
  const QueryRow near_row = QuerySpacing(sessions.client_space, actors.remote);
  ASSERT_TRUE(near_row.found);
  for (int i = 0; i < 3; ++i) {
    Drive(sessions.host_space, actors.source, sessions.client_space, kFarX,
          kFarY);
  }
  const QueryRow far_row = QuerySpacing(sessions.client_space, actors.remote);
  ASSERT_TRUE(far_row.found);
  ASSERT_LT(far_row.proximity, near_row.proximity)
      << "control: the linked relation follows the source";
  Drive(sessions.host_space, actors.source, sessions.client_space, kNearX,
        kNearY);
  const QueryRow before = QuerySpacing(sessions.client_space, actors.remote);

  const int events_while_linked = g_object_events.load();
  EXPECT_GT(events_while_linked, 0) << "control: the linked relation acts";
  if (how == EndOfLink::kReceiverUnlinks) {
    ASSERT_EQ(sb_unlink(sessions.client_space, actors.table), 1);
  } else {
    sb_unshare_element(sessions.host_space, actors.object);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }

  for (int i = 0; i < 3; ++i) {
    Drive(sessions.host_space, actors.source, sessions.client_space, kFarX,
          kFarY);
  }
  const QueryRow after = QuerySpacing(sessions.client_space, actors.remote);
  EXPECT_EQ(g_object_events.load(), events_while_linked)
      << "a retired source still drove a phase change on the receiver";
  sb_register_on_relation_event(sessions.client_space, nullptr, nullptr);
  EXPECT_FALSE(after.found && after.proximity != before.proximity)
      << FormatRow("before", before) << " " << FormatRow("after", after)
      << ": a relation from a retired source still moved the receiver";
}

}  // namespace

TEST(RelationEventCTest, RelationsStopResolvingAfterTheReceiverUnlinks) {
  ExpectRelationsStopResolving(EndOfLink::kReceiverUnlinks);
}

TEST(RelationEventCTest, RelationsStopResolvingAfterTheSharerUnshares) {
  ExpectRelationsStopResolving(EndOfLink::kSharerUnshares);
}
