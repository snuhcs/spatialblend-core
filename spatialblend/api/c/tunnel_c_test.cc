// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// ABI 4.5: the tunneled core link. Two or more spaces are linked ONLY through
// sb_tunnel_write / sb_tunnel_read, pumped in this test by a thread that stands
// in for the application's carrier (Fusion reliable data in the Unity app).
// No TCP listener is opened and no address is dialed anywhere in this file:
// nothing here calls sb_host_space, sb_connect_space, or a socket function.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "spatialblend/api/c/c_test_embodied.h"
#include "spatialblend/api/c/spatialblend.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kRoomShared = 100;
constexpr int64_t kObjectShared = 200;
constexpr int64_t kActorShared = 300;
constexpr int64_t kGlbFromHost = 710;
constexpr int64_t kGlbFromClient = 720;
constexpr int64_t kSmallShared = 730;

void SetSquare(sb_space_t space, sb_element_t elem, sb_geometry_role_t role,
               double x0, double y0, double x1, double y1) {
  const double poly[] = {x0, y0, x1, y0, x1, y1, x0, y1};
  sb_set_element_polygon(space, elem, role, poly, 4);
}

// A deterministic payload that is not periodic at any small stride.
std::vector<uint8_t> MakePayload(size_t n, uint32_t seed) {
  std::vector<uint8_t> bytes(n);
  uint32_t x = seed * 2654435761u + 1;
  for (size_t i = 0; i < n; ++i) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    bytes[i] = static_cast<uint8_t>(x >> 11);
  }
  return bytes;
}

struct Received {
  sb_element_t local = 0;
  std::vector<uint8_t> payload;
};

// What a space has been told about, by shared id.
struct AddedLog {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::pair<int64_t, Received>> items;
};

void OnElementAdded(sb_space_t, sb_element_t local, int64_t shared_id,
                    const char*, const void* data, int64_t size, void* user) {
  auto* log = static_cast<AddedLog*>(user);
  Received r;
  r.local = local;
  if (data && size > 0) {
    r.payload.assign(static_cast<const uint8_t*>(data),
                     static_cast<const uint8_t*>(data) + size);
  }
  std::lock_guard<std::mutex> lock(log->mu);
  log->items.emplace_back(shared_id, std::move(r));
  log->cv.notify_all();
}

// What a space has been told about links the peer removed (ABI 4.9).
struct LinkLog {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::pair<sb_element_t, int32_t>> items;  // local, reason
  bool WaitFor(size_t n, int timeout_ms = 10000) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                       [&] { return items.size() >= n; });
  }
  size_t Size() {
    std::lock_guard<std::mutex> lock(mu);
    return items.size();
  }
};

void OnLinkRemoved(sb_space_t, sb_element_t local, int64_t, int32_t reason,
                   void* user) {
  auto* log = static_cast<LinkLog*>(user);
  std::lock_guard<std::mutex> lock(log->mu);
  log->items.emplace_back(local, reason);
  log->cv.notify_all();
}

bool WaitForShared(AddedLog* log, int64_t shared_id, Received* out,
                   int timeout_ms) {
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  std::unique_lock<std::mutex> lock(log->mu);
  for (;;) {
    for (const auto& item : log->items) {
      if (item.first == shared_id) {
        if (out) *out = item.second;
        return true;
      }
    }
    if (log->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
      for (const auto& item : log->items) {
        if (item.first == shared_id) {
          if (out) *out = item.second;
          return true;
        }
      }
      return false;
    }
  }
}

// Moves bytes between one host-side link and one join-side link using only the
// public tunnel calls. `chunk` bounds how many bytes one read or write moves
// (0 = whatever core offers), so a small chunk proves core needs no framing from
// the carrier. When either end has ended the other is closed, as the app does
// when the peer's carrier goes away (PlayerLeft / runner shutdown).
class Pump {
 public:
  // `throttle_us` is how long the carrier rests after moving one chunk: with a
  // small `chunk` it is a slow carrier, like Fusion reliable data (about 13 KB/s
  // measured against the Photon cloud).
  Pump(sb_tunnel_t host_link, sb_tunnel_t join_link, uint32_t chunk,
       int64_t throttle_us = 0)
      : host_(host_link), join_(join_link), chunk_(chunk),
        throttle_us_(throttle_us) {
    thread_ = std::thread([this] { Run(); });
  }
  ~Pump() { Stop(); }
  void Stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
  }
  uint64_t bytes_to_host() const { return to_host_.load(); }
  uint64_t bytes_to_join() const { return to_join_.load(); }
  uint64_t would_blocks() const { return blocked_.load(); }

 private:
  // One direction. Bytes core refused (sb_tunnel_write returned 2, would block)
  // are kept and offered again before anything more is read: the carrier a real
  // app runs does the same.
  struct Dir {
    std::vector<uint8_t> pending;
    size_t offset = 0;
  };
  bool Move(sb_tunnel_t from, sb_tunnel_t to, std::atomic<uint64_t>* count,
            Dir* dir) {
    std::vector<uint8_t> buf(chunk_ ? chunk_ : 65536);
    bool moved = false;
    for (;;) {
      if (dir->offset < dir->pending.size()) {
        const int32_t rc = sb_tunnel_write(
            to, dir->pending.data() + dir->offset,
            static_cast<uint32_t>(dir->pending.size() - dir->offset));
        if (rc == 2) {
          blocked_.fetch_add(1);
          return moved;
        }
        if (rc != 1) {  // the link is gone: the carrier to the other peer ends too
          sb_tunnel_close(from);
          sb_tunnel_close(to);
          dir->pending.clear();
          dir->offset = 0;
          return moved;
        }
        dir->pending.clear();
        dir->offset = 0;
        moved = true;
        if (throttle_us_ > 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(throttle_us_));
        }
        if (chunk_) return moved;  // one chunk per turn: a slow, fragmenting carrier
      }
      uint32_t n = 0;
      const int32_t rc =
          sb_tunnel_read(from, buf.data(), static_cast<uint32_t>(buf.size()), &n);
      if (rc <= 0 && n == 0) {
        // The end of the stream, or the link is gone: the carrier to the other
        // peer ends too.
        sb_tunnel_close(from);
        sb_tunnel_close(to);
        return moved;
      }
      if (n == 0) return moved;
      count->fetch_add(n);
      dir->pending.assign(buf.begin(), buf.begin() + n);
      dir->offset = 0;
    }
  }
  void Run() {
    while (!stop_.load()) {
      const bool a = Move(join_, host_, &to_host_, &up_);
      const bool b = Move(host_, join_, &to_join_, &down_);
      if (!a && !b) std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  }
  sb_tunnel_t host_, join_;
  uint32_t chunk_;
  int64_t throttle_us_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> to_host_{0}, to_join_{0};
  std::atomic<uint64_t> blocked_{0};
  Dir up_, down_;
  std::thread thread_;
};

struct Peer {
  sb_session_t session = 0;
  sb_space_t space = 0;
  sb_tunnel_t host_link = 0;  // the host's end
  sb_tunnel_t join_link = 0;  // this peer's end
  sb_handle_t handle = 0;
  std::unique_ptr<Pump> pump;
  AddedLog added;
  LinkLog links;
};

class TunnelRoom {
 public:
  ~TunnelRoom() {
    for (auto& p : peers_) {
      if (p->pump) p->pump->Stop();
    }
    for (auto& p : peers_) {
      if (p->space) sb_register_on_element_added(p->space, nullptr, nullptr);
      if (p->space) sb_register_on_link_removed(p->space, nullptr, nullptr);
    }
    if (host_space) sb_register_on_link_removed(host_space, nullptr, nullptr);
    if (host_space) sb_register_on_element_added(host_space, nullptr, nullptr);
    for (auto& p : peers_) {
      if (p->session) sb_destroy_session(p->session);
    }
    if (host_session) sb_destroy_session(host_session);
  }

  ::testing::AssertionResult Host() {
    host_session = sb_create_session();
    host_space = host_session ? sb_add_space(host_session) : 0;
    if (!host_space) return ::testing::AssertionFailure() << "no host space";
    const sb_handle_t h = sb_host_space_tunneled(host_space);
    if (h == 0 || !sb_try_wait(h, 2000)) {
      return ::testing::AssertionFailure() << "sb_host_space_tunneled failed";
    }
    sb_register_on_element_added(host_space, OnElementAdded, &host_added);
    sb_register_on_link_removed(host_space, OnLinkRemoved, &host_links);
    return ::testing::AssertionSuccess();
  }

  // Joins a blank peer to the blank host. Returns null on failure.
  Peer* Join(uint32_t chunk = 0, int timeout_ms = 20000, int64_t throttle_us = 0) {
    auto peer = std::make_unique<Peer>();
    peer->session = sb_create_session();
    peer->space = peer->session ? sb_add_space(peer->session) : 0;
    if (!peer->space) return nullptr;
    if (sb_accept_tunnel(host_space, &peer->host_link) != 1) return nullptr;
    peer->handle = sb_connect_space_tunneled(peer->space, &peer->join_link);
    if (peer->handle == 0) return nullptr;
    peer->pump = std::make_unique<Pump>(peer->host_link, peer->join_link, chunk,
                                        throttle_us);
    if (!sb_try_wait(peer->handle, timeout_ms)) return nullptr;
    sb_register_on_element_added(peer->space, OnElementAdded, &peer->added);
    sb_register_on_link_removed(peer->space, OnLinkRemoved, &peer->links);
    // The Subscribe stream settles asynchronously after the bootstrap.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    peers_.push_back(std::move(peer));
    return peers_.back().get();
  }

  sb_session_t host_session = 0;
  sb_space_t host_space = 0;
  AddedLog host_added;
  LinkLog host_links;
  std::vector<std::unique_ptr<Peer>> peers_;
};

// Host room (shared id 100) shared to every peer; each peer resolves it onto
// itself with its own geometry.
::testing::AssertionResult ShareRoom(TunnelRoom* room, sb_element_t* host_room,
                                     std::vector<sb_element_t>* peer_rooms) {
  *host_room = sb_add_element(room->host_space, 0);
  sb_set_element_shared_id(room->host_space, *host_room, kRoomShared);
  SetSquare(room->host_space, *host_room, SB_GEOMETRY_NAVIGABLE, 0, 0, 100, 100);
  sb_share_element(room->host_space, *host_room, nullptr, 0);
  for (auto& peer : room->peers_) {
    Received r;
    if (!WaitForShared(&peer->added, kRoomShared, &r, 5000)) {
      return ::testing::AssertionFailure() << "room did not reach a peer";
    }
    sb_resolve_shared_element(peer->space, r.local, r.local);
    SetSquare(peer->space, r.local, SB_GEOMETRY_NAVIGABLE, 0, 0, 100, 100);
    peer_rooms->push_back(r.local);
  }
  return ::testing::AssertionSuccess();
}

// A child of the shared room on `space`, with `shared_id`.
sb_element_t AddChild(sb_space_t space, sb_element_t room, int64_t shared_id) {
  const sb_element_t e = sb_add_element(space, room);
  sb_set_element_shared_id(space, e, shared_id);
  return e;
}

}  // namespace

TEST(TunnelCTest, AbiMinorIsNineAndTheHandshakeNeedsNoAddress) {
  EXPECT_EQ(sb_abi_version(), 4u);
  EXPECT_EQ(sb_abi_minor_version(), 14u);
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* peer = room.Join();
  ASSERT_NE(peer, nullptr);
  EXPECT_NE(peer->host_link, 0);
  EXPECT_NE(peer->join_link, 0);
  EXPECT_EQ(sb_get_error(peer->handle), nullptr);
  // Everything the join needed went through the two links.
  EXPECT_GT(peer->pump->bytes_to_join(), 0u) << "host -> joiner (bootstrap)";
  EXPECT_GT(peer->pump->bytes_to_host(), 0u) << "joiner -> host";
}

TEST(TunnelCTest, ShareRoundTripCarriesLargeGlbBothWays) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* peer = room.Join();
  ASSERT_NE(peer, nullptr);

  // Scan GLBs are megabytes. 8 MiB down and 6 MiB up, each byte-exact.
  const auto down = MakePayload(8u << 20, 1);
  const auto up = MakePayload(6u << 20, 2);

  sb_element_t host_room = 0;
  std::vector<sb_element_t> peer_rooms;
  ASSERT_TRUE(ShareRoom(&room, &host_room, &peer_rooms));

  const sb_element_t host_elem = AddChild(room.host_space, host_room, kGlbFromHost);
  sb_set_element_tag(room.host_space, host_elem, "Glb");
  sb_share_element(room.host_space, host_elem, down.data(),
                   static_cast<int64_t>(down.size()));
  Received got;
  ASSERT_TRUE(WaitForShared(&peer->added, kGlbFromHost, &got, 30000))
      << "8 MiB payload did not reach the joiner";
  ASSERT_EQ(got.payload.size(), down.size());
  EXPECT_EQ(got.payload, down);

  const sb_element_t client_elem = AddChild(peer->space, peer_rooms[0], kGlbFromClient);
  sb_set_element_tag(peer->space, client_elem, "Glb");
  sb_share_element(peer->space, client_elem, up.data(),
                   static_cast<int64_t>(up.size()));
  ASSERT_TRUE(WaitForShared(&room.host_added, kGlbFromClient, &got, 30000))
      << "6 MiB payload did not reach the host";
  ASSERT_EQ(got.payload.size(), up.size());
  EXPECT_EQ(got.payload, up);
}

TEST(TunnelCTest, AShareThatTakesLongerThanTheTcpCallDeadlineStillArrives) {
  // Client calls carry a fixed 1 s deadline over TCP (kDefaultTimeoutMs). A
  // carrier the application runs at its own pace (Fusion reliable data moves
  // about 13 KB/s) cannot move a scan in that time, so over the tunnel the
  // deadline does not apply and the link ending is what reports a lost peer.
  // Here the carrier moves ~100 KB/s and the payload needs about 3 s.
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* peer = room.Join(/*chunk=*/2048, /*timeout_ms=*/30000, /*throttle_us=*/20000);
  ASSERT_NE(peer, nullptr);
  sb_element_t host_room = 0;
  std::vector<sb_element_t> peer_rooms;
  ASSERT_TRUE(ShareRoom(&room, &host_room, &peer_rooms));

  const auto payload = MakePayload(300 * 1024, 9);
  const sb_element_t elem = AddChild(peer->space, peer_rooms[0], kGlbFromClient);
  const auto start = Clock::now();
  sb_share_element(peer->space, elem, payload.data(),
                   static_cast<int64_t>(payload.size()));
  Received got;
  ASSERT_TRUE(WaitForShared(&room.host_added, kGlbFromClient, &got, 60000));
  const auto took_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           Clock::now() - start).count();
  EXPECT_GT(took_ms, 1000) << "the test must outlast the TCP call deadline";
  EXPECT_EQ(got.payload, payload);
}

TEST(TunnelCTest, AnyCarrierChunkingWorks) {
  // The carrier may split the stream at any boundary: one byte at a time.
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* peer = room.Join(/*chunk=*/1, /*timeout_ms=*/60000);
  ASSERT_NE(peer, nullptr);
  sb_element_t host_room = 0;
  std::vector<sb_element_t> peer_rooms;
  ASSERT_TRUE(ShareRoom(&room, &host_room, &peer_rooms));
  const auto payload = MakePayload(3000, 3);
  const sb_element_t elem = AddChild(room.host_space, host_room, kSmallShared);
  sb_share_element(room.host_space, elem, payload.data(),
                   static_cast<int64_t>(payload.size()));
  Received got;
  ASSERT_TRUE(WaitForShared(&peer->added, kSmallShared, &got, 60000));
  EXPECT_EQ(got.payload, payload);
}

// The host shares a room, an object and an avatar element. The peer binds
// Spacing on the received avatar. A pose sampled on the host reaches the peer.
TEST(TunnelCTest, BindingAndPoseRoundTripHostToPeer) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* peer = room.Join();
  ASSERT_NE(peer, nullptr);

  sb_element_t host_room = 0;
  std::vector<sb_element_t> peer_rooms;
  ASSERT_TRUE(ShareRoom(&room, &host_room, &peer_rooms));

  const sb_element_t object = sb_add_element(room.host_space, host_room);
  sb_set_element_shared_id(room.host_space, object, kObjectShared);
  sb_set_element_tag(room.host_space, object, "Object");
  SetSquare(room.host_space, object, SB_GEOMETRY_CONTAINMENT, 60, 60, 80, 80);
  sb_share_element(room.host_space, object, nullptr, 0);
  Received rx;
  ASSERT_TRUE(WaitForShared(&peer->added, kObjectShared, &rx, 5000));
  sb_resolve_shared_element(peer->space, rx.local, rx.local);
  SetSquare(peer->space, rx.local, SB_GEOMETRY_CONTAINMENT, 60, 60, 80, 80);
  sb_set_element_tag(peer->space, rx.local, "Object");

  const sb_element_t actor_elem = sb_add_element(room.host_space, host_room);
  sb_set_element_shared_id(room.host_space, actor_elem, kActorShared);
  const sb_actor_t source = sb_create_actor(room.host_space);
  spatialblend_c_test::PublishStartPose(room.host_space, source);
  ASSERT_NE(source, 0);
  sb_add_anchor(room.host_space, source, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(room.host_space, source, actor_elem);
  ASSERT_EQ(sb_bind(room.host_space, source, "root", SB_KERNEL_SPACING), 1);
  sb_set_min_move(room.host_space, 0.0);
  sb_set_min_yaw(room.host_space, 0.0);
  sb_share_element(room.host_space, actor_elem, nullptr, 0);
  ASSERT_TRUE(WaitForShared(&peer->added, kActorShared, &rx, 5000));
  sb_resolve_shared_element(peer->space, rx.local, rx.local);

  const sb_actor_t remote = sb_create_actor(peer->space);
  spatialblend_c_test::PublishStartPose(peer->space, remote);
  ASSERT_NE(remote, 0);
  sb_add_anchor(peer->space, remote, "root", 0, 0, 0, nullptr);
  sb_set_actor_element(peer->space, remote, rx.local);
  ASSERT_EQ(sb_find_actor_by_shared_id(peer->space, kActorShared), remote);
  ASSERT_EQ(sb_bind(peer->space, remote, "root", SB_KERNEL_SPACING), 1);

  // The avatar walks toward the object: every pose the host emits is applied on
  // the peer.
  int applied = 0;
  for (int i = 0; i <= 8; ++i) {
    const double a = i / 8.0;
    const int64_t before =
        sb_get_last_emitted_update_sequence(room.host_space, source);
    sb_update_actor_state(room.host_space, source, 1.0 * (1 - a) + 70.0 * a,
                          1.0 * (1 - a) + 70.0 * a, 0.0);
    const int64_t after =
        sb_get_last_emitted_update_sequence(room.host_space, source);
    if (after <= before) continue;
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (Clock::now() < deadline &&
           sb_get_last_applied_remote_sequence(peer->space, room.host_space,
                                               source) < after) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_GE(sb_get_last_applied_remote_sequence(peer->space, room.host_space,
                                                  source),
              after)
        << "pose " << i << " did not reach the peer over the tunnel";
    ++applied;
  }
  EXPECT_GT(applied, 0);

  // Core's relation state on the peer agrees with the source's.
  sb_relation_state_v2_t s_host[4], s_peer[4];
  std::memset(s_host, 0, sizeof(s_host));
  std::memset(s_peer, 0, sizeof(s_peer));
  for (auto& s : s_host) s.struct_size = sizeof(s);
  for (auto& s : s_peer) s.struct_size = sizeof(s);
  const int32_t nh = sb_get_relation_states_v2(room.host_space, source, s_host, 4,
                                               sizeof(s_host[0]));
  const int32_t np = sb_get_relation_states_v2(peer->space, remote, s_peer, 4,
                                               sizeof(s_peer[0]));
  ASSERT_GT(nh, 0);
  ASSERT_GT(np, 0);
  EXPECT_NEAR(s_host[0].proximity, s_peer[0].proximity, 1e-6);
}

TEST(TunnelCTest, ThreePeersAreRelayedThroughTheHost) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* a = room.Join();
  Peer* b = room.Join();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a->host_link, b->host_link);

  sb_element_t host_room = 0;
  std::vector<sb_element_t> peer_rooms;
  ASSERT_TRUE(ShareRoom(&room, &host_room, &peer_rooms));
  const auto from_host = MakePayload(100000, 4);
  const sb_element_t h = AddChild(room.host_space, host_room, 801);
  sb_share_element(room.host_space, h, from_host.data(),
                   static_cast<int64_t>(from_host.size()));
  Received got;
  ASSERT_TRUE(WaitForShared(&a->added, 801, &got, 10000));
  EXPECT_EQ(got.payload, from_host);
  ASSERT_TRUE(WaitForShared(&b->added, 801, &got, 10000));
  EXPECT_EQ(got.payload, from_host);

  // A peer's share reaches the host and is relayed to the other peer only.
  const auto from_a = MakePayload(250000, 5);
  const sb_element_t ea = AddChild(a->space, peer_rooms[0], 802);
  sb_share_element(a->space, ea, from_a.data(), static_cast<int64_t>(from_a.size()));
  ASSERT_TRUE(WaitForShared(&room.host_added, 802, &got, 10000));
  EXPECT_EQ(got.payload, from_a);
  ASSERT_TRUE(WaitForShared(&b->added, 802, &got, 10000));
  EXPECT_EQ(got.payload, from_a);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_FALSE(WaitForShared(&a->added, 802, nullptr, 0))
      << "the sharer must not get its own element back";
}

TEST(TunnelCTest, APeerLeavingLeavesTheHostServingTheOthers) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* a = room.Join();
  Peer* b = room.Join();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  sb_element_t host_room = 0;
  std::vector<sb_element_t> peer_rooms;
  ASSERT_TRUE(ShareRoom(&room, &host_room, &peer_rooms));

  // Peer A leaves: its carrier goes away (PlayerLeft). The host closes A's link.
  a->pump->Stop();
  sb_tunnel_close(a->join_link);
  sb_destroy_session(a->session);
  a->session = 0;
  a->space = 0;
  // The host closes A's link when A's player leaves.
  sb_tunnel_close(a->host_link);
  uint8_t one = 0;
  EXPECT_EQ(sb_tunnel_write(a->host_link, &one, 1), -1)
      << "a closed link is forgotten";

  // The room still works for B.
  const auto payload = MakePayload(4000, 6);
  const sb_element_t h = AddChild(room.host_space, host_room, 810);
  sb_share_element(room.host_space, h, payload.data(),
                   static_cast<int64_t>(payload.size()));
  Received got;
  ASSERT_TRUE(WaitForShared(&b->added, 810, &got, 10000));
  EXPECT_EQ(got.payload, payload);

  // And a new peer can still join after one left.
  EXPECT_NE(room.Join(), nullptr);
}

TEST(TunnelCTest, HostLeavingEndsEveryPeerLink) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* a = room.Join();
  ASSERT_NE(a, nullptr);
  sb_destroy_session(room.host_session);  // the host's runner is gone
  room.host_session = 0;
  room.host_space = 0;
  // The host's links end with it. The pump (the carrier) hands that to A's
  // side, whose link then is gone: that is what the app surfaces as "host left".
  const auto deadline = Clock::now() + std::chrono::seconds(15);
  uint8_t byte = 0;
  auto ended = [&] {
    const int32_t rc = sb_tunnel_write(a->join_link, &byte, 1);
    return rc != 1 && rc != 2;
  };
  while (Clock::now() < deadline && !ended()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(ended())
      << "the joiner's link must end when the host leaves";
}

TEST(TunnelCTest, JoinFailsWhenItsLinkEndsBeforeTheHostAnswers) {
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  sb_tunnel_t link = 0;
  const sb_handle_t handle = sb_connect_space_tunneled(space, &link);
  ASSERT_NE(handle, 0);
  ASSERT_NE(link, 0);
  EXPECT_EQ(sb_try_wait(handle, 0), 0) << "still pending: nobody pumps the link";
  sb_tunnel_close(link);
  EXPECT_EQ(sb_try_wait(handle, 20000), 0);
  EXPECT_NE(sb_get_error(handle), nullptr);
  sb_destroy_session(session);
}

TEST(TunnelCTest, DestroyingASessionMidJoinDoesNotHang) {
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  sb_tunnel_t link = 0;
  ASSERT_NE(sb_connect_space_tunneled(space, &link), 0);
  const auto start = Clock::now();
  sb_destroy_session(session);
  EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start)
                .count(),
            10);
  uint8_t b = 0;
  EXPECT_EQ(sb_tunnel_write(link, &b, 1), -1) << "the link died with its session";
}

TEST(TunnelCTest, ArgumentAndRoleErrors) {
  // Null / unknown handles.
  uint8_t b[4] = {0};
  uint32_t n = 0;
  EXPECT_EQ(sb_tunnel_write(0, b, 1), -1);
  EXPECT_EQ(sb_tunnel_write(123456, b, 1), -1);
  EXPECT_EQ(sb_tunnel_read(0, b, 4, &n), -1);
  sb_tunnel_close(0);       // harmless
  sb_tunnel_close(123456);  // harmless
  EXPECT_EQ(sb_accept_tunnel(0, nullptr), -1);
  sb_tunnel_t link = 7;
  EXPECT_EQ(sb_accept_tunnel(0, &link), -1);
  EXPECT_EQ(link, 0);
  EXPECT_EQ(sb_connect_space_tunneled(0, &link), 0);
  EXPECT_EQ(sb_connect_space_tunneled(0, nullptr), 0);
  EXPECT_EQ(sb_host_space_tunneled(0), 0);

  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  // Not hosted tunneled: nothing to accept on.
  EXPECT_EQ(sb_accept_tunnel(space, &link), 0);
  ASSERT_NE(sb_host_space_tunneled(space), 0);
  // One session hosts or joins once, whatever the transport.
  EXPECT_EQ(sb_host_space_tunneled(space), 0);
  EXPECT_EQ(sb_connect_space_tunneled(space, &link), 0);
  EXPECT_EQ(sb_connect_space(space, "127.0.0.1", 1), 0);
  EXPECT_EQ(sb_accept_tunnel(space, &link), 1);
  EXPECT_NE(link, 0);
  EXPECT_EQ(sb_tunnel_read(link, nullptr, 4, &n), -1);
  EXPECT_EQ(sb_tunnel_read(link, b, 4, nullptr), -1);
  EXPECT_EQ(sb_tunnel_write(link, nullptr, 1), -1);
  EXPECT_EQ(sb_tunnel_write(link, nullptr, 0), 1);
  EXPECT_EQ(sb_tunnel_read(link, b, 4, &n), 1);
  sb_tunnel_close(link);
  sb_tunnel_close(link);  // twice is harmless
  sb_destroy_session(session);

  // A TCP-hosted space cannot accept tunnels.
  const sb_session_t tcp = sb_create_session();
  const sb_space_t tcp_space = sb_add_space(tcp);
  ASSERT_NE(sb_host_space(tcp_space, 0), 0);
  EXPECT_EQ(sb_accept_tunnel(tcp_space, &link), 0);
  sb_destroy_session(tcp);
}

// ---- App quit, join staging, join retries and lock order ----------------

// sb_cleanup_all is the app-quit path: it must end every session's
// tunnels like sb_destroy_session does. Left alone, a pending join's thread is
// still joinable at static destruction (std::terminate) and writes to a freed
// session if the join later answers.
TEST(TunnelCTest, CleanupAllRetiresPendingJoinsAndLiveLinks) {
  const sb_session_t host_session = sb_create_session();
  const sb_space_t host_space = sb_add_space(host_session);
  ASSERT_NE(sb_host_space_tunneled(host_space), 0);
  sb_tunnel_t host_link = 0;
  ASSERT_EQ(sb_accept_tunnel(host_space, &host_link), 1);

  const sb_session_t join_session = sb_create_session();
  const sb_space_t join_space = sb_add_space(join_session);
  sb_tunnel_t join_link = 0;
  const sb_handle_t handle = sb_connect_space_tunneled(join_space, &join_link);
  ASSERT_NE(handle, 0);
  EXPECT_EQ(sb_try_wait(handle, 0), 0) << "pending: nobody pumps the link";

  const auto start = Clock::now();
  sb_cleanup_all();
  EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start).count(), 10);

  uint8_t b = 0;
  EXPECT_EQ(sb_tunnel_write(host_link, &b, 1), -1) << "the host link died with its session";
  EXPECT_EQ(sb_tunnel_write(join_link, &b, 1), -1) << "the join link died with its session";
  EXPECT_EQ(sb_try_wait(handle, 0), 0);
  // The process can still start over after the cleanup.
  const sb_session_t again = sb_create_session();
  EXPECT_NE(again, 0);
  sb_destroy_session(again);
}

TEST(TunnelCTest, CleanupAllAfterACompletedJoinLeavesNothingBehind) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* peer = room.Join();
  ASSERT_NE(peer, nullptr);
  peer->pump->Stop();
  sb_register_on_element_added(peer->space, nullptr, nullptr);
  sb_register_on_element_added(room.host_space, nullptr, nullptr);
  sb_cleanup_all();
  uint8_t b = 0;
  EXPECT_EQ(sb_tunnel_write(peer->join_link, &b, 1), -1);
  EXPECT_EQ(sb_tunnel_write(peer->host_link, &b, 1), -1);
  // The room's destructor destroys sessions that cleanup already ended.
}

// The join's worker thread only stages the bootstrap: the space changes
// when the app calls sb_try_wait, on the app's thread, not before.
TEST(TunnelCTest, TheBootstrapIsAppliedByOsTryWaitNotByTheJoinThread) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  const sb_element_t on_host = sb_add_element(room.host_space, 0);
  sb_set_element_shared_id(room.host_space, on_host, 555);

  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  sb_tunnel_t host_link = 0, join_link = 0;
  ASSERT_EQ(sb_accept_tunnel(room.host_space, &host_link), 1);
  const sb_handle_t handle = sb_connect_space_tunneled(space, &join_link);
  ASSERT_NE(handle, 0);
  {
    Pump pump(host_link, join_link, 0);
    // The host's bootstrap has crossed the carrier once bytes flowed both ways
    // for a while; the app has not called sb_try_wait.
    const auto until = Clock::now() + std::chrono::milliseconds(1500);
    while (Clock::now() < until && pump.bytes_to_join() == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_GT(pump.bytes_to_join(), 0u);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    EXPECT_EQ(sb_find_element_by_shared_id(space, 555), 0)
        << "a thread other than the app's changed the space";
    EXPECT_NE(sb_try_wait(handle, 20000), 0);
    EXPECT_NE(sb_find_element_by_shared_id(space, 555), 0)
        << "sb_try_wait applies the bootstrap";
    pump.Stop();
  }
  sb_tunnel_close(join_link);
  sb_tunnel_close(host_link);
  sb_destroy_session(session);
}

// A join cannot hang. The carrier is alive but the host never answers;
// no timer ends the join, closing the link does.
TEST(TunnelCTest, ClosingTheLinkFailsAJoinWhoseHostIsSilent) {
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  sb_tunnel_t link = 0;
  const sb_handle_t handle = sb_connect_space_tunneled(space, &link);
  ASSERT_NE(handle, 0);
  // A carrier that takes what core sends and delivers nothing back.
  std::atomic<bool> stop{false};
  std::thread carrier([&] {
    uint8_t buf[4096];
    while (!stop) {
      uint32_t n = 0;
      if (sb_tunnel_read(link, buf, sizeof(buf), &n) <= 0 && n == 0) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_EQ(sb_try_wait(handle, 0), 0) << "still waiting for a host that is silent";
  const auto start = Clock::now();
  sb_tunnel_close(link);
  EXPECT_EQ(sb_try_wait(handle, 20000), 0);
  EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start).count(), 10);
  EXPECT_NE(sb_get_error(handle), nullptr);
  stop = true;
  carrier.join();
  sb_destroy_session(session);
}

// Also when the host has answered and the app has not yet applied the
// bootstrap.
TEST(TunnelCTest, ClosingTheLinkFailsAJoinWhoseBootstrapWaitsForTheApp) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  sb_tunnel_t host_link = 0, join_link = 0;
  ASSERT_EQ(sb_accept_tunnel(room.host_space, &host_link), 1);
  const sb_handle_t handle = sb_connect_space_tunneled(space, &join_link);
  ASSERT_NE(handle, 0);
  {
    Pump pump(host_link, join_link, 0);
    const auto until = Clock::now() + std::chrono::milliseconds(1500);
    while (Clock::now() < until && pump.bytes_to_join() == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const auto start = Clock::now();
    sb_tunnel_close(join_link);  // the app never serviced the join
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start).count(), 10);
    pump.Stop();
  }
  EXPECT_EQ(sb_try_wait(handle, 0), 0);
  EXPECT_NE(sb_get_error(handle), nullptr);
  sb_tunnel_close(host_link);
  sb_destroy_session(session);
}

// A joiner that leaves can join again in the same session.
TEST(TunnelCTest, AJoinerThatLeftCanRejoinInTheSameSession) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* peer = room.Join();
  ASSERT_NE(peer, nullptr);

  peer->pump->Stop();
  sb_tunnel_close(peer->join_link);
  sb_tunnel_close(peer->host_link);

  sb_tunnel_t host_link = 0, join_link = 0;
  ASSERT_EQ(sb_accept_tunnel(room.host_space, &host_link), 1);
  const sb_handle_t handle = sb_connect_space_tunneled(peer->space, &join_link);
  ASSERT_NE(handle, 0) << "the session's client role was not released";
  Pump pump(host_link, join_link, 0);
  EXPECT_NE(sb_try_wait(handle, 20000), 0);
  EXPECT_EQ(sb_get_error(handle), nullptr);
  pump.Stop();
  sb_tunnel_close(join_link);
  sb_tunnel_close(host_link);
}

// A join that failed (link closed while pending) can be retried too.
TEST(TunnelCTest, AFailedJoinCanBeRetriedInTheSameSession) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  const sb_session_t session = sb_create_session();
  const sb_space_t space = sb_add_space(session);
  sb_tunnel_t first = 0;
  const sb_handle_t h1 = sb_connect_space_tunneled(space, &first);
  ASSERT_NE(h1, 0);
  sb_tunnel_close(first);
  EXPECT_EQ(sb_try_wait(h1, 20000), 0);

  sb_tunnel_t host_link = 0, join_link = 0;
  ASSERT_EQ(sb_accept_tunnel(room.host_space, &host_link), 1);
  const sb_handle_t h2 = sb_connect_space_tunneled(space, &join_link);
  ASSERT_NE(h2, 0);
  Pump pump(host_link, join_link, 0);
  EXPECT_NE(sb_try_wait(h2, 20000), 0);
  pump.Stop();
  sb_tunnel_close(join_link);
  sb_tunnel_close(host_link);
  sb_destroy_session(session);
}

// Close racing read and write on the same link from other
// threads, and a send to a peer that is gone.
TEST(TunnelCTest, CloseRacingReadAndWriteAndSendsToADeadPeerAreSafe) {
  for (int round = 0; round < 20; ++round) {
    const sb_session_t session = sb_create_session();
    const sb_space_t space = sb_add_space(session);
    ASSERT_NE(sb_host_space_tunneled(space), 0);
    sb_tunnel_t link = 0;
    ASSERT_EQ(sb_accept_tunnel(space, &link), 1);
    std::atomic<bool> go{false};
    std::vector<uint8_t> junk(16 * 1024, 0x5a);
    std::thread writer([&] {
      while (!go) {}
      for (int i = 0; i < 500; ++i) {
        const int32_t rc = sb_tunnel_write(link, junk.data(), junk.size());
        if (rc == 0 || rc == -1) break;  // closed / forgotten: never a crash
      }
    });
    std::thread reader([&] {
      uint8_t buf[1024];
      while (!go) {}
      for (int i = 0; i < 5000; ++i) {
        uint32_t n = 0;
        if (sb_tunnel_read(link, buf, sizeof(buf), &n) != 1) break;
      }
    });
    std::thread closer([&] {
      while (!go) {}
      std::this_thread::sleep_for(std::chrono::microseconds(100 * (round % 5)));
      sb_tunnel_close(link);
    });
    go = true;
    writer.join();
    reader.join();
    closer.join();
    uint8_t b = 0;
    EXPECT_EQ(sb_tunnel_write(link, &b, 1), -1);
    sb_destroy_session(session);
  }
}

// The two registry locks are taken in one order. Hosting or joining
// starts from a space (spaces then sessions) while destroying a session ends
// with its spaces (sessions then spaces); with the orders crossed this
// deadlocks. The watchdog turns a deadlock into a failure.
TEST(TunnelCTest, SessionAndSpaceRegistryLocksDoNotDeadlock) {
  std::atomic<bool> finished{false};
  std::thread watchdog([&] {
    for (int i = 0; i < 1200 && !finished; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!finished) {
      std::fprintf(stderr, "DEADLOCK: registry lock order inversion\n");
      std::_Exit(3);
    }
  });
  std::vector<std::thread> workers;
  for (int t = 0; t < 4; ++t) {
    workers.emplace_back([] {
      for (int i = 0; i < 150; ++i) {
        const sb_session_t session = sb_create_session();
        const sb_space_t space = sb_add_space(session);
        const sb_handle_t h = sb_host_space_tunneled(space);
        sb_tunnel_t link = 0;
        if (h != 0) (void)sb_accept_tunnel(space, &link);
        sb_destroy_session(session);
      }
    });
  }
  for (auto& w : workers) w.join();
  finished = true;
  watchdog.join();
}

// ABI 4.9: unlink across spaces over the tunnel. Order (b): a receiver unlinks
// while the sharer still shares; the sharer is told through the host.
TEST(TunnelCTest, ReceiverUnlinkReachesTheSharerThroughTheHost) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* a = room.Join();
  Peer* b = room.Join();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  sb_element_t host_room = 0;
  std::vector<sb_element_t> peer_rooms;
  ASSERT_TRUE(ShareRoom(&room, &host_room, &peer_rooms));

  // Peer A shares; peer B links it onto an element of its own.
  const sb_element_t ea = AddChild(a->space, peer_rooms[0], 810);
  sb_share_element(a->space, ea, nullptr, 0);
  Received got;
  ASSERT_TRUE(WaitForShared(&b->added, 810, &got, 10000));
  const sb_element_t b_local = AddChild(b->space, peer_rooms[1], 9810);
  sb_resolve_shared_element(b->space, got.local, b_local);
  // The host received it too and keeps it as a new element.
  ASSERT_TRUE(WaitForShared(&room.host_added, 810, &got, 10000));
  sb_resolve_shared_element(room.host_space, got.local, got.local);

  EXPECT_EQ(sb_unlink(b->space, b_local), 1);
  ASSERT_TRUE(a->links.WaitFor(1));
  EXPECT_EQ(a->links.items[0].first, ea);
  EXPECT_EQ(a->links.items[0].second, SB_LINK_REMOVED_PEER_UNLINKED);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_EQ(room.host_links.Size(), 0u) << "only the sharer is told";
  EXPECT_EQ(b->links.Size(), 0u);
  EXPECT_EQ(a->links.Size(), 1u);
}

// Order (a): the sharer unshares while receivers stay linked; each is told, the
// host (which kept a replica) removes it, and the link is gone for a re-share.
TEST(TunnelCTest, SharerUnshareReleasesTheLinkedReceiversOverTheTunnel) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  Peer* a = room.Join();
  Peer* b = room.Join();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  sb_element_t host_room = 0;
  std::vector<sb_element_t> peer_rooms;
  ASSERT_TRUE(ShareRoom(&room, &host_room, &peer_rooms));

  const sb_element_t eh = AddChild(room.host_space, host_room, 820);
  sb_share_element(room.host_space, eh, nullptr, 0);
  Received got;
  ASSERT_TRUE(WaitForShared(&a->added, 820, &got, 10000));
  const sb_element_t a_local = AddChild(a->space, peer_rooms[0], 9820);
  sb_resolve_shared_element(a->space, got.local, a_local);
  ASSERT_TRUE(WaitForShared(&b->added, 820, &got, 10000));
  const sb_element_t b_local = AddChild(b->space, peer_rooms[1], 9821);
  sb_resolve_shared_element(b->space, got.local, b_local);

  // A lets go first; B and the host's share are unaffected.
  EXPECT_EQ(sb_unlink(a->space, a_local), 1);
  ASSERT_TRUE(room.host_links.WaitFor(1));
  EXPECT_EQ(room.host_links.items[0].first, eh);

  sb_unshare_element(room.host_space, eh);
  ASSERT_TRUE(b->links.WaitFor(1));
  EXPECT_EQ(b->links.items[0].first, b_local);
  EXPECT_EQ(b->links.items[0].second, SB_LINK_REMOVED_PEER_UNSHARED);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_EQ(a->links.Size(), 0u) << "A had already unlinked";
  EXPECT_EQ(sb_find_element_by_shared_id(a->space, 9820), a_local);
  EXPECT_EQ(sb_find_element_by_shared_id(b->space, 9821), b_local);

  // Re-share after both ends of the story: B links anew.
  sb_share_element(room.host_space, eh, nullptr, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  size_t seen = 0;
  {
    std::lock_guard<std::mutex> lock(b->added.mu);
    for (const auto& item : b->added.items) seen += item.first == 820;
  }
  EXPECT_EQ(seen, 2u) << "the re-share is delivered again";
}

// A joiner's bootstrap copies the host's element ids verbatim. The joiner's own next element ids must skip them:
// sb_add_element never answers 0 for an id the host's tree already used (seen in the Unity room matrix: every add of
// the joiner failed once the host's elements had arrived).
TEST(TunnelCTest, AJoinersNewElementsNeverCollideWithTheHostsBootstrappedIds) {
  TunnelRoom room;
  ASSERT_TRUE(room.Host());
  const sb_element_t host_room = sb_add_element(room.host_space, 0);
  ASSERT_NE(host_room, 0);
  sb_set_element_shared_id(room.host_space, host_room, 1);
  for (int i = 0; i < 40; ++i) ASSERT_NE(sb_add_element(room.host_space, host_room), 0);
  Peer* peer = room.Join();
  ASSERT_NE(peer, nullptr);
  for (int i = 0; i < 80; ++i) {
    const sb_element_t elem = sb_add_element(peer->space, 0);
    EXPECT_NE(elem, 0) << "add " << i << " collided with a bootstrapped id";
  }
}
