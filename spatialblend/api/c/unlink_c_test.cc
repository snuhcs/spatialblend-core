// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// ABI 4.9: unlink across spaces. sb_unlink removes the cross-space
// link of a receiver's element on both sides and tells the sharer through
// sb_on_link_removed_t; a sharer's unshare tells every receiver that had linked
// an element. Both orders run over three in-process transports: spaces on one
// hub, and a TCP host/client in each direction. The tunnel runs in
// tunnel_c_test.cc, the relation resolution in relation_event_c_test.cc.

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "spatialblend/api/c/spatialblend.h"

namespace {

constexpr int64_t kRoomShared = 100;
constexpr int64_t kShareA = 500;
constexpr int64_t kShareB = 501;
// What a receiver's own element carries, so it can be found again.
constexpr int64_t kLocalA = 9500;
constexpr int64_t kLocalB = 9501;

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

template <typename T>
struct Log {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<T> items;
  void Push(T item) {
    std::lock_guard<std::mutex> lock(mu);
    items.push_back(std::move(item));
    cv.notify_all();
  }
  size_t Size() {
    std::lock_guard<std::mutex> lock(mu);
    return items.size();
  }
  void Clear() {
    std::lock_guard<std::mutex> lock(mu);
    items.clear();
  }
  std::vector<T> Snapshot() {
    std::lock_guard<std::mutex> lock(mu);
    return items;
  }
  // Waits until at least `n` items exist.
  bool WaitFor(size_t n, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                       [&] { return items.size() >= n; });
  }
};

struct Added {
  sb_element_t local;
  int64_t shared_id;
};
struct LinkRemoved {
  sb_element_t local;
  int64_t peer;
  int32_t reason;
};

void OnAdded(sb_space_t, sb_element_t local, int64_t shared_id, const char*,
             const void*, int64_t, void* user) {
  static_cast<Log<Added>*>(user)->Push({local, shared_id});
}
void OnRemoved(sb_space_t, sb_element_t local, void* user) {
  static_cast<Log<sb_element_t>*>(user)->Push(local);
}
void OnLinkRemoved(sb_space_t, sb_element_t local, int64_t peer,
                   int32_t reason, void* user) {
  static_cast<Log<LinkRemoved>*>(user)->Push({local, peer, reason});
}

// What one app space observes.
struct Observer {
  Log<Added> added;
  Log<sb_element_t> removed;
  Log<LinkRemoved> link_removed;
  void Attach(sb_space_t space) {
    sb_register_on_element_added(space, OnAdded, &added);
    sb_register_on_element_removed(space, OnRemoved, &removed);
    sb_register_on_link_removed(space, OnLinkRemoved, &link_removed);
  }
  static void Detach(sb_space_t space) {
    sb_register_on_element_added(space, nullptr, nullptr);
    sb_register_on_element_removed(space, nullptr, nullptr);
    sb_register_on_link_removed(space, nullptr, nullptr);
  }
  bool HasShared(int64_t shared_id) {
    for (const Added& a : added.Snapshot()) {
      if (a.shared_id == shared_id) return true;
    }
    return false;
  }
  bool WaitForShared(int64_t shared_id, int timeout_ms = 5000) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
      if (HasShared(shared_id)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return HasShared(shared_id);
  }
};

enum class Transport {
  kLocal,            // two spaces on one session hub
  kTcpHostShares,    // the host shares, the client receives
  kTcpClientShares,  // the client shares, the host receives
};

const char* Name(Transport t) {
  switch (t) {
    case Transport::kLocal: return "Local";
    case Transport::kTcpHostShares: return "TcpHostShares";
    case Transport::kTcpClientShares: return "TcpClientShares";
  }
  return "?";
}

// Gives an event time to NOT arrive. Bounded and only used to show absence.
void Settle() { std::this_thread::sleep_for(std::chrono::milliseconds(300)); }

// A sharer and a receiver, joined by one of the transports. Both are blank when
// they connect (a bootstrap would otherwise clone a scene). With a room, the
// sharer's room is shared first and kept by the receiver as a new element, so
// shared elements and replicas have a parent on both sides.
class Rig {
 public:
  ~Rig() {
    if (sharer) Observer::Detach(sharer);
    if (receiver) Observer::Detach(receiver);
    for (sb_space_t space : late_spaces_) Observer::Detach(space);
    for (sb_session_t session : late_sessions_) sb_destroy_session(session);
    if (session_b_) sb_destroy_session(session_b_);
    if (session_a_) sb_destroy_session(session_a_);
  }

  ::testing::AssertionResult Setup(Transport transport, bool with_room = true) {
    session_a_ = sb_create_session();
    if (!session_a_) return ::testing::AssertionFailure() << "no session";
    if (transport == Transport::kLocal) {
      sharer = sb_add_space(session_a_);
      receiver = sb_add_space(session_a_);
      if (!sharer || !receiver) {
        return ::testing::AssertionFailure() << "no spaces";
      }
    } else {
      session_b_ = sb_create_session();
      if (!session_b_) return ::testing::AssertionFailure() << "no session b";
      const sb_space_t host = sb_add_space(session_a_);
      const sb_space_t client = sb_add_space(session_b_);
      if (!host || !client) return ::testing::AssertionFailure() << "no spaces";
      port_ = PickFreePort();
      if (port_ <= 0) return ::testing::AssertionFailure() << "no free port";
      const sb_handle_t h = sb_host_space(host, port_);
      if (!h || !sb_try_wait(h, 2000)) {
        return ::testing::AssertionFailure() << "host failed";
      }
      const sb_handle_t c = sb_connect_space(client, "127.0.0.1", port_);
      if (!c || !sb_try_wait(c, 5000)) {
        return ::testing::AssertionFailure() << "connect failed";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      sharer = transport == Transport::kTcpHostShares ? host : client;
      receiver = transport == Transport::kTcpHostShares ? client : host;
    }
    sharer_obs.Attach(sharer);
    receiver_obs.Attach(receiver);
    if (with_room) {
      sharer_room = sb_add_element(sharer, 0);
      sb_set_element_shared_id(sharer, sharer_room, kRoomShared);
      sb_share_element(sharer, sharer_room, nullptr, 0);
      if (!receiver_obs.WaitForShared(kRoomShared)) {
        return ::testing::AssertionFailure() << "room did not arrive";
      }
      receiver_room = receiver_obs.added.Snapshot()[0].local;
      sb_resolve_shared_element(receiver, receiver_room, receiver_room);
      receiver_obs.added.Clear();
    }
    return ::testing::AssertionSuccess();
  }

  // A joiner that connects now, after whatever happened so far. TCP only. The
  // rig owns it; its observer is attached before the connect because the
  // bootstrap replays the live shares.
  Observer* LateJoin() {
    sb_session_t session = sb_create_session();
    if (!session) return nullptr;
    late_sessions_.push_back(session);
    const sb_space_t space = sb_add_space(session);
    if (!space) return nullptr;
    late_spaces_.push_back(space);
    late_observers_.push_back(std::make_unique<Observer>());
    Observer* obs = late_observers_.back().get();
    obs->Attach(space);
    const sb_handle_t c = sb_connect_space(space, "127.0.0.1", port_);
    if (!c || !sb_try_wait(c, 5000)) return nullptr;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return obs;
  }

  // The sharer shares a fresh child of its room carrying `shared_id`.
  sb_element_t Share(int64_t shared_id) {
    const sb_element_t e = sb_add_element(sharer, sharer_room);
    sb_set_element_shared_id(sharer, e, shared_id);
    sb_share_element(sharer, e, nullptr, 0);
    return e;
  }
  void ShareAgain(sb_element_t e) { sb_share_element(sharer, e, nullptr, 0); }

  // The receiver links the `n`th received share onto a new element of its own
  // (carrying `local_shared_id`) and returns it.
  sb_element_t LinkReceived(size_t n_th, int64_t local_shared_id) {
    if (!receiver_obs.added.WaitFor(n_th + 1)) return 0;
    const Added got = receiver_obs.added.Snapshot()[n_th];
    const sb_element_t local = sb_add_element(receiver, receiver_room);
    sb_set_element_shared_id(receiver, local, local_shared_id);
    sb_resolve_shared_element(receiver, got.local, local);
    return local;
  }

  // The receiver keeps the `n`th received share as a new element.
  sb_element_t KeepReceived(size_t n_th) {
    if (!receiver_obs.added.WaitFor(n_th + 1)) return 0;
    const sb_element_t replica = receiver_obs.added.Snapshot()[n_th].local;
    sb_resolve_shared_element(receiver, replica, replica);
    return replica;
  }

  bool Has(sb_space_t space, int64_t shared_id, sb_element_t elem) const {
    return sb_find_element_by_shared_id(space, shared_id) == elem;
  }

  sb_space_t sharer = 0, receiver = 0;
  sb_element_t sharer_room = 0, receiver_room = 0;
  Observer sharer_obs, receiver_obs;

 private:
  sb_session_t session_a_ = 0, session_b_ = 0;
  int32_t port_ = 0;
  std::vector<sb_space_t> late_spaces_;
  std::vector<sb_session_t> late_sessions_;
  std::vector<std::unique_ptr<Observer>> late_observers_;
};

class UnlinkCTest : public ::testing::TestWithParam<Transport> {};

INSTANTIATE_TEST_SUITE_P(
    Transports, UnlinkCTest,
    ::testing::Values(Transport::kLocal, Transport::kTcpHostShares,
                      Transport::kTcpClientShares),
    [](const ::testing::TestParamInfo<Transport>& info) {
      return std::string(Name(info.param));
    });

}  // namespace

// (a) The sharer unshares while the receiver stays linked: the receiver is told
// once, keeps its element, and holds no link any more.
TEST_P(UnlinkCTest, SharerUnshareReleasesTheReceiversLink) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(GetParam()));
  const sb_element_t source = rig.Share(kShareA);
  const sb_element_t local = rig.LinkReceived(0, kLocalA);
  ASSERT_NE(local, 0);

  sb_unshare_element(rig.sharer, source);
  ASSERT_TRUE(rig.receiver_obs.link_removed.WaitFor(1));
  const auto events = rig.receiver_obs.link_removed.Snapshot();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].local, local);
  EXPECT_EQ(events[0].reason, SB_LINK_REMOVED_PEER_UNSHARED);
  EXPECT_NE(events[0].peer, 0);
  Settle();
  EXPECT_EQ(rig.receiver_obs.link_removed.Size(), 1u) << "exactly once";
  EXPECT_EQ(rig.receiver_obs.removed.Size(), 0u)
      << "the receiver's own element is not removed";
  EXPECT_EQ(rig.sharer_obs.link_removed.Size(), 0u);
  EXPECT_TRUE(rig.Has(rig.receiver, kLocalA, local)) << "the element survived";
  EXPECT_EQ(sb_unlink(rig.receiver, local), 0) << "the link is already gone";
}

// (b) The receiver unlinks while the sharer still shares: the sharer is told,
// the receiver's element is kept, and the share stays the sharer's.
TEST_P(UnlinkCTest, ReceiverUnlinkNotifiesTheSharerAndKeepsTheElement) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(GetParam()));
  const sb_element_t source = rig.Share(kShareA);
  const sb_element_t local = rig.LinkReceived(0, kLocalA);
  ASSERT_NE(local, 0);

  EXPECT_EQ(sb_unlink(rig.receiver, local), 1);
  ASSERT_TRUE(rig.sharer_obs.link_removed.WaitFor(1));
  const auto events = rig.sharer_obs.link_removed.Snapshot();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].local, source);
  EXPECT_EQ(events[0].reason, SB_LINK_REMOVED_PEER_UNLINKED);
  EXPECT_NE(events[0].peer, 0);
  if (GetParam() != Transport::kTcpHostShares) {
    // The receiver is the local-hub neighbour or the host: its id is its handle.
    EXPECT_EQ(events[0].peer, static_cast<int64_t>(rig.receiver));
  }
  Settle();
  EXPECT_EQ(rig.sharer_obs.link_removed.Size(), 1u);
  EXPECT_EQ(rig.receiver_obs.link_removed.Size(), 0u)
      << "the caller is not told about its own unlink";
  EXPECT_EQ(sb_unlink(rig.receiver, local), 0) << "already unlinked";
  EXPECT_TRUE(rig.Has(rig.receiver, kLocalA, local)) << "the element is kept";

  // The share is still the sharer's; unsharing it later tells nobody (there is
  // no link left to release) and removes nothing on the receiver.
  sb_unshare_element(rig.sharer, source);
  Settle();
  EXPECT_EQ(rig.receiver_obs.link_removed.Size(), 0u);
  EXPECT_EQ(rig.receiver_obs.removed.Size(), 0u);
  EXPECT_TRUE(rig.Has(rig.receiver, kLocalA, local));
}

// After an unlink the sharer shares again and the receiver links anew; the new
// link ends like the first, in either order.
TEST_P(UnlinkCTest, ReShareAfterUnlinkLinksAnew) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(GetParam()));
  const sb_element_t source = rig.Share(kShareA);
  const sb_element_t first = rig.LinkReceived(0, kLocalA);
  ASSERT_NE(first, 0);
  ASSERT_EQ(sb_unlink(rig.receiver, first), 1);
  ASSERT_TRUE(rig.sharer_obs.link_removed.WaitFor(1));

  rig.ShareAgain(source);
  const sb_element_t second = rig.LinkReceived(1, kLocalB);
  ASSERT_NE(second, 0);

  sb_unshare_element(rig.sharer, source);
  ASSERT_TRUE(rig.receiver_obs.link_removed.WaitFor(1));
  const auto events = rig.receiver_obs.link_removed.Snapshot();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].local, second) << "the new link, not the first";
  EXPECT_EQ(events[0].reason, SB_LINK_REMOVED_PEER_UNSHARED);

  rig.ShareAgain(source);
  const sb_element_t third = rig.LinkReceived(2, kLocalA + 100);
  ASSERT_NE(third, 0);
  EXPECT_EQ(sb_unlink(rig.receiver, third), 1);
  ASSERT_TRUE(rig.sharer_obs.link_removed.WaitFor(2));
  EXPECT_EQ(rig.sharer_obs.link_removed.Snapshot()[1].reason,
            SB_LINK_REMOVED_PEER_UNLINKED);
  EXPECT_EQ(rig.sharer_obs.link_removed.Snapshot()[1].local, source);
}

// An AddAsNew replica is released into local ownership by an unlink: not
// deleted, and a later unshare of the source no longer removes it.
TEST_P(UnlinkCTest, UnlinkingAReplicaKeepsItAsALocalElement) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(GetParam()));
  const sb_element_t source = rig.Share(kShareA);
  const sb_element_t replica = rig.KeepReceived(0);
  ASSERT_NE(replica, 0);
  ASSERT_TRUE(rig.Has(rig.receiver, kShareA, replica));

  EXPECT_EQ(sb_unlink(rig.receiver, replica), 1);
  ASSERT_TRUE(rig.sharer_obs.link_removed.WaitFor(1));
  EXPECT_EQ(rig.sharer_obs.link_removed.Snapshot()[0].local, source);

  sb_unshare_element(rig.sharer, source);
  Settle();
  EXPECT_EQ(rig.receiver_obs.removed.Size(), 0u)
      << "the released replica is no longer the source's to remove";
  EXPECT_TRUE(rig.Has(rig.receiver, kShareA, replica));
}

// The contrast: an AddAsNew replica the sharer's unshare removes is reported as
// a removal, not as a surviving link end.
TEST_P(UnlinkCTest, UnshareOfALinkedReplicaIsARemovalNotALinkRemoval) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(GetParam()));
  const sb_element_t source = rig.Share(kShareA);
  const sb_element_t replica = rig.KeepReceived(0);
  ASSERT_NE(replica, 0);
  sb_unshare_element(rig.sharer, source);
  ASSERT_TRUE(rig.receiver_obs.removed.WaitFor(1));
  EXPECT_EQ(rig.receiver_obs.removed.Snapshot()[0], replica);
  Settle();
  EXPECT_EQ(rig.receiver_obs.link_removed.Size(), 0u);
}

// Only the named link goes: another share linked in the same pair of spaces
// keeps its link, and ending one source does not report the other.
TEST_P(UnlinkCTest, UnrelatedLinksAreUntouched) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(GetParam()));
  const sb_element_t source_a = rig.Share(kShareA);
  const sb_element_t source_b = rig.Share(kShareB);
  const sb_element_t local_a = rig.LinkReceived(0, kLocalA);
  const sb_element_t local_b = rig.LinkReceived(1, kLocalB);
  ASSERT_NE(local_a, 0);
  ASSERT_NE(local_b, 0);

  EXPECT_EQ(sb_unlink(rig.receiver, local_a), 1);
  ASSERT_TRUE(rig.sharer_obs.link_removed.WaitFor(1));
  EXPECT_EQ(rig.sharer_obs.link_removed.Snapshot()[0].local, source_a);

  sb_unshare_element(rig.sharer, source_a);  // already unlinked: nobody told
  Settle();
  EXPECT_EQ(rig.receiver_obs.link_removed.Size(), 0u);

  sb_unshare_element(rig.sharer, source_b);
  ASSERT_TRUE(rig.receiver_obs.link_removed.WaitFor(1));
  const auto events = rig.receiver_obs.link_removed.Snapshot();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].local, local_b) << "B's link was intact until now";
  EXPECT_EQ(sb_unlink(rig.receiver, local_b), 0);
}

TEST_P(UnlinkCTest, ArgumentErrorsAndASharersOwnElementAreNotUnlinked) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(GetParam()));
  EXPECT_EQ(sb_unlink(0, 1), 0);
  EXPECT_EQ(sb_unlink(rig.receiver, 0), 0);
  EXPECT_EQ(sb_unlink(rig.receiver, 987654), 0);
  const sb_element_t source = rig.Share(kShareA);
  const sb_element_t local = rig.LinkReceived(0, kLocalA);
  ASSERT_NE(local, 0);
  EXPECT_EQ(sb_unlink(rig.sharer, source), 0)
      << "a sharer's own source is unshared, not unlinked";
  EXPECT_EQ(rig.sharer_obs.link_removed.Size(), 0u);
  EXPECT_EQ(sb_unlink(rig.receiver, local), 1)
      << "the refused call left the link intact";
}

// A pending share (the app has not resolved it) is not a link to remove.
TEST_P(UnlinkCTest, AnUnresolvedShareIsNotUnlinkable) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(GetParam()));
  rig.Share(kShareA);
  ASSERT_TRUE(rig.receiver_obs.added.WaitFor(1));
  const sb_element_t temp = rig.receiver_obs.added.Snapshot()[0].local;
  EXPECT_EQ(sb_unlink(rig.receiver, temp), 0);
  Settle();
  EXPECT_EQ(rig.sharer_obs.link_removed.Size(), 0u);
}

// An unlinked element is the app's again: with no link and no replica
// ownership, sb_remove_local_element accepts it (it refuses a linked one).
TEST(UnlinkCLocalTest, AnUnlinkedElementCanBeRemoved) {
  sb_session_t session = sb_create_session();
  ASSERT_NE(session, 0);
  const sb_space_t sharer = sb_add_space(session);
  const sb_space_t receiver = sb_add_space(session);
  Observer sharer_obs, receiver_obs;
  sharer_obs.Attach(sharer);
  receiver_obs.Attach(receiver);

  const sb_element_t room = sb_add_element(receiver, 0);  // its own, not shared
  const sb_element_t source = sb_add_element(sharer, 0);
  sb_share_element(sharer, source, nullptr, 0);
  ASSERT_EQ(receiver_obs.added.Size(), 1u);
  const sb_element_t local = sb_add_element(receiver, room);
  sb_resolve_shared_element(receiver, receiver_obs.added.Snapshot()[0].local,
                            local);
  EXPECT_EQ(sb_remove_local_element(receiver, local), 0) << "linked: refused";
  EXPECT_EQ(sb_unlink(receiver, local), 1);
  EXPECT_EQ(sb_remove_local_element(receiver, local), 1) << "unlinked: allowed";

  Observer::Detach(sharer);
  Observer::Detach(receiver);
  sb_destroy_session(session);
}

// Two receivers on one hub link the same share: one unlinking does not touch
// the other, and the sharer's unshare still reaches the one that stayed.
TEST(UnlinkCLocalTest, AnotherReceiversLinkIsUntouched) {
  sb_session_t session = sb_create_session();
  ASSERT_NE(session, 0);
  const sb_space_t sharer = sb_add_space(session);
  const sb_space_t r1 = sb_add_space(session);
  const sb_space_t r2 = sb_add_space(session);
  Observer sharer_obs, o1, o2;
  sharer_obs.Attach(sharer);
  o1.Attach(r1);
  o2.Attach(r2);

  const sb_element_t source = sb_add_element(sharer, 0);
  sb_set_element_shared_id(sharer, source, kShareA);
  sb_share_element(sharer, source, nullptr, 0);
  ASSERT_EQ(o1.added.Size(), 1u);
  ASSERT_EQ(o2.added.Size(), 1u);
  const sb_element_t l1 = sb_add_element(r1, 0);
  const sb_element_t l2 = sb_add_element(r2, 0);
  sb_resolve_shared_element(r1, o1.added.Snapshot()[0].local, l1);
  sb_resolve_shared_element(r2, o2.added.Snapshot()[0].local, l2);

  EXPECT_EQ(sb_unlink(r1, l1), 1);
  ASSERT_EQ(sharer_obs.link_removed.Size(), 1u);
  EXPECT_EQ(sharer_obs.link_removed.Snapshot()[0].peer,
            static_cast<int64_t>(r1));
  EXPECT_EQ(o2.link_removed.Size(), 0u);

  sb_unshare_element(sharer, source);
  ASSERT_EQ(o2.link_removed.Size(), 1u) << "r2 stayed linked until the unshare";
  EXPECT_EQ(o2.link_removed.Snapshot()[0].local, l2);
  EXPECT_EQ(o1.link_removed.Size(), 0u) << "r1 had already let go";

  Observer::Detach(sharer);
  Observer::Detach(r1);
  Observer::Detach(r2);
  sb_destroy_session(session);
}

// Both receivers on one hub link, then the sharer unshares: each is told once
// and each keeps its element. One receiver's cleanup must not hide the other's.
TEST(UnlinkCLocalTest, UnshareReleasesEveryReceiverOnTheHub) {
  sb_session_t session = sb_create_session();
  ASSERT_NE(session, 0);
  const sb_space_t sharer = sb_add_space(session);
  const sb_space_t r1 = sb_add_space(session);
  const sb_space_t r2 = sb_add_space(session);
  Observer sharer_obs, o1, o2;
  sharer_obs.Attach(sharer);
  o1.Attach(r1);
  o2.Attach(r2);
  const sb_element_t source = sb_add_element(sharer, 0);
  sb_share_element(sharer, source, nullptr, 0);
  ASSERT_EQ(o1.added.Size(), 1u);
  ASSERT_EQ(o2.added.Size(), 1u);
  const sb_element_t l1 = sb_add_element(r1, 0);
  const sb_element_t l2 = sb_add_element(r2, 0);
  sb_resolve_shared_element(r1, o1.added.Snapshot()[0].local, l1);
  sb_resolve_shared_element(r2, o2.added.Snapshot()[0].local, l2);

  sb_unshare_element(sharer, source);
  ASSERT_EQ(o1.link_removed.Size(), 1u);
  ASSERT_EQ(o2.link_removed.Size(), 1u);
  EXPECT_EQ(o1.link_removed.Snapshot()[0].local, l1);
  EXPECT_EQ(o2.link_removed.Snapshot()[0].local, l2);
  EXPECT_EQ(sb_unlink(r1, l1), 0);
  EXPECT_EQ(sb_unlink(r2, l2), 0);

  Observer::Detach(sharer);
  Observer::Detach(r1);
  Observer::Detach(r2);
  sb_destroy_session(session);
}

// ---------------------------------------------------------------------------
// Late joiners: the host's table of live shares must drop a share the
// moment the host's link to it ends, so a joiner is never handed it through an
// element the host no longer stands behind.
// ---------------------------------------------------------------------------

// Control: a share the host received and linked IS handed to a late joiner.
TEST(UnlinkCLateJoinTest, ControlALinkedShareReachesALateJoiner) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(Transport::kTcpClientShares));
  rig.Share(kShareA);
  ASSERT_NE(rig.LinkReceived(0, kLocalA), 0);
  Observer* late = rig.LateJoin();
  ASSERT_NE(late, nullptr);
  EXPECT_TRUE(late->WaitForShared(kShareA)) << "the live share is bootstrapped";
}

// The host (a receiver) unlinks the client's share: a late joiner gets nothing
// through the host element that no longer stands for it. (The bootstrap also
// skips a share whose host element is gone; HubReceiver drops the table entry
// itself so it neither lingers nor warns on every later join.)
TEST(UnlinkCLateJoinTest, HostUnlinkLeavesNoStaleShareForALateJoiner) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(Transport::kTcpClientShares));
  rig.Share(kShareA);
  const sb_element_t host_local = rig.LinkReceived(0, kLocalA);
  ASSERT_NE(host_local, 0);
  ASSERT_EQ(sb_unlink(rig.receiver, host_local), 1);
  ASSERT_TRUE(rig.sharer_obs.link_removed.WaitFor(1));

  Observer* late = rig.LateJoin();
  ASSERT_NE(late, nullptr);
  Settle();
  EXPECT_FALSE(late->HasShared(kShareA));
  EXPECT_EQ(late->link_removed.Size(), 0u);
}

// The client unshares a share the host had linked: the unshare already drops it
// from the table; the link release is the new part.
TEST(UnlinkCLateJoinTest, UnshareLeavesNoStaleShareForALateJoiner) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(Transport::kTcpClientShares));
  const sb_element_t source = rig.Share(kShareA);
  ASSERT_NE(rig.LinkReceived(0, kLocalA), 0);
  sb_unshare_element(rig.sharer, source);
  ASSERT_TRUE(rig.receiver_obs.link_removed.WaitFor(1));

  Observer* late = rig.LateJoin();
  ASSERT_NE(late, nullptr);
  Settle();
  EXPECT_FALSE(late->HasShared(kShareA));
}

// The client (a receiver) unlinks a share the HOST still shares: the share is
// still live for everyone else, so a late joiner receives it.
TEST(UnlinkCLateJoinTest, AClientUnlinkDoesNotWithdrawTheHostsShare) {
  Rig rig;
  ASSERT_TRUE(rig.Setup(Transport::kTcpHostShares));
  rig.Share(kShareA);
  const sb_element_t client_local = rig.LinkReceived(0, kLocalA);
  ASSERT_NE(client_local, 0);
  ASSERT_EQ(sb_unlink(rig.receiver, client_local), 1);
  ASSERT_TRUE(rig.sharer_obs.link_removed.WaitFor(1));

  Observer* late = rig.LateJoin();
  ASSERT_NE(late, nullptr);
  EXPECT_TRUE(late->WaitForShared(kShareA));
}
