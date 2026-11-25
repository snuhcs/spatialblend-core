// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Tunneled core link (ABI 4.5), included by spatialblend_c.cc after
// sb_connect_space. See the block comment in spatialblend.h.
//
// Registries (g_tunnel_mutex only; never held while calling into gRPC or while
// taking g_sessions_mutex / g_spaces_mutex):
//   g_tunnels    sb_tunnel_t -> the link, the session that owns it, and (on a
//                joiner's link) its join
//   g_async_ops  sb_handle_t -> the state of one tunneled join, which runs on
//                its own thread because joining blocks until the host answers
//                and that needs the app to pump. The thread only stages the
//                bootstrap; the app thread applies it in sb_try_wait.

namespace {

// One tunneled join. The worker thread blocks in ConnectToHub (which needs the
// app to pump the link) and only STAGES the bootstrap; the app thread applies it
// to the space in sb_try_wait, so no other thread changes the space on behalf
// of a join. Nothing here has a timer: closing the link, or retiring the
// session, fails a pending join.
struct TunnelAsyncOp {
  std::mutex mu;
  std::condition_variable cv;
  int state = 0;  // 0 pending, 1 completed, 2 failed
  std::string error;
  std::thread worker;
  sb_session_t session = 0;
  sb_space_t space = 0;
  SessionContext* ctx = nullptr;              // valid until the session is retired
  SessionContext::SpaceInfo* info = nullptr;  // likewise
  std::shared_ptr<spatialblend::RemoteHub> hub;  // set once staged
  bool staged = false;     // the bootstrap arrived and waits for the app thread
  bool applied = false;    // the app thread applied it
  bool apply_failed = false;
  bool cancelled = false;  // the link was closed or the session retired
  bool wired = false;      // the app thread installed the hub on the space
  bool released = false;   // the join was torn down by closing its link
};

struct TunnelEntry {
  sb_session_t session = 0;
  std::shared_ptr<spatialblend::TunnelLink> link;
  std::shared_ptr<TunnelAsyncOp> join;  // set on a joiner's link
};

// Leaked on purpose: a joinable std::thread destroyed by static destruction
// calls std::terminate, and nothing may run after main returns.
std::mutex& g_tunnel_mutex = *new std::mutex;
std::unordered_map<sb_tunnel_t, TunnelEntry>& g_tunnels =
    *new std::unordered_map<sb_tunnel_t, TunnelEntry>;
std::unordered_map<sb_handle_t, std::shared_ptr<TunnelAsyncOp>>& g_async_ops =
    *new std::unordered_map<sb_handle_t, std::shared_ptr<TunnelAsyncOp>>;
std::atomic<sb_tunnel_t> g_next_tunnel_id{1};
// Distinct from 0 (failure), kSyncDoneHandle (-1) and the small ids other
// handle kinds may use.
std::atomic<sb_handle_t> g_next_async_handle{1LL << 32};

std::shared_ptr<spatialblend::TunnelLink> FindTunnel(sb_tunnel_t link) {
  std::lock_guard<std::mutex> lock(g_tunnel_mutex);
  auto it = g_tunnels.find(link);
  return it == g_tunnels.end() ? nullptr : it->second.link;
}

sb_tunnel_t RegisterTunnel(sb_session_t session,
                           std::shared_ptr<spatialblend::TunnelLink> link,
                           std::shared_ptr<TunnelAsyncOp> join = nullptr) {
  const sb_tunnel_t id = g_next_tunnel_id.fetch_add(1);
  std::lock_guard<std::mutex> lock(g_tunnel_mutex);
  g_tunnels[id] = {session, std::move(link), std::move(join)};
  return id;
}

void FinishAsyncOp(const std::shared_ptr<TunnelAsyncOp>& op, bool ok,
                   std::string error) {
  {
    std::lock_guard<std::mutex> lock(op->mu);
    op->state = ok ? 1 : 2;
    op->error = std::move(error);
  }
  op->cv.notify_all();
}

void CancelAsyncOp(const std::shared_ptr<TunnelAsyncOp>& op) {
  {
    std::lock_guard<std::mutex> lock(op->mu);
    op->cancelled = true;
  }
  op->cv.notify_all();
}

// Joins the threads of joins that have finished. They are done with everything
// but returning, so this does not wait on a peer.
void ReapFinishedOps() {
  std::vector<std::shared_ptr<TunnelAsyncOp>> done;
  {
    std::lock_guard<std::mutex> lock(g_tunnel_mutex);
    for (auto& [handle, op] : g_async_ops) {
      std::lock_guard<std::mutex> olock(op->mu);
      if (op->state != 0) done.push_back(op);
    }
  }
  for (auto& op : done) {
    if (op->worker.joinable() &&
        op->worker.get_id() != std::this_thread::get_id()) {
      op->worker.join();
    }
  }
}

// The session owning `space`, or nullptr. Lock order: g_sessions_mutex before
// g_spaces_mutex. A path that starts from a space therefore never holds both:
// SessionContextOfSpace reads the session id, releases g_spaces_mutex, and only
// then takes g_sessions_mutex.
SessionContext* SessionOfSpace(sb_space_t space) {
  return SessionContextOfSpace(space);
}

sb_session_t SessionIdOfSpace(sb_space_t space) {
  std::lock_guard<std::mutex> lock(g_spaces_mutex);
  auto sit = g_space_to_session.find(space);
  return sit == g_space_to_session.end() ? 0 : sit->second;
}

// Runs on the APP thread (sb_try_wait): applies a staged bootstrap to the
// space, and, once the join has completed, installs the hub on the space.
void ServiceAsyncOp(const std::shared_ptr<TunnelAsyncOp>& op) {
  std::shared_ptr<spatialblend::RemoteHub> to_apply;
  {
    std::lock_guard<std::mutex> lock(op->mu);
    if (op->staged && !op->applied && !op->cancelled) to_apply = op->hub;
  }
  if (to_apply) {
    const bool ok = to_apply->ApplyStagedBootstrap().ok();
    {
      std::lock_guard<std::mutex> lock(op->mu);
      op->applied = true;
      op->apply_failed = !ok;
    }
    op->cv.notify_all();
  }
  std::shared_ptr<spatialblend::RemoteHub> to_wire;
  {
    std::lock_guard<std::mutex> lock(op->mu);
    if (op->state == 1 && !op->wired && !op->released) {
      op->wired = true;
      to_wire = op->hub;
    }
  }
  if (to_wire) {
    op->info->remote_hub = to_wire;
    WireClientHub(op->ctx, to_wire);
  }
}

// Undoes a completed join whose link was closed, so the session can join again.
// Caller holds no lock and has already joined the op's worker.
void ReleaseClientRole(const std::shared_ptr<TunnelAsyncOp>& op) {
  std::shared_ptr<spatialblend::RemoteHub> hub;
  bool completed = false;
  {
    std::lock_guard<std::mutex> lock(op->mu);
    if (op->released) return;
    op->released = true;
    completed = op->state == 1;
    hub = std::move(op->hub);
  }
  if (!completed) return;  // a failed join released the role itself
  SessionContext* ctx = op->ctx;
  {
    std::lock_guard<std::mutex> slock(g_sessions_mutex);
    ctx->context->GetHub()->ClearCallbacks();
  }
  if (hub) (void)hub->PrepareDisconnect();
  op->info->remote_hub.reset();
  hub.reset();
  std::lock_guard<std::mutex> slock(g_sessions_mutex);
  ctx->hub_role = HubRole::kNone;
}

}  // namespace

static int32_t TunnelAsyncTryWait(sb_handle_t handle, int32_t timeout_ms) {
  std::shared_ptr<TunnelAsyncOp> op;
  {
    std::lock_guard<std::mutex> lock(g_tunnel_mutex);
    auto it = g_async_ops.find(handle);
    if (it == g_async_ops.end()) return 0;
    op = it->second;
  }
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 0);
  for (;;) {
    ServiceAsyncOp(op);
    std::unique_lock<std::mutex> lock(op->mu);
    if (op->state != 0 || timeout_ms <= 0 ||
        std::chrono::steady_clock::now() >= deadline) {
      const bool ok = op->state == 1;
      const bool finished = op->state != 0;
      lock.unlock();
      if (finished) ReapFinishedOps();
      return ok ? 1 : 0;
    }
    // Woken when the bootstrap is staged (to apply it here) or the join ends.
    op->cv.wait_until(lock, deadline, [&] {
      return op->state != 0 || op->cancelled || (op->staged && !op->applied);
    });
  }
}

static const char* TunnelAsyncError(sb_handle_t handle) {
  std::shared_ptr<TunnelAsyncOp> op;
  {
    std::lock_guard<std::mutex> lock(g_tunnel_mutex);
    auto it = g_async_ops.find(handle);
    if (it == g_async_ops.end()) return nullptr;
    op = it->second;
  }
  std::lock_guard<std::mutex> lock(op->mu);
  // The string lives as long as the op (until its session is retired).
  return op->state == 2 && !op->error.empty() ? op->error.c_str() : nullptr;
}

// Closes the links of every session for which `match` is true, fails and joins
// their joins. The caller holds no registry lock.
template <typename Match>
static void RetireTunnelsIf(Match match) {
  std::vector<std::shared_ptr<spatialblend::TunnelLink>> links;
  std::vector<std::shared_ptr<TunnelAsyncOp>> ops;
  {
    std::lock_guard<std::mutex> lock(g_tunnel_mutex);
    for (auto it = g_tunnels.begin(); it != g_tunnels.end();) {
      if (match(it->second.session)) {
        links.push_back(it->second.link);
        it = g_tunnels.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = g_async_ops.begin(); it != g_async_ops.end();) {
      if (match(it->second->session)) {
        ops.push_back(it->second);
        it = g_async_ops.erase(it);
      } else {
        ++it;
      }
    }
  }
  // A join still waiting on its host fails as soon as its link ends.
  for (auto& op : ops) CancelAsyncOp(op);
  for (auto& link : links) link->Close();
  for (auto& op : ops) {
    if (op->worker.joinable()) op->worker.join();
  }
}

// Closes every link of the session and joins its join threads. Runs before
// sb_destroy_session takes any registry lock.
static void RetireTunnelsForSession(sb_session_t session) {
  RetireTunnelsIf([session](sb_session_t s) { return s == session; });
}

// Same for every session (sb_cleanup_all).
static void RetireAllTunnels() {
  RetireTunnelsIf([](sb_session_t) { return true; });
}

sb_handle_t sb_host_space_tunneled(sb_space_t space) {
  return HostSpaceImpl(space, 0, /*tunneled=*/true);
}

int32_t sb_accept_tunnel(sb_space_t space, sb_tunnel_t* out_link) {
  if (!out_link) return -1;
  *out_link = 0;
  auto* info = GetSpace(space);
  if (!info || !info->space) return -1;
  const sb_session_t session = SessionIdOfSpace(space);
  auto receiver = info->receiver;
  if (!session || !receiver) return 0;
  auto link = receiver->AcceptTunnel();
  if (!link.ok()) return 0;
  *out_link = RegisterTunnel(
      session, std::shared_ptr<spatialblend::TunnelLink>(std::move(*link)));
  return 1;
}

sb_handle_t sb_connect_space_tunneled(sb_space_t space, sb_tunnel_t* out_link) {
  if (!out_link) return 0;
  *out_link = 0;
  auto* info = GetSpace(space);
  if (!info || !info->space) return 0;
  SessionContext* ctx = SessionOfSpace(space);
  const sb_session_t session = SessionIdOfSpace(space);
  if (!ctx || !ctx->context || !session) return 0;
  ReapFinishedOps();

  // One session = one hub role, as for sb_connect_space.
  {
    std::lock_guard<std::mutex> slock(g_sessions_mutex);
    if (ctx->hub_role != HubRole::kNone) return 0;
    ctx->hub_role = HubRole::kClient;
  }
  auto release_role = [ctx] {
    std::lock_guard<std::mutex> slock(g_sessions_mutex);
    ctx->hub_role = HubRole::kNone;
  };

  auto created = spatialblend::TunnelLink::Create();
  if (!created.ok()) {
    release_role();
    return 0;
  }
  std::shared_ptr<spatialblend::TunnelLink> link(std::move(*created));
  auto channel = spatialblend::RemoteHub::MakeTunnelChannel(link.get());
  if (!channel) {
    release_role();
    return 0;
  }

  auto op = std::make_shared<TunnelAsyncOp>();
  op->session = session;
  op->space = space;
  op->ctx = ctx;
  op->info = info;
  const sb_handle_t handle = g_next_async_handle.fetch_add(1);
  *out_link = RegisterTunnel(session, link, op);
  const spatialblend::space_id_t space_id = info->space->GetSpaceId();
  {
    std::lock_guard<std::mutex> lock(g_tunnel_mutex);
    g_async_ops[handle] = op;
  }
  op->worker = std::thread([op, info, channel, link, space_id,
                            release_role]() mutable {
    auto fail = [&](const char* why) {
      link->Close();
      release_role();
      FinishAsyncOp(op, false, why);
    };
    // Blocks until the host answers ConnectToHub; the link ending fails it.
    // Changes nothing in the space.
    auto hub = spatialblend::RemoteHub::ConnectOnChannel(info->space, channel,
                                                         space_id);
    channel.reset();
    if (!hub) {
      fail("The host did not complete the core link over the tunnel "
           "(the link ended before the host answered).");
      return;
    }
    // Hand the bootstrap to the app thread and wait for it to apply it.
    {
      std::unique_lock<std::mutex> lock(op->mu);
      op->hub = hub;
      op->staged = true;
      op->cv.notify_all();
      op->cv.wait(lock, [&] { return op->applied || op->cancelled; });
      if (op->cancelled || op->apply_failed) {
        const bool cancelled = op->cancelled;
        op->hub.reset();
        lock.unlock();
        fail(cancelled ? "The link ended before the join was applied."
                       : "The host's bootstrap could not be applied.");
        return;
      }
    }
    // No timer: the bootstrap completes, or the link ends and the call fails.
    if (!hub->WaitForBootstrapComplete(-1).ok() || !hub->IsConnected()) {
      {
        std::lock_guard<std::mutex> lock(op->mu);
        op->hub.reset();
      }
      fail("The host did not complete the core link over the tunnel "
           "(the link ended or the bootstrap did not arrive).");
      return;
    }
    FinishAsyncOp(op, true, std::string());
  });
  return handle;
}

int32_t sb_tunnel_write(sb_tunnel_t link, const uint8_t* bytes, uint32_t n) {
  auto l = FindTunnel(link);
  if (!l || (n > 0 && !bytes)) return -1;
  const absl::Status status = l->Write(bytes, n);
  if (status.ok()) return 1;
  return absl::IsUnavailable(status) ? 2 : 0;
}

int32_t sb_tunnel_read(sb_tunnel_t link, uint8_t* buf, uint32_t cap,
                       uint32_t* n) {
  auto l = FindTunnel(link);
  if (!l || !n || (cap > 0 && !buf)) return -1;
  size_t got = 0;
  const auto result = l->Read(buf, cap, &got);
  *n = static_cast<uint32_t>(got);
  return result == spatialblend::TunnelLink::ReadResult::kOk ? 1 : 0;
}

void sb_tunnel_close(sb_tunnel_t link) {
  TunnelEntry entry;
  {
    std::lock_guard<std::mutex> lock(g_tunnel_mutex);
    auto it = g_tunnels.find(link);
    if (it == g_tunnels.end()) return;
    entry = std::move(it->second);
    g_tunnels.erase(it);
  }
  entry.link->Close();
  if (entry.join) {
    // The joiner leaves the room: a join still waiting fails now, a completed
    // one is undone, and the session may join again.
    CancelAsyncOp(entry.join);
    if (entry.join->worker.joinable() &&
        entry.join->worker.get_id() != std::this_thread::get_id()) {
      entry.join->worker.join();
    }
    ReleaseClientRole(entry.join);
  }
}
