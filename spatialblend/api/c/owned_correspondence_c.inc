// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Independent optional feature over the existing correspondence pair ABI.
uint32_t sb_get_owned_correspondence_feature_version() { return 1; }

sb_support_correspondence_registration_t sb_try_register_owned_correspondences(
    sb_session_t session, uint32_t version,
    const sb_support_correspondence_t* pairs, uint32_t count) {
  if (version != 1 || !pairs || count == 0 || count > 128 ||
      g_inside_relation_callback)
    return 0;
  std::unique_lock<std::mutex> lock(g_sessions_mutex, std::try_to_lock);
  if (!lock.owns_lock()) return 0;
  auto owner = g_sessions.find(session);
  if (owner == g_sessions.end() || owner->second->rich_retiring ||
      !owner->second->context)
    return 0;
  auto& context = *owner->second;
  std::vector<
      std::pair<spatialblend::element_handle_t, spatialblend::element_handle_t>>
      links;
  for (uint32_t i = 0; i < count; ++i) {
    const auto& p = pairs[i];
    auto a = context.spaces.find(p.source_space),
         b = context.spaces.find(p.target_space);
    if (p.source_element <= 0 || p.target_element <= 0 ||
        p.source_space == p.target_space || a == context.spaces.end() ||
        b == context.spaces.end() ||
        !a->second->space->TryHasElementOutsideCallback(p.source_element) ||
        !b->second->space->TryHasElementOutsideCallback(p.target_element))
      return 0;
    links.push_back({{p.source_space, p.source_element},
                     {p.target_space, p.target_element}});
  }
  const auto handle = ReserveSupportRegistrationId();
  if (!handle) return 0;
  SupportCorrespondenceRegistration registration{session, {}, true};
  auto endpoints_live = [&]() {
    for (const auto& [a, b] : links)
      if (!context.spaces.at(a.space_id)
               ->space->TryHasElementOutsideCallback(a.element_id) ||
          !context.spaces.at(b.space_id)
               ->space->TryHasElementOutsideCallback(b.element_id))
        return false;
    return true;
  };
  if (!context.context->GetHub()->GetLinker().TryInstallOwnedLinks(
          links, &registration.installed, endpoints_live))
    return 0;
  g_support_registrations.emplace(handle, std::move(registration));
  return handle;
}

int32_t sb_try_validate_owned_correspondences(
    sb_session_t session, sb_support_correspondence_registration_t registration,
    uint32_t version) {
  if (version != 1 || registration <= 0) return -1;
  if (g_inside_relation_callback) return -4;
  std::unique_lock<std::mutex> lock(g_sessions_mutex, std::try_to_lock);
  if (!lock.owns_lock()) return -4;
  auto owner = g_sessions.find(session);
  auto registered = g_support_registrations.find(registration);
  if (owner == g_sessions.end() || owner->second->rich_retiring ||
      !owner->second->context || registered == g_support_registrations.end() ||
      registered->second.session != session || !registered->second.strict_owned)
    return 0;
  auto endpoints_live = [&]() {
    for (const auto& link : registered->second.installed) {
      auto a = owner->second->spaces.find(link.source.space_id);
      auto b = owner->second->spaces.find(link.target.space_id);
      if (a == owner->second->spaces.end() ||
          b == owner->second->spaces.end() ||
          !a->second->space->TryHasElementOutsideCallback(
              link.source.element_id) ||
          !b->second->space->TryHasElementOutsideCallback(
              link.target.element_id))
        return false;
    }
    return true;
  };
  // Nonblocking endpoint checks under the linker read lock fence owned
  // subtree deletion. The result is still only an as-of-call value.
  return owner->second->context->GetHub()->GetLinker().TryValidateOwnedLinks(
      registered->second.installed, endpoints_live);
}

int32_t sb_try_dispose_owned_correspondences(
    sb_session_t session, sb_support_correspondence_registration_t registration,
    uint32_t version) {
  if (version != 1 || registration <= 0) return -1;
  if (g_inside_relation_callback) return -4;
  std::unique_lock<std::mutex> lock(g_sessions_mutex, std::try_to_lock);
  if (!lock.owns_lock()) return -4;
  auto registered = g_support_registrations.find(registration);
  if (registered == g_support_registrations.end())
    return 1;  // Already retired.
  if (registered->second.session != session || !registered->second.strict_owned)
    return 0;
  auto owner = g_sessions.find(session);
  if (owner != g_sessions.end() && owner->second->context) {
    auto& linker = owner->second->context->GetHub()->GetLinker();
    if (linker.TryRemoveOwnedLinks(registered->second.installed) == -4)
      return -4;
  }
  g_support_registrations.erase(registered);
  return 1;
}
