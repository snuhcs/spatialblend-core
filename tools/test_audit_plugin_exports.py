#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Unit tests for the plugin export audit. No native build required."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_plugin_exports as audit  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


class AuditV2PluginExportsTest(unittest.TestCase):
    def test_headers_list_public_os_symbols(self) -> None:
        names = audit.expected_symbols(ROOT / "spatialblend/api/c")
        for required in (
            "sb_create_session",
            "sb_host_space",
            "sb_connect_space",
            "sb_share_element",
            "sb_register_on_relation_event",
            "sb_create_actor",
            "sb_get_observability_version",
            "sb_get_relation_states_v2",
        ):
            self.assertIn(required, names)
        self.assertTrue(all(name.startswith("sb_") for name in names))
        self.assertNotIn("__declspec", names)
        self.assertGreater(len(names), 100)

    def test_elf_version_and_linker_boundaries(self) -> None:
        exported = audit.parse_nm(
            "00000000003acbe0 T sb_create_session@@VERS_0.0.1\n"
            "00000000008be590 D __start_pb_defaults\n"
            "                 U abort\n"
        )
        missing, foreign = audit.audit_symbols({"sb_create_session"}, exported)
        self.assertEqual(missing, [])
        self.assertEqual(foreign, [])

    def test_parse_nm_skips_local_symbols_and_version_node(self) -> None:
        text = (
            "0000000000000000 A VERS_0.0.1\n"
            "0000000000000110 b _ZN9grpc_core7ExecCtx9exec_ctx_E\n"
            "0000000000000020 d _ZN6google8protobuf8internal15ThreadSafeArena13thread_cache_E\n"
            "0000000000323e04 T sb_create_space@@VERS_0.0.1\n"
            "0000000000001000 W weak_global\n"
            "0000000000001000 w weak_lower\n"
            "0000000000002000 B real_global_bss\n"
            "0000000000002000 A some_abs\n"
        )
        self.assertEqual(
            audit.parse_nm(text),
            {"sb_create_space@@VERS_0.0.1", "weak_global", "weak_lower", "real_global_bss", "some_abs"},
        )

    def test_parse_nm_skips_llvm_nm_version_node(self) -> None:
        # llvm-nm prints the version-definition node with its own version appended.
        text = (
            "0000000000000000 A VERS_0.0.1@@VERS_0.0.1\n"
            "0000000000000000 A VERS_0.0.1@VERS_0.0.1\n"
            "0000000000000000 A VERS_0.0.1@@VERS_0.0.2\n"
            "0000000000323e04 T sb_create_space@@VERS_0.0.1\n"
        )
        exported = audit.parse_nm(text)
        self.assertEqual(exported, {"VERS_0.0.1@@VERS_0.0.2", "sb_create_space@@VERS_0.0.1"})
        missing, foreign = audit.audit_symbols({"sb_create_space"}, exported)
        self.assertEqual(missing, [])
        # A node whose halves name different versions is not the version script's node: still reported.
        self.assertEqual(foreign, ["VERS_0.0.1"])

    def test_foreign_symbol_fails(self) -> None:
        missing, foreign = audit.audit_symbols(
            {"sb_create_session"},
            {"sb_create_session@@VERS_0.0.1", "__stop_linkarr_upb_AllExts", "SSL_read"},
        )
        self.assertEqual(missing, [])
        self.assertEqual(foreign, ["SSL_read"])

    def test_macho_underscore_prefix(self) -> None:
        exported = audit.parse_nm("0000000000001234 T _sb_create_session\n")
        missing, foreign = audit.audit_symbols({"sb_create_session"}, exported)
        self.assertEqual(missing, [])
        self.assertEqual(foreign, [])

    def test_missing_expected_symbol(self) -> None:
        missing, foreign = audit.audit_symbols(
            {"sb_create_session", "sb_host_space"},
            {"sb_create_session"},
        )
        self.assertEqual(missing, ["sb_host_space"])
        self.assertEqual(foreign, [])

    def test_dumpbin_export_rows(self) -> None:
        text = (
            "    ordinal hint RVA      name\n"
            "          1    0 00001000 sb_create_session\n"
            "          2    1 00001110 SSL_library_init\n"
        )
        names = audit.parse_dumpbin(text)
        self.assertEqual(names, {"sb_create_session", "SSL_library_init"})

    def test_sha256sums_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "libspatialblend.so"
            library.write_bytes(b"v2-plugin")
            sidecar = Path(tmp) / "SHA256SUMS"
            digest = audit.write_sha256sums(library, sidecar)
            self.assertEqual(len(digest), 64)
            self.assertEqual(
                sidecar.read_text(encoding="utf-8"),
                f"{digest}  libspatialblend.so\n",
            )

    def test_so_audit_uses_the_given_nm(self) -> None:
        seen: list[list[str]] = []

        def fake_run(command: list[str]) -> str:
            seen.append(command)
            return "0000000000000000 T sb_create_session\n"

        original = audit._run
        audit._run = fake_run
        try:
            names = audit.list_exports(
                Path("libspatialblend.so"), None, "/ndk/llvm-nm"
            )
        finally:
            audit._run = original
        self.assertEqual(seen[0][0], "/ndk/llvm-nm")
        self.assertEqual(seen[0][1:3], ["-D", "--defined-only"])
        self.assertEqual(names, {"sb_create_session"})

    def test_checksum_is_written_only_when_the_audit_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            headers = Path(tmp) / "headers"
            headers.mkdir()
            (headers / "api.h").write_text(
                "SB_EXPORT void sb_create_session();\n", encoding="utf-8"
            )
            library = Path(tmp) / "libspatialblend.so"
            library.write_bytes(b"not-a-real-so")
            checksum = Path(tmp) / "SHA256SUMS"
            original = audit.list_exports
            audit.list_exports = lambda *_args, **_kwargs: {"SSL_read"}
            try:
                code = audit.main(
                    [
                        "--library",
                        str(library),
                        "--headers",
                        str(headers),
                        "--checksum",
                        str(checksum),
                    ]
                )
            finally:
                audit.list_exports = original
            self.assertEqual(code, 1)
            self.assertFalse(checksum.exists())

            audit.list_exports = lambda *_args, **_kwargs: {"sb_create_session"}
            try:
                code = audit.main(
                    [
                        "--library",
                        str(library),
                        "--headers",
                        str(headers),
                        "--checksum",
                        str(checksum),
                    ]
                )
            finally:
                audit.list_exports = original
            self.assertEqual(code, 0)
            self.assertTrue(checksum.is_file())


if __name__ == "__main__":
    unittest.main()
