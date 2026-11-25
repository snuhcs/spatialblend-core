/* The symbol-version tag is a visibility mask only (export sb_*); the C ABI
 * generation is sb_abi_version() (3 now), which every binding checks at load.
 * Consumers load the library with dlopen/P-Invoke and never bind to the tag, so
 * it is NOT bumped with the ABI. */
VERS_0.0.1 {
    global:
        sb_*;
    local:
        *;
};
