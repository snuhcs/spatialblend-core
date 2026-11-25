#!/bin/bash
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

# Test script to verify SSH forwarding and git config transfer in devcontainer
# This script should be run inside the devcontainer

set -e

echo "=== DevContainer Setup Verification ==="
echo

# Test 1: Check if SSH directory exists and has correct permissions
echo "1. Testing SSH directory setup..."
if [ -d ~/.ssh ]; then
    echo "✓ SSH directory exists"
    ls -la ~/.ssh/
    
    # Check permissions
    if [ "$(stat -c %a ~/.ssh)" = "700" ]; then
        echo "✓ SSH directory has correct permissions (700)"
    else
        echo "✗ SSH directory has incorrect permissions: $(stat -c %a ~/.ssh)"
    fi
else
    echo "✗ SSH directory does not exist"
fi
echo

# Test 2: Check if SSH keys exist
echo "2. Testing SSH keys..."
if ls ~/.ssh/id_* 1> /dev/null 2>&1; then
    echo "✓ SSH keys found:"
    ls -la ~/.ssh/id_*
    
    # Check key permissions
    for key in ~/.ssh/id_*; do
        if [ -f "$key" ] && [ "$(stat -c %a "$key")" = "600" ]; then
            echo "✓ Key $key has correct permissions (600)"
        else
            echo "✗ Key $key has incorrect permissions: $(stat -c %a "$key")"
        fi
    done
else
    echo "✗ No SSH keys found"
fi
echo

# Test 3: Check SSH agent
echo "3. Testing SSH agent..."
if [ -n "$SSH_AUTH_SOCK" ]; then
    echo "✓ SSH_AUTH_SOCK is set: $SSH_AUTH_SOCK"
    
    # Check if ssh-agent is running
    if ssh-add -l >/dev/null 2>&1; then
        echo "✓ SSH agent is running and has keys loaded:"
        ssh-add -l
    else
        echo "✗ SSH agent is not running or has no keys"
    fi
else
    echo "✗ SSH_AUTH_SOCK is not set"
fi
echo

# Test 4: Check git config
echo "4. Testing git config..."
if [ -f ~/.gitconfig ]; then
    echo "✓ Git config file exists"
    echo "Git config contents:"
    cat ~/.gitconfig
else
    echo "✗ Git config file does not exist"
fi
echo

# Test 5: Test SSH connection (if GitHub is accessible)
echo "5. Testing SSH connection to GitHub..."
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "✓ SSH connection to GitHub successful"
elif ssh -T git@github.com 2>&1 | grep -q "Permission denied"; then
    echo "⚠ SSH connection to GitHub failed (Permission denied) - this might be expected if no GitHub key is configured"
else
    echo "✗ SSH connection to GitHub failed"
fi
echo

# Test 6: Check environment variables
echo "6. Testing environment variables..."
echo "HOME: $HOME"
echo "USER: $USER"
echo "SSH_AUTH_SOCK: $SSH_AUTH_SOCK"
echo

# Test 7: Check if we can clone a repository via SSH
echo "7. Testing git clone via SSH..."
if git clone git@github.com:octocat/Hello-World.git /tmp/test-clone 2>/dev/null; then
    echo "✓ Successfully cloned repository via SSH"
    rm -rf /tmp/test-clone
elif git clone https://github.com/octocat/Hello-World.git /tmp/test-clone 2>/dev/null; then
    echo "⚠ Successfully cloned repository via HTTPS (SSH not tested)"
    rm -rf /tmp/test-clone
else
    echo "✗ Failed to clone test repository"
fi
echo

echo "=== Verification Complete ==="
echo
echo "Summary:"
echo "- SSH directory and keys should be properly mounted and configured"
echo "- Git config should be transferred from host"
echo "- SSH agent should be running with forwarded socket"
echo "- Environment variables should be properly set"
