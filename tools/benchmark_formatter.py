#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""
Benchmark Results Formatter
Converts Google Benchmark JSON output to beautiful table format
"""

import json
import sys
import argparse
from typing import Dict, List, Any
import os
import time

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def format_time(us: float) -> str:
    """Format time in human readable format"""
    if us < 1000:
        return f"{us:.1f}μs"
    elif us < 1000000:
        return f"{us/1000:.1f}ms"
    else:
        return f"{us/1000000:.1f}s"

def format_rate(rate: float) -> str:
    """Format rate in human readable format"""
    if rate > 1000000:
        return f"{rate/1000000:.1f}M/s"
    elif rate > 1000:
        return f"{rate/1000:.1f}k/s"
    else:
        return f"{rate:.1f}/s"

def print_benchmark_table(benchmarks: List[Dict[str, Any]], title: str = "Benchmark Results"):
    """Print benchmarks in a beautiful table format"""
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{title:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    
    if not benchmarks:
        print(f"{Colors.YELLOW}No benchmark results found{Colors.END}")
        return
    
    # Calculate column widths
    max_name_len = max(len(b.get('name', '')) for b in benchmarks)
    max_time_len = 12
    max_rate_len = 12
    max_label_len = max(len(b.get('label', '')) for b in benchmarks) if any(b.get('label') for b in benchmarks) else 0
    
    # Print header
    header = f"{'Benchmark Name':<{max_name_len}} {'Time':<{max_time_len}} {'Rate':<{max_rate_len}}"
    if max_label_len > 0:
        header += f" {'Label':<{max_label_len}}"
    
    print(f"{Colors.BOLD}{Colors.BLUE}{header}{Colors.END}")
    print(f"{Colors.BLUE}{'-' * len(header)}{Colors.END}")
    
    # Print each benchmark
    for bench in benchmarks:
        name = bench.get('name', 'Unknown')
        real_time = bench.get('real_time', 0)
        items_per_second = bench.get('items_per_second', 0)
        label = bench.get('label', '')
        
        # Format the row
        row = f"{name:<{max_name_len}} {format_time(real_time):<{max_time_len}}"
        
        if items_per_second > 0:
            row += f" {format_rate(items_per_second):<{max_rate_len}}"
        else:
            row += f" {'N/A':<{max_rate_len}}"
        
        if max_label_len > 0:
            row += f" {label:<{max_label_len}}"
        
        print(row)
    
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}\n")

def print_summary_stats(benchmarks: List[Dict[str, Any]]):
    """Print summary statistics"""
    if not benchmarks:
        return
    
    times = [b.get('real_time', 0) for b in benchmarks]
    rates = [b.get('items_per_second', 0) for b in benchmarks if b.get('items_per_second', 0) > 0]
    
    print(f"{Colors.BOLD}{Colors.GREEN}📊 Summary Statistics:{Colors.END}")
    print(f"  • Total benchmarks: {len(benchmarks)}")
    print(f"  • Fastest: {format_time(min(times))} ({min(benchmarks, key=lambda x: x.get('real_time', float('inf'))).get('name', 'Unknown')})")
    print(f"  • Slowest: {format_time(max(times))} ({max(benchmarks, key=lambda x: x.get('real_time', 0)).get('name', 'Unknown')})")
    
    if rates:
        print(f"  • Highest rate: {format_rate(max(rates))}")
        print(f"  • Average rate: {format_rate(sum(rates)/len(rates))}")
    
    print()

def main():
    parser = argparse.ArgumentParser(description='Format Google Benchmark JSON results')
    parser.add_argument('input', nargs='?', help='JSON file to format (or read from stdin)')
    parser.add_argument('--title', default='Benchmark Results', help='Title for the table')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.debug:
        print(f"{Colors.YELLOW}🔍 Debug mode enabled{Colors.END}")
        print(f"{Colors.CYAN}Reading input from: {args.input if args.input else 'stdin'}{Colors.END}")
    
    # Read JSON data
    try:
        if args.input and os.path.exists(args.input):
            if args.debug:
                print(f"{Colors.CYAN}Reading from file: {args.input}{Colors.END}")
            with open(args.input, 'r') as f:
                data = json.load(f)
        else:
            if args.debug:
                print(f"{Colors.CYAN}Reading from stdin{Colors.END}")
            # Read from stdin
            data = json.load(sys.stdin)
        
        if args.debug:
            print(f"{Colors.GREEN}✓ JSON loaded successfully{Colors.END}")
            print(f"{Colors.CYAN}Data keys: {list(data.keys())}{Colors.END}")
            
    except Exception as e:
        print(f"{Colors.RED}Error reading JSON: {e}{Colors.END}")
        sys.exit(1)
    
    # Extract benchmarks
    benchmarks = data.get('benchmarks', [])
    
    if args.debug:
        print(f"{Colors.CYAN}Found {len(benchmarks)} benchmarks{Colors.END}")
        if benchmarks:
            print(f"{Colors.CYAN}First benchmark keys: {list(benchmarks[0].keys())}{Colors.END}")
    
    if not benchmarks:
        print(f"{Colors.RED}No benchmark data found{Colors.END}")
        sys.exit(1)
    
    # Print the table
    if args.debug:
        print(f"{Colors.YELLOW}📊 Formatting table...{Colors.END}")
    
    table_start = time.time()
    print_benchmark_table(benchmarks, args.title)
    table_time = time.time() - table_start
    
    if args.debug:
        print(f"{Colors.CYAN}Table formatting took: {table_time:.3f}s{Colors.END}")
    
    # Print summary if requested
    if args.summary:
        if args.debug:
            print(f"{Colors.YELLOW}📊 Calculating summary...{Colors.END}")
        
        summary_start = time.time()
        print_summary_stats(benchmarks)
        summary_time = time.time() - summary_start
        
        if args.debug:
            print(f"{Colors.CYAN}Summary calculation took: {summary_time:.3f}s{Colors.END}")
    
    total_time = time.time() - start_time
    if args.debug:
        print(f"{Colors.GREEN}✓ Total formatter execution time: {total_time:.3f}s{Colors.END}")

if __name__ == '__main__':
    main() 