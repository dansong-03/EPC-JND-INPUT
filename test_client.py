#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple client for vpjnd_server.py.

Example:
    python test_client.py --w 1920 --h 1080 --org org.bin --rec rec.bin --jnd jnd.bin --frame 0 --qp 27

Protocol sent to the server:
    header: "W H frame_idx qp Y_ONLY\n"
    payload: OrgY + RecY + JndY
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Send one VP-JNDnet request to vpjnd_server.py')
    parser.add_argument('--w', type=int, required=True, help='Frame width')
    parser.add_argument('--h', type=int, required=True, help='Frame height')
    parser.add_argument('--org', type=str, default=None, help='OrgY raw file path')
    parser.add_argument('--rec', type=str, default=None, help='RecY raw file path')
    parser.add_argument('--jnd', type=str, default=None, help='JndY raw file path')
    parser.add_argument('--frame', type=int, default=0, help='Frame index')
    parser.add_argument('--qp', type=int, default=27, help='QP value')
    return parser.parse_args()


def load_or_random(path, width, height, name):
    expected_size = width * height
    if path is None:
        rng = np.random.default_rng()
        return rng.integers(0, 256, size=(height, width), dtype=np.uint8).tobytes()

    if not os.path.exists(path):
        raise FileNotFoundError('{} file not found: {}'.format(name, path))

    with open(path, 'rb') as f:
        data = f.read()

    if len(data) != expected_size:
        raise ValueError(
            '{} file size mismatch: expected {} bytes, got {} bytes ({})'.format(
                name, expected_size, len(data), path
            )
        )

    return data


def main():
    args = parse_args()
    if args.w <= 0 or args.h <= 0:
        raise ValueError('Width and height must be positive')

    org_bytes = load_or_random(args.org, args.w, args.h, 'OrgY')
    rec_bytes = load_or_random(args.rec, args.w, args.h, 'RecY')
    jnd_bytes = load_or_random(args.jnd, args.w, args.h, 'JndY')

    cmd = [sys.executable, 'vpjnd_server.py']
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )

    try:
        header = '{} {} {} {} Y_ONLY\n'.format(args.w, args.h, args.frame, args.qp).encode('utf-8')

        proc.stdin.write(header)
        proc.stdin.write(org_bytes)
        proc.stdin.write(rec_bytes)
        proc.stdin.write(jnd_bytes)
        proc.stdin.flush()

        result_line = proc.stdout.readline()
        if not result_line:
            stderr_output = proc.stderr.read().decode('utf-8', errors='replace')
            raise RuntimeError('No response from server. stderr:\n{}'.format(stderr_output))

        print(result_line.decode('utf-8', errors='replace').rstrip('\r\n'))
    finally:
        if proc.stdin:
            proc.stdin.close()
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait()


if __name__ == '__main__':
    main()
