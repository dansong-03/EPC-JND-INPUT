#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VP-JNDnet persistent inference service.

Startup:
    python vpjnd_server.py

Input protocol on stdin for each request:
    1) A text header line terminated by '\n':
           W H frame_idx qp mode
    2) Three raw uint8 planes, read strictly in this order:
           OrgY: W * H bytes
           RecY: W * H bytes
           JndY: W * H bytes

Output protocol on stdout for each request:
    p=<float> y=<0_or_1>\n

Notes:
    - The TensorFlow graph is built once at startup.
    - The checkpoint is restored once at startup.
    - The process keeps serving requests until stdin is closed.
"""

from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

import input
import network

tf.compat.v1.disable_eager_execution()
tf1 = tf.compat.v1


GRID_SIZE = input.GRID_SIZE
PATCH_SIZE = input.PATCH_SIZE
NUM_PATCHES = input.NUM_PATCHES_PER_IMAGE
CHECKPOINT_FILE = os.path.join('.', 'checkpoint', 'PWJND-model', 'logs', 'best_model.ckpt')


def read_exact(num_bytes):
    """Read exactly num_bytes from stdin.buffer or return None on EOF."""
    chunks = []
    remaining = num_bytes
    stream = sys.stdin.buffer

    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            if not chunks:
                return None
            return None
        chunks.append(chunk)
        remaining -= len(chunk)

    return b''.join(chunks)


def read_header_line():
    """Read one header line from stdin.buffer or return None on EOF."""
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    return line


def parse_header(header_bytes):
    try:
        header = header_bytes.decode('utf-8').strip()
    except UnicodeDecodeError as exc:
        print('Failed to decode header: {}'.format(exc), file=sys.stderr)
        sys.exit(1)

    parts = header.split()
    if len(parts) != 5:
        print('Invalid header field count: {!r}'.format(header), file=sys.stderr)
        sys.exit(1)

    try:
        width = int(parts[0])
        height = int(parts[1])
        frame_idx = int(parts[2])
        qp = int(parts[3])
        mode = parts[4]
    except ValueError as exc:
        print('Failed to parse header values: {!r}, error={}'.format(header, exc), file=sys.stderr)
        sys.exit(1)

    if width <= 0 or height <= 0:
        print('Invalid frame size: W={}, H={}'.format(width, height), file=sys.stderr)
        sys.exit(1)

    return width, height, frame_idx, qp, mode


def y_to_rgb_normalized(y_plane, width, height):
    y = np.frombuffer(y_plane, dtype=np.uint8)
    if y.size != width * height:
        print('Invalid plane size: expected {}, got {}'.format(width * height, y.size), file=sys.stderr)
        sys.exit(1)
    y = y.reshape((height, width))
    rgb = np.stack([y, y, y], axis=-1)
    return rgb.astype(np.float32) / 255.0 - 0.5


def compute_grid_starts(length, patch_size, grid_size):
    if length <= patch_size:
        return np.zeros((grid_size,), dtype=np.int32)

    centers = np.linspace(patch_size / 2.0, length - patch_size / 2.0, grid_size)
    starts = np.round(centers - patch_size / 2.0).astype(np.int32)
    starts = np.clip(starts, 0, length - patch_size)
    return starts


def extract_fixed_grid_patches(image, patch_size=PATCH_SIZE, grid_size=GRID_SIZE):
    """Extract deterministic clamped patches with shape (64, 32, 32, 3)."""
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError('Expected 3 channels, got {}'.format(channels))

    if height < patch_size or width < patch_size:
        pad_h = max(0, patch_size - height)
        pad_w = max(0, patch_size - width)
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        height, width, _ = image.shape

    y_starts = compute_grid_starts(height, patch_size, grid_size)
    x_starts = compute_grid_starts(width, patch_size, grid_size)

    patches = np.empty((grid_size * grid_size, patch_size, patch_size, 3), dtype=np.float32)
    patch_index = 0
    for y0 in y_starts:
        y0 = int(np.clip(y0, 0, height - patch_size))
        y1 = y0 + patch_size
        for x0 in x_starts:
            x0 = int(np.clip(x0, 0, width - patch_size))
            x1 = x0 + patch_size
            patches[patch_index] = image[y0:y1, x0:x1, :]
            patch_index += 1

    return patches


def preprocess_request(org_y, rec_y, jnd_y, width, height):
    org_img = y_to_rgb_normalized(org_y, width, height)
    rec_img = y_to_rgb_normalized(rec_y, width, height)
    jnd_img = y_to_rgb_normalized(jnd_y, width, height)

    ref_patches = extract_fixed_grid_patches(org_img)
    rec_patches = extract_fixed_grid_patches(rec_img)
    jnd_patches = extract_fixed_grid_patches(jnd_img)
    return ref_patches, rec_patches, jnd_patches


def ensure_probabilities(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError('Empty model output')

    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        raise ValueError('Invalid model output: contains NaN or Inf')

    if np.any(values < 0.0) or np.any(values > 1.0):
        values = 1.0 / (1.0 + np.exp(-values))

    return values.astype(np.float32)


def build_server():
    graph = tf.Graph()
    with graph.as_default():
        ref_ph = tf1.placeholder(tf.float32, shape=[NUM_PATCHES, PATCH_SIZE, PATCH_SIZE, 3], name='ref_ph')
        rec_ph = tf1.placeholder(tf.float32, shape=[NUM_PATCHES, PATCH_SIZE, PATCH_SIZE, 3], name='rec_ph')
        jnd_ph = tf1.placeholder(tf.float32, shape=[NUM_PATCHES, PATCH_SIZE, PATCH_SIZE, 3], name='jnd_ph')
        keep_prob_ph = tf1.placeholder(tf.float32, shape=(), name='keep_prob_ph')

        _, aux = network.inference_jnd(rec_ph, ref_ph, jnd_ph, keep_prob_ph, return_aux=True)
        patch_scores = tf.reshape(aux['patch_quality_scores'], [NUM_PATCHES], name='patch_scores')

        saver = tf1.train.Saver()

    session = tf1.Session(graph=graph)
    saver.restore(session, CHECKPOINT_FILE)

    return {
        'graph': graph,
        'session': session,
        'ref_ph': ref_ph,
        'rec_ph': rec_ph,
        'jnd_ph': jnd_ph,
        'keep_prob_ph': keep_prob_ph,
        'patch_scores': patch_scores,
    }


def main():
    server = build_server()
    session = server['session']

    try:
        while True:
            header_bytes = read_header_line()
            if header_bytes is None:
                break

            width, height, frame_idx, qp, mode = parse_header(header_bytes)
            del frame_idx, qp, mode

            plane_bytes = width * height
            org_y = read_exact(plane_bytes)
            if org_y is None:
                break
            rec_y = read_exact(plane_bytes)
            if rec_y is None:
                break
            jnd_y = read_exact(plane_bytes)
            if jnd_y is None:
                break

            try:
                ref_patches, rec_patches, jnd_patches = preprocess_request(org_y, rec_y, jnd_y, width, height)
            except Exception as exc:
                print('Preprocess failed: {}'.format(exc), file=sys.stderr)
                sys.exit(1)

            feed_dict = {
                server['ref_ph']: ref_patches,
                server['rec_ph']: rec_patches,
                server['jnd_ph']: jnd_patches,
                server['keep_prob_ph']: 1.0,
            }
            patch_probs = session.run(server['patch_scores'], feed_dict=feed_dict)
            patch_probs = ensure_probabilities(patch_probs)

            frame_prob = float(np.mean(patch_probs))
            frame_label = 1 if frame_prob >= 0.5 else 0

            sys.stdout.write('p={:.6f} y={}\n'.format(frame_prob, frame_label))
            sys.stdout.flush()
    finally:
        session.close()


if __name__ == '__main__':
    main()
