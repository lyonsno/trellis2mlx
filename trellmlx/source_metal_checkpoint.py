"""Durable, complete pointwise BVH-query replay for the optional Metal finisher.

The primitive is unchanged. Query partitioning preserves every output; saved
prefixes can be reused only for byte-identical points and bound producer inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import weakref

import numpy as np


class FinisherCheckpointBoundary(BaseException):
    """Unwind native resources before asking the scheduler to publish a successor."""


def _hash_array(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class DistanceQueryCheckpoint:
    def __init__(self, root, identity, *, chunk_rows=32768, resume=None, on_boundary=None):
        if type(chunk_rows) is not int or chunk_rows <= 0:
            raise ValueError('query chunk size must be a positive integer')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'state.json'
        self.identity = json.loads(json.dumps(identity, sort_keys=True))
        self.chunk_rows = chunk_rows
        self.on_boundary = on_boundary or (lambda store: None)
        self.call_index = 0
        self.completed_rows = 0
        self.phase = None
        self.allowed_phases = None
        self.resources = []
        if resume is None:
            if any(self.root.iterdir()):
                raise FileExistsError('existing checkpoint state requires explicit resume descriptor')
            self.state = dict(schema='trellis2mlx.metal-query-state.v1', identity=self.identity,
                              original=None, calls=[])
        else:
            if resume.get('root') != str(self.root):
                raise ValueError('checkpoint root identity differs')
            self.state = json.loads(json.dumps(resume['state']))
            if (self.state.get('schema') != 'trellis2mlx.metal-query-state.v1'
                    or self.state.get('identity') != self.identity):
                raise ValueError('checkpoint producer identity differs')
            self._verify_saved(resume)
        _atomic_json(self.path, self.state)

    def _file(self, name):
        path = self.root / name
        if Path(name).name != name or path.is_symlink():
            raise ValueError('checkpoint file escapes its root')
        return path

    def _verify_saved(self, descriptor):
        original = self.state['original']
        if original and _hash_file(self._file(original['file'])) != original['sha256']:
            raise ValueError('original surface checkpoint bytes changed')
        expected = descriptor['buffers']
        for index, call in enumerate(self.state['calls']):
            if (type(call.get('rows')) is not int or call['rows'] < 0
                    or type(call.get('done')) is not int or not 0 <= call['done'] <= call['rows']
                    or type(call.get('return_uvw')) is not bool):
                raise ValueError('invalid query checkpoint progress')
            for name, dtype, width in self._columns(call['return_uvw']):
                file = f'query-{index:03d}-{name}.npy'
                array = np.load(self._file(file), mmap_mode='r', allow_pickle=False)
                shape = (call['rows'],) if width == 1 else (call['rows'], width)
                if array.dtype != dtype or array.shape != shape:
                    raise ValueError('query checkpoint buffer shape/dtype changed')
                if _hash_array(array[:call['done']]) != expected[file]:
                    raise ValueError('query checkpoint bytes changed')

    @staticmethod
    def _columns(return_uvw):
        columns = [('distances', np.dtype('float32'), 1), ('face_ids', np.dtype('int64'), 1)]
        if return_uvw:
            columns.append(('uvw', np.dtype('float32'), 3))
        return columns

    def original(self):
        import torch
        record = self.state['original']
        if record is None:
            return None
        with np.load(self._file(record['file']), allow_pickle=False) as z:
            return torch.from_numpy(z['vertices']), torch.from_numpy(z['faces'])

    def has_original(self):
        return self.state['original'] is not None

    def freeze_original(self, vertices, faces):
        if self.state['original'] is not None:
            return
        path = self._file('original.npz')
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as f:
            np.savez(f, vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        self.state['original'] = {'file': path.name, 'sha256': _hash_file(path)}
        _atomic_json(self.path, self.state)

    def descriptor(self):
        buffers = {}
        for index, call in enumerate(self.state['calls']):
            for name, _, _ in self._columns(call['return_uvw']):
                file = f'query-{index:03d}-{name}.npy'
                array = np.load(self._file(file), mmap_mode='r', allow_pickle=False)
                buffers[file] = _hash_array(array[:call['done']])
        _atomic_json(self.path, self.state)
        return {'root': str(self.root), 'state': json.loads(json.dumps(self.state)), 'buffers': buffers}

    def request_yield(self):
        _atomic_json(self.path, self.state)
        raise FinisherCheckpointBoundary()

    def set_phase(self, value):
        self.phase = value

    def reset_replay(self):
        self.call_index = 0

    def track_resource(self, resource):
        self.resources.append(weakref.ref(resource))

    def resources_released(self):
        return all(reference() is None for reference in self.resources)

    def wrap_bvh(self, primitive):
        self.track_resource(primitive)
        store = self

        class BVH:
            def __getattr__(self, name):
                return getattr(primitive, name)

            def unsigned_distance(self, points, return_uvw=False):
                return store._query(primitive, points, return_uvw)

        return BVH()

    def _query(self, primitive, points, return_uvw):
        import torch
        if self.allowed_phases is not None and self.phase not in self.allowed_phases:
            return primitive.unsigned_distance(points, return_uvw=return_uvw)
        values = points.detach().to(device='cpu', dtype=torch.float32).contiguous().numpy()
        index = self.call_index
        self.call_index += 1
        signature = {'points_sha256': _hash_array(values), 'rows': len(values), 'return_uvw': bool(return_uvw)}
        if index < len(self.state['calls']):
            call = self.state['calls'][index]
            if any(call.get(k) != v for k, v in signature.items()):
                raise ValueError(f'checkpoint query identity differs at call {index}')
        else:
            call = {**signature, 'done': 0}
            self.state['calls'].append(call)
            for name, dtype, width in self._columns(return_uvw):
                shape = (len(values),) if width == 1 else (len(values), width)
                np.lib.format.open_memmap(self._file(f'query-{index:03d}-{name}.npy'),
                                         mode='w+', dtype=dtype, shape=shape).flush()
        arrays = [np.load(self._file(f'query-{index:03d}-{name}.npy'), mmap_mode='r+', allow_pickle=False)
                  for name, _, _ in self._columns(return_uvw)]
        for start in range(call['done'], len(values), self.chunk_rows):
            end = min(start + self.chunk_rows, len(values))
            started = time.perf_counter()
            out = primitive.unsigned_distance(torch.from_numpy(values[start:end]), return_uvw=return_uvw)
            if not isinstance(out, (tuple, list)) or len(out) != 3:
                raise ValueError('incomplete distance primitive output')
            for target, output in zip(arrays, out):
                data = output.detach().cpu().numpy()
                if data.shape != target[start:end].shape or data.dtype != target.dtype or not np.isfinite(data).all():
                    raise ValueError('partial/nonfinite or retyped distance result')
                target[start:end] = data
                target.flush()
            call['done'] = end
            self.completed_rows = end
            elapsed = time.perf_counter() - started
            self.state.setdefault('chunks', []).append(dict(call=index, start=start, end=end,
                                                           seconds=elapsed, phase=self.phase))
            print(json.dumps(dict(distance_chunk=index, start=start, end=end, seconds=elapsed,
                                  phase=self.phase)), flush=True)
            _atomic_json(self.path, self.state)
            self.on_boundary(self)
        # The remesher subtracts its band in place. Never expose saved buffers.
        result = [torch.from_numpy(np.array(array, copy=True)) for array in arrays]
        return result[0], result[1], result[2] if return_uvw else None
