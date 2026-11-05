"""CPU control/data tests; actual Metal and queue passage have separate witnesses."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from trellmlx import source_metal_remesh as source
from test_source_metal_remesh import geometry_fixture


def checkpoint_type():
    assert hasattr(source, 'DistanceQueryCheckpoint'), 'finisher has no resumable distance-query adapter'
    return source.DistanceQueryCheckpoint


class Distance:
    def __init__(self):
        self.calls = []

    def unsigned_distance(self, points, return_uvw=False):
        self.calls.append(points.clone())
        return (points[:, 0].clone(), torch.arange(len(points), dtype=torch.int64) * 0,
                points * .25 if return_uvw else None)


def test_actual_finisher_freezes_original_and_wraps_distance_calls():
    v, f, backend, events, filled = geometry_fixture()

    class Checkpoint:
        def original(self):
            return filled, f

        def freeze_original(self, vertices, faces):
            assert torch.equal(vertices, filled)

        def wrap_bvh(self, bvh):
            events.append('checkpoint-bvh')
            return bvh

    source.rebuild_surface(v, f, backend=backend, grid_size=768, target_faces=17,
                           checkpoint=Checkpoint())
    assert events == ['init', 'init', 'bvh', 'checkpoint-bvh', 'remesh', 'init', 'simplify']


def test_partial_query_resumes_all_results_without_repeating_prefix(tmp_path):
    cls = checkpoint_type()
    points = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    first = Distance()

    def stop(store):
        if store.completed_rows == 2:
            store.request_yield()

    store = cls(tmp_path, {'source': 'first'}, chunk_rows=2, on_boundary=stop)
    with pytest.raises(source.FinisherCheckpointBoundary):
        store.wrap_bvh(first).unsigned_distance(points, return_uvw=True)
    assert [len(q) for q in first.calls] == [2]
    descriptor = store.descriptor()
    second = Distance()
    resumed = cls(tmp_path, {'source': 'first'}, chunk_rows=2, resume=descriptor)
    actual = resumed.wrap_bvh(second).unsigned_distance(points, return_uvw=True)
    expected = Distance().unsigned_distance(points, return_uvw=True)
    assert [len(q) for q in second.calls] == [2, 1]
    for left, right in zip(actual, expected):
        assert torch.equal(left, right)


def test_cached_distances_are_not_mutated_by_remesher_inplace_subtraction(tmp_path):
    cls = checkpoint_type()
    points = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    store = cls(tmp_path, {'source': 'first'}, chunk_rows=2)
    output = store.wrap_bvh(Distance()).unsigned_distance(points)
    output[0].sub_(100)
    resumed = cls(tmp_path, {'source': 'first'}, chunk_rows=2, resume=store.descriptor())
    primitive = Distance()
    assert torch.equal(resumed.wrap_bvh(primitive).unsigned_distance(points)[0], points[:, 0])
    assert primitive.calls == []


@pytest.mark.parametrize('change', ['config', 'points', 'data'])
def test_incompatible_or_corrupt_checkpoint_refuses(tmp_path, change):
    cls = checkpoint_type()
    points = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    store = cls(tmp_path, {'source': 'first'}, chunk_rows=2)
    store.wrap_bvh(Distance()).unsigned_distance(points)
    descriptor = store.descriptor()
    if change == 'config':
        with pytest.raises(ValueError, match='identity'):
            cls(tmp_path, {'source': 'second'}, chunk_rows=2, resume=descriptor)
    elif change == 'points':
        resumed = cls(tmp_path, {'source': 'first'}, chunk_rows=2, resume=descriptor)
        with pytest.raises(ValueError, match='query identity'):
            resumed.wrap_bvh(Distance()).unsigned_distance(points + 1)
    else:
        target = tmp_path / 'query-000-distances.npy'
        values = np.load(target, mmap_mode='r+')
        values[0] += 1
        values.flush()
        with pytest.raises(ValueError, match='bytes changed'):
            cls(tmp_path, {'source': 'first'}, chunk_rows=2, resume=descriptor)


def test_existing_state_requires_explicit_resume_descriptor(tmp_path):
    cls = checkpoint_type()
    store = cls(tmp_path, {'source': 'first'}, chunk_rows=2)
    store.wrap_bvh(Distance()).unsigned_distance(torch.ones((3, 3)))
    with pytest.raises(FileExistsError, match='resume'):
        cls(tmp_path, {'source': 'first'}, chunk_rows=2)


def test_missing_primitive_output_cannot_complete_a_chunk(tmp_path):
    cls = checkpoint_type()

    class Partial(Distance):
        def unsigned_distance(self, points, return_uvw=False):
            return (points[:, 0].clone(),)

    store = cls(tmp_path, {'source': 'first'}, chunk_rows=2)
    with pytest.raises(ValueError, match='incomplete'):
        store.wrap_bvh(Partial()).unsigned_distance(torch.ones((3, 3)))
    assert store.state['calls'][0]['done'] == 0


def test_completed_calls_replay_before_resuming_later_partial_query(tmp_path):
    cls = checkpoint_type()
    first = torch.arange(9, dtype=torch.float32).reshape(3,3)
    second = torch.arange(15, dtype=torch.float32).reshape(5,3) + 1
    store = cls(tmp_path, {'source':'first'}, chunk_rows=2)
    store.wrap_bvh(Distance()).unsigned_distance(first)

    def stop(current):
        current.request_yield()

    store.on_boundary = stop
    with pytest.raises(source.FinisherCheckpointBoundary):
        store.wrap_bvh(Distance()).unsigned_distance(second)
    resumed = cls(tmp_path, {'source':'first'}, chunk_rows=2, resume=store.descriptor())
    primitive = Distance()
    wrapper = resumed.wrap_bvh(primitive)
    assert torch.equal(wrapper.unsigned_distance(first)[0], first[:,0])
    assert primitive.calls == []
    assert torch.equal(wrapper.unsigned_distance(second)[0], second[:,0])
    assert [len(v) for v in primitive.calls] == [2,1]
