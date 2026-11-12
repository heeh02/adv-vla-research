"""Lossless observation witness for the actual flat native dict/OrderedDict.

Never changes, replaces, or restores the parent observation. The existing
strict identity validator still rejects unsupported leaves and other objects.
"""
from collections import OrderedDict
import copy

from openpi.eval import selective_reference as R


def observation_snapshot(observation):
    if type(observation) not in (dict, OrderedDict):
        raise TypeError('Native observation must be an exact dict or OrderedDict')
    if any(type(key) is not str for key in observation):
        raise TypeError('Native observation keys must be exact strings')
    snapshot = {
        'schema': 'native-observation-witness-v1',
        'container_type': 'collections.OrderedDict' if type(observation) is OrderedDict else 'builtins.dict',
        'key_order': tuple(observation),
        'values': {key: copy.deepcopy(value) for key, value in observation.items()},
    }
    R._identity(snapshot)  # Full original leaf dtype/shape/bytes; no coercion.
    return snapshot


def observation_identity(observation):
    return R._identity(observation_snapshot(observation))


def observation_binding(read):
    if not callable(read):
        raise TypeError('Actual parent observation reader is required')

    def snapshot():
        return observation_snapshot(read())

    def verify_unchanged(saved):
        if R._identity(snapshot()) != R._identity(saved):
            raise ValueError('parent_observation_changed_no_restore')

    return R.StateBinding('root.observation', snapshot, verify_unchanged)
