"""Alignment test for the independent incremental path used to price floor(b)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from anamnesis.extraction.model_loader import HookState


def test_incremental_uses_fresh_cache_next_token_alignment_and_all_surfaces():
    path = Path(__file__).parents[1]/'scripts/extraction_perf_phase0.py'
    spec = importlib.util.spec_from_file_location('phase0', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Loaded:
        def __init__(self):
            self.hook_state = HookState()
            self.prefills = 0
            self.positions = []
            self.model = Model(self)

        def clear_hook_state(self):
            self.hook_state.clear()

        def enable_hooks(self):
            self.hook_state.enabled = True

        def disable_hooks(self):
            self.hook_state.enabled = False

    class Model(torch.nn.Module):
        def __init__(self, owner):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.owner = owner

        def forward(self, ids, **kw):
            cache = kw.get('past_key_values')
            if cache is None:
                self.owner.prefills += 1
                assert not self.owner.hook_state.enabled
                return SimpleNamespace(past_key_values={'length': ids.shape[1]})
            pos = int(kw['position_ids'][0,0])
            assert cache['length'] == pos
            assert kw['cache_position'].tolist() == [pos]
            self.owner.positions.append(pos)
            cache['length'] += 1
            value = float(ids[0,0])
            for name in ('pre_rope_keys','v_proj_values','queries'):
                getattr(self.owner.hook_state,name)[0].append(torch.full((1,2,1,3), value))
            for name in ('gate_activations','attn_outputs'):
                getattr(self.owner.hook_state,name)[0].append(torch.full((1,1,4), value))
            return SimpleNamespace(
                past_key_values=cache,
                hidden_states=(torch.zeros(1,1,4),torch.full((1,1,4),value)),
                attentions=(torch.full((1,2,1,pos+1),1/(pos+1)),),
                logits=torch.full((1,1,16),value))

    loaded = Loaded()
    row = dict(tokens=[1,2,3,4,5,6], start=2, end=6)
    pm = np.zeros((2,6,4),dtype=np.float32)
    a = module.incremental_raw(loaded,row,pm)
    b = module.incremental_raw(loaded,row,pm)
    assert loaded.prefills == 2
    assert loaded.positions == [2,3,4,2,3,4]
    assert a.prompt_length == 2
    np.testing.assert_array_equal(a.chosen_token_ids,[4,5,6])
    assert [x.shape for x in a.attentions] == [(1,2,3),(1,2,4),(1,2,5)]
    for t,value in enumerate([3,4,5]):
        np.testing.assert_array_equal(a.hidden_states[t][1],np.full(4,value))
        np.testing.assert_array_equal(a.logits[t],b.logits[t])
        for name in ('pre_rope_keys','v_proj_values','queries','gate_activations','attn_outputs'):
            assert len(getattr(a,name)[0]) == 3
            assert np.all(getattr(a,name)[0][t] == value)
    assert not loaded.hook_state.pre_rope_keys
