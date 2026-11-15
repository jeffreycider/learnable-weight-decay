"""Bind shared Chisel and fitted private maps to one live training run.

This module plugs the existing *anchored* coordinate actor into the ordinary
TransformerOnlineLifecycle. Both parameter sets are registered, so its version
guards, delayed VJP, optimizer ownership and pending-action resume cover both.
It does not launch training, fit replay examples, or import optimizer moments.
The caller explicitly constructs the online optimizer and pins that policy.
"""
import copy

from torch import nn

from .coordinate_chisel import CoordinateChisel, CheckpointProbes


class CoordinateOnlineBinding(nn.Module):
    schema = 'coordinate_online_binding_v1'

    def __init__(self, shared, private, *, run_id, source_private_key):
        super().__init__()
        if type(shared) is not CoordinateChisel or type(private) is not CheckpointProbes:
            raise TypeError('binding currently supports the declared anchored coordinate actor only')
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('nonempty live run identity required')
        if private.state_key != ('online', run_id):
            raise ValueError('private maps must belong to this live run')
        if shared.frame_mode != private.frame_mode:
            raise ValueError('shared and private frame modes differ')
        self.shared = shared
        self.private = private
        self.run_id = run_id
        self.source_private_key = tuple(source_private_key)

    @classmethod
    def from_fitted(cls, shared, private, snapshot, *, run_id):
        """Copy fitted weights into an explicit new live identity, without aliases.

        Private maps persist across subsequent same-layout GPT steps, including
        changes to the fixed measurement anchors. Their offline state identity
        is retained as provenance; it is not silently reused as a live identity.
        Shared and private optimizer states are intentionally not imported here.
        """
        if type(shared) is not CoordinateChisel or type(private) is not CheckpointProbes:
            raise TypeError('unsupported action family')
        private.validate_layout(snapshot)
        owner = copy.deepcopy(shared)
        live_private = owner.make_private(snapshot, state_key=('online', run_id), seed=0)
        expected_source = live_private.get_extra_state()
        expected_source['state_key'] = private.state_key
        if private.get_extra_state() != expected_source:
            raise ValueError('source private architecture differs from shared actor')
        source_state = copy.deepcopy(private.state_dict())
        source_state['_extra_state'] = live_private.get_extra_state()
        live_private.load_state_dict(source_state)
        return cls(owner, live_private, run_id=run_id, source_private_key=private.state_key)

    @property
    def output_heads(self):
        return self.shared.output_heads

    def forward(self, snapshot):
        return self.shared(snapshot, self.private)

    def get_extra_state(self):
        return dict(schema=self.schema, run_id=self.run_id,
                    source_private_key=self.source_private_key,
                    architecture=self.shared.configuration(),
                    private_metadata=self.private.get_extra_state(),
                    optimizer_import='none; online optimizer is caller-owned')

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('coordinate online identity or architecture differs')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Validate identity before nn.Module can copy any parameter tensors.
        self.set_extra_state(state_dict.get('_extra_state'))
        if state_dict.get('private._extra_state') != self.private.get_extra_state():
            raise ValueError('private online identity differs')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
