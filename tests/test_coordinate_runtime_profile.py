import torch
from scripts.profile_coordinate_runtime import backward_kind


def test_backward_phase_classification_matches_existing_panel_boundaries():
    assert backward_kind(torch.empty(2,3,50304))=='student_backward'
    assert backward_kind(torch.empty(128,10))=='student_backward'
    assert backward_kind([torch.empty(()),torch.empty(4,4)])=='norm_edit_backward'
    assert backward_kind([torch.empty(768,8),torch.empty(768)])=='actor_backward'
