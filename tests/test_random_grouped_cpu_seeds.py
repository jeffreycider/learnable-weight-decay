import torch
from scripts.prepare_random_grouped_cpu_seeds import fresh_actor,assert_zero_heads
from scripts.fit_cpu_edit_teacher import exact_state_equal


def test_random_actor_is_reproducible_zero_head_and_preserves_caller_rng():
    torch.set_num_threads(1);torch.manual_seed(19);before=torch.get_rng_state().clone()
    a,rng=fresh_actor(940067);b,other=fresh_actor(940067);c,_=fresh_actor(940068)
    assert torch.equal(torch.get_rng_state(),before)
    assert torch.equal(rng,other) and exact_state_equal(a.state_dict(),b.state_dict())
    assert_zero_heads(a);assert_zero_heads(c)
    assert not torch.equal(a.matrix_input.weight,c.matrix_input.weight)
