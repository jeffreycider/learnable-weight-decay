import copy
import unittest

import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call

from baseline.model import GPT
from rewriter.algebra import Anchors, Factors, RawAction
from rewriter.function_preservation import FunctionalPreservation, chunked_teacher_kl, gpt_logits


def make_action(model, magnitude=.001):
    gen = torch.Generator().manual_seed(819)
    matrices, vectors, leaves = {}, {}, []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2:
            n, m = parameter.shape[::-1] if name == 'embed.weight' else parameter.shape
            anchors = Anchors.random(n, m, 2, generator=gen)
            p = (torch.randn(n, 2, generator=gen) * magnitude).requires_grad_()
            q = (torch.randn(m, 2, generator=gen) * magnitude).requires_grad_()
            matrices[name] = Factors.canonical(p, q, anchors)
            leaves.extend([p, q])
        else:
            vector = (torch.randn(parameter.shape, generator=gen) * magnitude).requires_grad_()
            vectors[name] = vector
            leaves.append(vector)
    return RawAction(matrices, vectors), leaves


class IndependentLogits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gpt = model

    def forward(self, inputs):
        # Separate reference orchestration used only by unit tests.
        hidden = self.gpt.norm1(self.gpt.embed(inputs))
        for layer in self.gpt.blocks:
            hidden = layer(hidden)
        raw = self.gpt.proj(self.gpt.norm2(hidden)).float()
        return 15 * raw * torch.rsqrt(raw * raw + 225)


class FunctionPreservationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.model = GPT(vocab_size=16, num_layers=1, model_dim=128).float()
        self.inputs = torch.randint(16, (3, 4))
        self.targets = torch.randint(16, (3, 4))

    def test_logits_match_baseline_ce_and_bfloat16_casts(self):
        for bf16 in (False, True):
            with self.subTest(bfloat16_embedding=bf16):
                model = copy.deepcopy(self.model)
                if bf16:
                    model.embed.bfloat16()
                expected = model(self.inputs, self.targets)
                logits = gpt_logits(model, self.inputs)
                actual = F.cross_entropy(logits.flatten(0, 1), self.targets.flatten(), reduction='sum')
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_ce_diagnostic_uses_known_next_tokens_and_one_student_forward(self):
        engine=FunctionalPreservation(self.model,microbatch_sequences=3,kl_dtype=torch.float64)
        teacher,=engine.cache_teacher(self.inputs,data_role='past_training')
        loss,metrics=engine.student_parameter_loss(engine.parameters,teacher,return_ce=True)
        expected=-teacher.log_probs[:,:-1].gather(2,self.inputs[:,1:,None]).mean()
        self.assertEqual(float(loss),0.)
        self.assertAlmostEqual(metrics['teacher_ce'],float(expected),places=12)
        self.assertEqual(metrics['ce_change'],0.)
        self.assertEqual(metrics['ce_tokens'],9)
        self.assertEqual(engine.compute.student_forwards,1)

    def test_fresh_next_teacher_is_audit_only(self):
        engine=FunctionalPreservation(self.model,microbatch_sequences=3,kl_dtype=torch.float64)
        teacher,=engine.cache_teacher(self.inputs,data_role='fresh_next_training_audit')
        with self.assertRaisesRegex(ValueError,'audit-only'):
            engine.student_parameter_loss(engine.parameters,teacher)
        with torch.no_grad():
            loss=engine.student_parameter_loss(engine.parameters,teacher)
        self.assertEqual(float(loss),0.)

    def test_larger_microbatch_matches_streamed_loss_and_gradients(self):
        inputs = self.inputs.repeat(2, 1)
        action, leaves = make_action(self.model, magnitude=.02)
        losses, gradients = [], []
        for batch_size in (1, 4):
            engine = FunctionalPreservation(self.model, microbatch_sequences=batch_size,
                                            kl_dtype=torch.float64)
            teachers = engine.cache_teacher(inputs, data_role='past_training')
            loss = sum(engine.student_loss(action, teacher) * teacher.tokens / inputs.numel()
                       for teacher in teachers)
            losses.append(loss.detach())
            gradients.append(torch.autograd.grad(loss, leaves, retain_graph=True))
            self.assertEqual(engine.compute.teacher_forwards, (len(inputs)+batch_size-1)//batch_size)
        torch.testing.assert_close(losses[0], losses[1], atol=1e-7, rtol=1e-4)
        for first, second in zip(*gradients):
            torch.testing.assert_close(first, second, atol=2e-6, rtol=.01)

    def test_checkpoint_reload_invalidates_cache_preserves_storage_and_compilation(self):
        engine = FunctionalPreservation(self.model, microbatch_sequences=4)
        compiled_graphs = []
        def backend(graph, inputs):
            compiled_graphs.append(graph)
            return graph.forward
        engine.enable_compiled_forward(backend=backend, fullgraph=True)
        teacher, = engine.cache_teacher(self.inputs, data_role='past_training')
        original_model = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        pointers = {name: value.data_ptr() for name, value in engine.parameters.items()}
        state = {name: value.clone() for name, value in engine.parameters.items()}
        state['proj.bias'].add_(torch.arange(16))
        count = len(compiled_graphs)
        self.assertGreater(count, 0)
        engine.load_teacher_state(state, engine.buffers)
        self.assertEqual(engine.compute.teacher_forwards, 0)
        self.assertEqual(pointers, {name: value.data_ptr() for name, value in engine.parameters.items()})
        with self.assertRaisesRegex(ValueError, 'different frozen checkpoint'):
            engine.student_parameter_loss(engine.parameters, teacher)
        new_teacher, = engine.cache_teacher(self.inputs, data_role='past_training')
        self.assertFalse(torch.equal(new_teacher.log_probs, teacher.log_probs))
        self.assertEqual(len(compiled_graphs), count)
        expected_norm = sum(value.double().square().sum() for value in state.values())
        torch.testing.assert_close(engine.base_weight_norm_sq, expected_norm)
        for name, value in self.model.named_parameters():
            torch.testing.assert_close(value, original_model[name], atol=0, rtol=0)
        identifier = engine.teacher_id
        bad_state = dict(state)
        bad_state['proj.bias'] = bad_state['proj.bias'][:1]
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            engine.load_teacher_state(bad_state, engine.buffers)
        self.assertEqual(engine.teacher_id, identifier)
        torch.testing.assert_close(engine.parameters['proj.bias'], state['proj.bias'])

    def test_chunked_kl_value_and_gradient_match_full_autograd(self):
        logits = torch.randn(2, 3, 7, requires_grad=True)
        teacher = torch.randn(2, 3, 7).log_softmax(-1)
        actual = chunked_teacher_kl(logits, teacher, chunk_tokens=2)
        expected = (teacher.exp() * (teacher - logits.log_softmax(-1))).sum() / 6
        ga, = torch.autograd.grad(actual, logits, retain_graph=True)
        ge, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(ga, ge, atol=2e-8, rtol=2e-6)
        direction = torch.randn_like(logits)
        direction /= direction.norm()
        epsilon = .001
        finite = (chunked_teacher_kl(logits.detach() + epsilon * direction, teacher, 2)
                  - chunked_teacher_kl(logits.detach() - epsilon * direction, teacher, 2)) / (2 * epsilon)
        torch.testing.assert_close(finite, (ga * direction).sum(), atol=8e-5, rtol=.01)

    def test_fp64_kl_resolves_small_edits_at_full_vocabulary(self):
        generator = torch.Generator().manual_seed(71)
        raw = torch.randn(4, 50304, generator=generator) * 4
        teacher_logits = 15 * raw * (raw.square() + 225).rsqrt()
        logq = teacher_logits.double().log_softmax(-1)
        direction = torch.randn(teacher_logits.shape, generator=generator)
        identity = chunked_teacher_kl(teacher_logits, logq, chunk_tokens=2)
        self.assertEqual(identity.dtype, torch.float64)
        self.assertEqual(float(identity), 0.)
        for magnitude in (1e-6, 1e-4, 1e-2):
            with self.subTest(logit_perturbation=magnitude):
                logits = (teacher_logits + magnitude * direction).requires_grad_()
                actual = chunked_teacher_kl(logits, logq, chunk_tokens=2)
                expected = (logq.exp() * (logq - logits.double().log_softmax(-1))).sum() / 4
                torch.testing.assert_close(actual, expected, atol=5e-18, rtol=1e-8)
                self.assertGreater(float(actual.detach()), 0)
                actual_gradient, = torch.autograd.grad(actual, logits, retain_graph=True)
                expected_gradient, = torch.autograd.grad(expected, logits)
                self.assertEqual(actual_gradient.dtype, torch.float32)
                torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-12, rtol=3e-5)

    def test_fp64_teacher_cache_keeps_identity_exact_and_counts_storage(self):
        engine = FunctionalPreservation(self.model, chunk_tokens=2, kl_dtype=torch.float64)
        teacher, = engine.cache_teacher(self.inputs[:1], data_role='past_training')
        self.assertEqual(teacher.log_probs.dtype, torch.float64)
        self.assertEqual(teacher.storage_bytes, teacher.inputs.numel()*teacher.inputs.element_size()
                         + teacher.log_probs.numel()*8)
        zero_action, _ = make_action(self.model, magnitude=0)
        loss = engine.student_loss(zero_action, teacher)
        self.assertEqual(loss.dtype, torch.float64)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertEqual(engine.compute.student_backwards, 1)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_finite_anchored_student_gradient_matches_explicit_dense_rewrite(self):
        engine = FunctionalPreservation(self.model, chunk_tokens=2)
        teacher, = engine.cache_teacher(self.inputs[:1], data_role='past_training', sequence_ids=[71])
        action, leaves = make_action(self.model, magnitude=.02)
        scale = torch.tensor(.7, requires_grad=True)
        actual = engine.student_loss(action, teacher, scale=scale)
        replacements = {}
        for name, parameter in engine.parameters.items():
            if parameter.ndim == 2:
                factor = action.matrices[name]
                edit = factor.P @ factor.anchors.B.T + factor.anchors.A @ factor.Q.T
                if name == 'embed.weight':
                    edit = edit.T
            else:
                edit = action.vectors[name]
            replacements['gpt.' + name] = (parameter.detach().float() + scale * edit).to(parameter.dtype)
        reference = IndependentLogits(copy.deepcopy(self.model))
        reference_logits = functional_call(reference, replacements, (self.inputs[:1],), strict=False)
        qlog = teacher.log_probs
        expected = (qlog.exp() * (qlog - reference_logits.log_softmax(-1))).sum() / teacher.tokens
        gradients_actual = torch.autograd.grad(actual, leaves + [scale], retain_graph=True)
        gradients_expected = torch.autograd.grad(expected, leaves + [scale])
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
        for a, b in zip(gradients_actual, gradients_expected):
            torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-4)
        self.assertGreater(float(actual.detach()), 0)
        self.assertGreater(float(gradients_actual[-1].abs()), 0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in engine.parameters.values()))

    def test_bfloat16_functional_edit_matches_real_stored_weight_commit(self):
        self.model.embed.bfloat16()
        engine = FunctionalPreservation(self.model)
        action, _ = make_action(self.model, magnitude=.01)
        committed = copy.deepcopy(self.model)
        with torch.no_grad():
            for name, parameter in committed.named_parameters():
                if parameter.ndim == 2:
                    factor = action.matrices[name]
                    change = factor.P @ factor.anchors.B.T + factor.anchors.A @ factor.Q.T
                    if name == 'embed.weight':
                        change = change.T
                else:
                    change = action.vectors[name]
                parameter.copy_((parameter.float() + .7 * change).to(parameter.dtype))
        expected = gpt_logits(committed, self.inputs[:1])
        actual = engine._forward(engine.edited_parameters(action, scale=.7), self.inputs[:1])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_realized_norm_values_and_gradients_match_full_fp64_autograd(self):
        engine = FunctionalPreservation(self.model)
        action, leaves = make_action(self.model, magnitude=.02)
        replacements = engine.edited_parameters(action, scale=.7)
        actual_change, actual_edit = engine.realized_parameter_norm_change(replacements, chunk_rows=7)
        expected_change, expected_edit = torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)
        for name, base in engine.parameters.items():
            old, new = base.double(), replacements[name].double()
            delta = new - old
            expected_change = expected_change + 2 * (old * delta).sum() + delta.square().sum()
            expected_edit = expected_edit + delta.square().sum()
        torch.testing.assert_close(actual_change, expected_change, atol=1e-10, rtol=1e-12)
        torch.testing.assert_close(actual_edit, expected_edit, atol=1e-10, rtol=1e-12)
        actual_gradient = torch.autograd.grad(.7 * actual_change + .3 * actual_edit, leaves, retain_graph=True)
        expected_gradient = torch.autograd.grad(.7 * expected_change + .3 * expected_edit, leaves)
        for a, b in zip(actual_gradient, expected_gradient):
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
        expected_base = sum(p.double().square().sum() for p in engine.parameters.values())
        torch.testing.assert_close(engine.base_weight_norm_sq, expected_base, atol=1e-10, rtol=1e-12)
        self.assertEqual(engine.compute.teacher_forwards, 0)
        self.assertEqual(engine.compute.student_forwards, 0)

    def test_erased_bfloat16_edit_has_zero_realized_norm_but_cast_ste_gradient(self):
        self.model.embed.bfloat16()
        with torch.no_grad():
            self.model.embed.weight.fill_(1)
        engine = FunctionalPreservation(self.model)
        base = engine.parameters['embed.weight']
        raw_edit = torch.full(base.shape, 1e-5, requires_grad=True)
        replacements = dict(engine.parameters)
        replacements['embed.weight'] = (base.float() + raw_edit).to(base.dtype)
        self.assertTrue(torch.equal(replacements['embed.weight'], base))
        actual_change, actual_edit = engine.realized_parameter_norm_change(replacements)
        self.assertEqual(float(actual_change.detach()), 0)
        self.assertEqual(float(actual_edit.detach()), 0)
        raw_change = 2 * (base.float() * raw_edit).sum() + raw_edit.square().sum()
        self.assertGreater(float(raw_change.detach()), 0)
        # This is the standard differentiable-cast convention, not a derivative
        # of the literal discontinuous rounding function.
        gradient, = torch.autograd.grad(actual_change + actual_edit, raw_edit)
        self.assertTrue(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient, torch.full_like(gradient, 2), atol=0, rtol=0)

    def test_prebuilt_parameter_loss_reuses_exact_mapping_and_checks_dtype(self):
        engine = FunctionalPreservation(self.model)
        teacher, = engine.cache_teacher(self.inputs[:1], data_role='past_training')
        action, _ = make_action(self.model)
        parameters = engine.edited_parameters(action)
        actual = engine.student_parameter_loss(parameters, teacher)
        expected = engine.student_loss(action, teacher)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(engine.compute.student_forwards, 2)
        broken = dict(parameters)
        broken['embed.weight'] = broken['embed.weight'].double()
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            engine.student_parameter_loss(broken, teacher)
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            engine.realized_parameter_norm_change(broken)
        self.assertEqual(engine.compute.student_forwards, 2)

    def test_teacher_cache_is_frozen_and_compute_is_explicit(self):
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        rng = torch.random.get_rng_state().clone()
        engine = FunctionalPreservation(self.model, microbatch_sequences=1, chunk_tokens=2)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        teachers = engine.cache_teacher(self.inputs, data_role='past_training', sequence_ids=[5, 6, 7])
        self.assertEqual(len(teachers), 3)
        self.assertEqual(engine.compute.teacher_forwards, 3)
        self.assertEqual(engine.compute.teacher_tokens, 12)
        self.assertEqual(teachers[1].sequence_ids, (6,))
        self.assertEqual(teachers[0].log_probs.device.type, 'cpu')
        action, leaves = make_action(self.model, magnitude=0)
        loss = engine.student_loss(action, teachers[0])
        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=0, rtol=0)
        loss.backward()
        for name, p in self.model.named_parameters():
            torch.testing.assert_close(p, before[name], atol=0, rtol=0)
            self.assertIsNone(p.grad)
        self.assertEqual(engine.compute.student_forwards, 1)
        self.assertEqual(engine.compute.student_backwards, 1)
        self.assertEqual(engine.compute.student_backward_tokens, 4)
        # Mutating the original model/input after caching cannot change teacher
        # parameters or inputs. Functional student evaluation uses the snapshot.
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(3)
        inputs_before = teachers[0].inputs.clone()
        self.inputs.fill_(0)
        torch.testing.assert_close(teachers[0].inputs, inputs_before, atol=0, rtol=0)
        loss2 = engine.student_loss(make_action(self.model, magnitude=0)[0], teachers[0])
        torch.testing.assert_close(loss2, torch.zeros_like(loss2), atol=0, rtol=0)
        self.assertEqual(engine.compute.teacher_forwards, 3)
        self.assertEqual(engine.compute.student_forwards, 2)

    def test_rejects_validation_and_cross_checkpoint_teacher_cache(self):
        engine = FunctionalPreservation(self.model)
        for role in ('validation', 'test', 'heldout'):
            with self.assertRaisesRegex(ValueError, 'no validation/test'):
                engine.cache_teacher(self.inputs, data_role=role)
        teacher, = engine.cache_teacher(self.inputs[:1], data_role='past_training')
        other = FunctionalPreservation(self.model)
        with self.assertRaisesRegex(ValueError, 'different frozen checkpoint'):
            other.student_loss(make_action(self.model)[0], teacher)
        self.assertEqual(other.compute.student_forwards, 0)


if __name__ == '__main__':
    unittest.main()
