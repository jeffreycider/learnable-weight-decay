"""Read-only, streaming comparisons of predictions on identical token contexts.

These measurements do not perform autoregressive rollouts. A strict inversion
witness proves a vocabulary ranking changed; absence of a top-k witness does
not establish that the entire vocabulary ordering stayed unchanged.
"""
import torch


class PredictionChanges:
    def __init__(self, top_k=10, max_witnesses=8):
        if type(top_k) is not int or top_k < 1:
            raise ValueError('positive integer top_k required')
        self.top_k = top_k
        self.max_witnesses = max_witnesses
        self.tokens = 0
        self.counts = dict(argmax_changes=0, topk_order_changes=0,
                           topk_membership_changes=0, strict_ranking_witness_tokens=0,
                           argmax_changes_with_teacher_top_tie=0)
        self.margins = []
        self.witnesses = []

    @torch.no_grad()
    def update(self, teacher_logits, student_logits, token_ids=None):
        if (teacher_logits.shape != student_logits.shape or teacher_logits.ndim < 2
                or teacher_logits.device != student_logits.device):
            raise ValueError('matching token/vocabulary shapes and devices required')
        vocab = teacher_logits.shape[-1]
        if vocab < max(2, self.top_k):
            raise ValueError('vocabulary must contain top_k and at least two tokens')
        t = teacher_logits.detach().reshape(-1, vocab)
        s = student_logits.detach().reshape(-1, vocab)
        if not bool(torch.isfinite(t).all() & torch.isfinite(s).all()):
            raise ValueError('finite logits required')
        if token_ids is not None and len(token_ids) != len(t):
            raise ValueError('one context ID per token position required')
        if len(t) == 0:
            return
        tv, ti = t.topk(max(2, self.top_k), dim=-1, sorted=True)
        _, si = s.topk(self.top_k, dim=-1, sorted=True)
        ti = ti[:, :self.top_k]
        ta, sa = t.argmax(-1), s.argmax(-1)
        changed = ta != sa
        order_changed = (ti != si).any(-1)
        membership_changed = (ti.sort(-1).values != si.sort(-1).values).any(-1)

        # Check the old winner against the new winner, adjacent members of the
        # teacher's top-k, and the old kth token against every new top-k token.
        # The last set detects strict crossings into the top-k from outside it.
        pairs_a = torch.cat((ta[:, None], ti[:, :-1], ti[:, -1:].expand(-1, self.top_k)), -1)
        pairs_b = torch.cat((sa[:, None], ti[:, 1:], si), -1)
        tdiff = t.gather(1, pairs_a).double() - t.gather(1, pairs_b).double()
        sdiff = s.gather(1, pairs_a).double() - s.gather(1, pairs_b).double()
        inversions = (tdiff > 0) & (sdiff < 0)
        strict = inversions.any(-1)
        masks = dict(argmax_changes=changed, topk_order_changes=order_changed,
                     topk_membership_changes=membership_changed,
                     strict_ranking_witness_tokens=strict,
                     argmax_changes_with_teacher_top_tie=changed & (tv[:, 0] == tv[:, 1]))
        for key, mask in masks.items():
            self.counts[key] += int(mask.sum())

        # Bound temporary vocabulary-sized FP64 storage even for a damaged
        # model that changes every prediction in a large input microbatch.
        for indices in changed.nonzero().flatten().split(256):
            if not len(indices):
                continue
            selected = t.index_select(0, indices).double()
            logz = selected.logsumexp(-1)
            top = tv.index_select(0, indices)[:, :2].double()
            new_choice = selected.gather(1, sa.index_select(0, indices)[:, None]).squeeze(1)
            margins = torch.stack((top[:, 0] - top[:, 1],
                                  (top[:, 0] - logz).exp() - (top[:, 1] - logz).exp(),
                                  (top[:, 0] - logz).exp() - (new_choice - logz).exp()), -1)
            self.margins.append(margins.cpu())

        room = self.max_witnesses - len(self.witnesses)
        for row in strict.nonzero().flatten()[:max(0, room)].tolist():
            pair = int(inversions[row].nonzero()[0])
            a, b = int(pairs_a[row, pair]), int(pairs_b[row, pair])
            identity = self.tokens + row if token_ids is None else token_ids[row]
            if isinstance(identity, torch.Tensor):
                identity = identity.tolist()
            self.witnesses.append(dict(context_id=identity, token_a=a, token_b=b,
                teacher_logit_a=float(t[row, a]), teacher_logit_b=float(t[row, b]),
                student_logit_a=float(s[row, a]), student_logit_b=float(s[row, b])))
        self.tokens += len(t)

    def summary(self):
        result = dict(tokens=self.tokens, top_k=self.top_k, **self.counts,
                      fractions={k: v / self.tokens if self.tokens else None
                                 for k, v in self.counts.items()},
                      strict_ranking_witnesses=list(self.witnesses),
                      scope='Same input contexts; no autoregressive rollout. Strict ranking count is a lower bound from tested pairs, not an exhaustive vocabulary comparison.',
                      topk_tie_policy='Backend topk ordering; strict inversion witnesses exclude ties.')
        result['teacher_margins_at_argmax_changes'] = {}
        if self.margins:
            values = torch.cat(self.margins)
            for col, name in enumerate(('top1_top2_logit_gap', 'top1_top2_probability_gap',
                                        'old_winner_new_winner_probability_gap')):
                q = torch.quantile(values[:, col], torch.tensor([0., .5, .99, 1.], dtype=torch.float64))
                result['teacher_margins_at_argmax_changes'][name] = dict(zip(('min', 'median', 'p99', 'max'), q.tolist()))
        return result
