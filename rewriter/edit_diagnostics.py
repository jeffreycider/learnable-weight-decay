"""Optional detached measurements; never actor inputs or training objectives.

One reusable FP32 bank stores realized edits (4 bytes per editable parameter).
FP64 reductions are chunked. Reversal uses the adapter's actual FP32 base
motion and stored realized FP32 edit, not a thin proposal/sketch approximation.
For 162354816 parameters the bank costs 649419264 bytes (619.33 MiB); it is
optimizer diagnostics storage outside the low-bandwidth actor interface. Zero
edit/base-motion denominators produce zero fraction/cosine with an explicit
previous_edit_measured flag. A reversal fraction of1 means the next base step
has a component that fully opposes the prior edit; unrelated orthogonal motion
can still be present. Positive fraction is opposition, negative reinforcement.

Forward-cast measurements assume the selected PR323 activation dtype convention;
they measure parameter casts, not output/function preservation.
"""
import math
import torch


def parameter_family(name, parameter):
    return ('embedding' if 'embed' in name else 'head' if name.startswith('proj.')
            else 'hidden_matrices' if parameter.ndim == 2 else 'vectors')


class EditDiagnostics:
    def __init__(self, parameters, device, chunk_elements=1_048_576, forward_dtype=torch.bfloat16):
        self.families = {name:parameter_family(name,p) for name,p in parameters.items()}
        self.device, self.chunk_elements = device, chunk_elements
        self.forward_dtype = forward_dtype
        self.forward_label = ('forward_bfloat16_edit_norm' if forward_dtype == torch.bfloat16
                              else 'forward_float32_edit_norm' if forward_dtype == torch.float32
                              else 'forward_cast_edit_norm')
        self.previous_edit = {}
        self.previous_norm_sq = self._zero()
        self.pending = False

    def _zero(self):
        return torch.zeros((),device=self.device,dtype=torch.float64)

    @property
    def resident_bytes(self):
        return sum(x.numel()*x.element_size() for x in self.previous_edit.values())

    def _chunks(self,tensor):
        row_width = tensor[0].numel() if tensor.shape[0] else 1
        step = max(1,self.chunk_elements//max(1,row_width))
        for start in range(0,tensor.shape[0],step):
            yield slice(start,start+step)

    @torch.no_grad()
    def begin_commit(self):
        if self.pending:
            raise RuntimeError('measure previous edit before committing another')
        self.norm_change, self.previous_norm_sq, self.forward_norm_sq = self._zero(),self._zero(),self._zero()
        self.family_norm_change = {family:self._zero() for family in set(self.families.values())}
        self.family_forward_sq = {family:self._zero() for family in set(self.families.values())}
        self.written = set()

    @torch.no_grad()
    def prepare_parameter(self,name,target):
        if name not in self.previous_edit:
            self.previous_edit[name] = torch.empty(target.shape,device=target.device,dtype=torch.float32)
        self.written.add(name)

    @torch.no_grad()
    def record_chunk(self,name,start,before,after,actual):
        self.previous_edit[name][start:start+len(actual)].copy_(actual)
        for chunk in self._chunks(actual):
            old,new = before[chunk].double(),after[chunk].double()
            change = ((new-old)*(new+old)).sum()
            self.norm_change += change
            self.family_norm_change[self.families[name]] += change
            self.previous_norm_sq += actual[chunk].double().square().sum()
            old_forward = before[chunk].to(self.forward_dtype).double()
            new_forward = after[chunk].to(self.forward_dtype).double()
            forward_sq = (new_forward-old_forward).square().sum()
            self.forward_norm_sq += forward_sq
            self.family_forward_sq[self.families[name]] += forward_sq

    def _forward_metrics(self,norm,families):
        result = dict(forward_cast_dtype=str(self.forward_dtype),forward_cast_edit_norm=norm)
        result[self.forward_label] = norm
        result.update({f'{self.forward_label}_{family}':value for family,value in families.items()})
        return result

    @torch.no_grad()
    def finish_commit(self):
        for name in self.previous_edit.keys()-self.written:
            self.previous_edit[name].zero_()
        self.pending = True
        return dict(realized_squared_norm_change=float(self.norm_change),
                    **{f'realized_squared_norm_change_{family}':float(value)
                       for family,value in self.family_norm_change.items()},
                    **self._forward_metrics(float(self.forward_norm_sq.sqrt()),
                                            {f:float(v.sqrt()) for f,v in self.family_forward_sq.items()}),
                    diagnostics_extra_resident_bytes=self.resident_bytes)

    @torch.no_grad()
    def begin_reversal(self):
        self.dot,self.base_norm_sq = self._zero(),self._zero()

    @torch.no_grad()
    def observe_base_update(self,name,update):
        if not self.pending:
            return
        previous = self.previous_edit.get(name)
        for chunk in self._chunks(update):
            u = update[chunk].double()
            self.base_norm_sq += u.square().sum()
            if previous is not None:
                self.dot += (u*previous[chunk].double()).sum()

    @torch.no_grad()
    def finish_reversal(self):
        present = self.pending
        norm_sq,base_sq,dot = float(self.previous_norm_sq),float(self.base_norm_sq),float(self.dot)
        self.pending = False
        return dict(previous_edit_measured=present,
                    previous_realized_edit_norm=math.sqrt(norm_sq) if present else 0.,
                    base_update_previous_edit_dot=dot if present else 0.,
                    previous_edit_reversal_fraction=-dot/norm_sq if present and norm_sq>0 else 0.,
                    base_update_previous_edit_cosine=dot/math.sqrt(norm_sq*base_sq)
                        if present and norm_sq>0 and base_sq>0 else 0.,
                    diagnostics_extra_resident_bytes=self.resident_bytes)

    def identity_metrics(self):
        self.pending = False
        return dict(realized_squared_norm_change=0.,
                    **{f'realized_squared_norm_change_{family}':0. for family in set(self.families.values())},
                    **self._forward_metrics(0.,{f:0. for f in set(self.families.values())}),
                    diagnostics_extra_resident_bytes=self.resident_bytes)

    def state_dict(self):
        return dict(pending=self.pending,previous_edit=self.previous_edit if self.pending else {},
                    previous_norm_sq=self.previous_norm_sq if self.pending else self._zero())

    def load_state_dict(self,state):
        self.pending = state['pending']
        self.previous_edit = {name:x.to(device=self.device,dtype=torch.float32)
                              for name,x in state['previous_edit'].items()}
        self.previous_norm_sq = state['previous_norm_sq'].to(self.device)
