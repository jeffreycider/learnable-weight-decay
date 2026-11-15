"""Experimental dense FP32 student F/B capture, with the loss outside the graph.

This is deliberately a single-engine, fixed-microbatch helper. It preserves the
ordinary dense parameter VJP and lets tokenwise_panel own the unchanged teacher,
relative ranking loss, metrics, checkpoint weighting and shared/private credit.
Checkpoint reloads require explicit host-side binding. This is not a production
backend or a qualified cross-checkpoint capture cache.
"""
import torch


def _layout(value):
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device,
            value.requires_grad)


class DenseStudentGraph:
    """One in-flight dense student forward/backward on a fixed engine.

    Construct at a clean boundary using a real proposal's parameter dictionary.
    Teacher execution continues through the original engine, not this wrapper.
    Buffers are explicit graph inputs, copied like dense weights. Different
    editable masks, shapes, inference, and concurrent/reentrant use fail closed.
    """
    def __init__(self, engine, parameters, inputs, *, warmups=2,
                 action_family='fixed_anchored_v1'):
        if action_family != 'fixed_anchored_v1':
            raise ValueError('dense student graph is scoped to fixed_anchored_v1')
        if inputs.device.type != 'cuda':
            raise ValueError('dense student graph requires CUDA')
        self._initialize(engine, parameters, inputs, torch.cuda.make_graphed_callables,
                         warmups=warmups)

    def _initialize(self, engine, parameters, inputs, factory, *, warmups):
        # Separated only to exercise the ownership contract on CPU; the public
        # constructor always requires a CUDA capture.
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise ValueError('dense student capture requires gradient-enabled training mode')
        if engine.gpt_precision != 'fp32':
            raise ValueError('strict FP32 GPT required')
        if torch.is_autocast_enabled(inputs.device.type):
            raise ValueError('strict dense capture requires autocast disabled')
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError('TF32 must remain disabled')
        if inputs.ndim != 2 or inputs.dtype not in (torch.int32, torch.int64):
            raise ValueError('fixed integer token microbatch required')
        if not parameters or not any(v.requires_grad for v in parameters.values()):
            raise ValueError('a differentiable dense proposal is required')
        if any(v.dtype != torch.float32 or v.device != inputs.device for v in parameters.values()):
            raise ValueError('all dense parameters must be FP32 on the input device')
        engine._validate_replacements(parameters)
        self.engine = engine
        self.names = tuple(parameters)
        self.layouts = {n: _layout(v) for n, v in parameters.items()}
        self.input_layout = _layout(inputs)
        self.forward_callable = engine._forward_callable
        self.buffers = dict(engine.buffers)
        self.buffer_names = tuple(self.buffers)
        self.buffer_layouts = {n:_layout(v) for n,v in self.buffers.items()}
        self.buffer_signature = self._buffer_signature()
        self.training = engine.view.training
        self.module_signature = self._module_signature()
        self.teacher_id = engine.teacher_id
        self.pending = False
        self.bound = True
        self.usable = True
        self.action_family = 'fixed_anchored_v1'
        self._backward_scheduled = False
        self.replays = 0
        self._leaves = None
        # Warmup/capture must not populate the live proposal's leaf gradients.
        self.samples = tuple(v.detach().clone(memory_format=torch.preserve_format)
                             .requires_grad_(v.requires_grad) for v in parameters.values())
        self.sample_buffers = tuple(v.detach().clone(memory_format=torch.preserve_format)
                                    for v in self.buffers.values())
        self.sample_inputs = inputs.detach().clone(memory_format=torch.preserve_format)
        originals = (*parameters.values(),*self.buffers.values(),inputs)
        samples = (*self.samples,*self.sample_buffers,self.sample_inputs)
        if any(_layout(a) != _layout(b) for a,b in zip(originals,samples)):
            raise ValueError('sample cloning changed tensor layout; qualify that layout separately')

        def forward(*values):
            count = len(self.names)
            output = self.forward_callable(dict(zip(self.names, values[:count])),
                dict(zip(self.buffer_names, values[count:-1])), values[-1])
            if not output.requires_grad:
                raise RuntimeError('dense student capture lost its parameter VJP')
            return output

        with torch.enable_grad():
            self.call = factory(forward, samples,
                                num_warmup_iters=warmups, allow_unused_input=True)

    def bind_checkpoint(self):
        """Acknowledge a verified reload into the SAME engine outside capture.

        The caller still owns bank/hash/teacher validation. This method accepts
        changed weight/rotary values, never a different program or tensor layout.
        It performs no teacher reconstruction and no optimizer/gradient reset.
        """
        if self.pending:
            raise RuntimeError('finish the previous dense student VJP before binding')
        self.bound = False
        if not self.usable:
            raise RuntimeError('failed graph replay requires a new capture')
        if self.engine._forward_callable is not self.forward_callable or self._module_signature() != self.module_signature:
            raise RuntimeError('captured forward implementation or training mode changed')
        if tuple(self.engine.buffers) != self.buffer_names or any(
                _layout(self.engine.buffers[n]) != self.buffer_layouts[n] for n in self.buffer_names):
            raise ValueError('captured buffer layout changed')
        self.buffers = dict(self.engine.buffers)
        self.buffer_signature = self._buffer_signature()
        self.teacher_id = self.engine.teacher_id
        self.bound = True

    def _buffer_signature(self):
        return {n: (id(v), v.data_ptr(), v._version, _layout(v))
                for n, v in self.engine.buffers.items()}

    def _module_signature(self):
        # Includes execution attributes such as the view's precision mode and
        # attention head count, not only registered tensor shapes.
        return tuple((n,id(m),m.training,tuple(sorted((k,v) for k,v in vars(m).items()
                     if isinstance(v,(bool,int,float,str,torch.dtype)))))
                     for n,m in self.engine.view.named_modules())

    def __getattr__(self, name):
        if name == 'engine':
            raise AttributeError(name)
        return getattr(self.engine, name)

    def discard(self):
        """After a failed backward, abandon this capture and release live leaves.

        This is terminal: binding cannot make the capture usable again. Call
        after the failed autograd task has returned, not concurrently with it.
        Delete the wrapper at a safe boundary to release its private GPU pool.
        """
        self.usable = self.bound = self.pending = False
        self._backward_scheduled = False
        self._leaves = None

    def _backward_started(self, gradient, generation):
        if not self.pending or generation != self.replays or self._backward_scheduled:
            raise RuntimeError('stale or repeated dense student backward is forbidden')
        if tuple((v.data_ptr(),v._version,_layout(v)) for v in self._leaves) != self._leaf_signature:
            self.discard()
            raise RuntimeError('dense student weights changed before their backward')
        if self.engine.teacher_id != self.teacher_id or self._buffer_signature() != self.buffer_signature:
            self.discard()
            raise RuntimeError('engine checkpoint changed before dense student backward')
        if self._module_signature() != self.module_signature:
            self.discard()
            raise RuntimeError('captured program changed before dense student backward')
        if self.engine._forward_callable is not self.forward_callable:
            self.discard()
            raise RuntimeError('captured forward implementation changed before backward')
        self._backward_scheduled = True
        torch.autograd.Variable._execution_engine.queue_callback(self._backward_complete)
        return gradient

    def _backward_complete(self):
        # AccumulateGrad can adopt graph-returned storage when grad is None.
        # Clone before another microbatch overwrites those replay buffers. This
        # preserves true accumulation rather than silently aliasing the last VJP.
        for value in self._leaves:
            if value.requires_grad and value.grad is not None:
                value.grad = value.grad.clone()
        self._leaves = None
        self.pending = False
        self._backward_scheduled = False

    def _forward(self, parameters, inputs):
        """Return ephemeral logits: clone if retaining values past the next replay."""
        if self.pending:
            raise RuntimeError('finish the previous dense student VJP before replay')
        if not self.bound or not self.usable:
            raise RuntimeError('dense student graph is not bound to a verified checkpoint')
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError('use the original engine for inference/readouts')
        if tuple(parameters) != self.names or any(_layout(parameters[n]) != self.layouts[n] for n in self.names):
            raise ValueError('dense parameter layout or editable mask changed')
        if any(not value.is_leaf for value in parameters.values()):
            raise ValueError('tokenwise panel detached dense leaves are required')
        if _layout(inputs) != self.input_layout:
            raise ValueError('token microbatch layout changed')
        if self.engine._forward_callable is not self.forward_callable or self._module_signature() != self.module_signature:
            raise RuntimeError('captured forward implementation or training mode changed')
        if self.engine.teacher_id != self.teacher_id or self._buffer_signature() != self.buffer_signature:
            raise RuntimeError('captured engine identity/buffers changed; recapture explicitly')
        if self.action_family != 'fixed_anchored_v1':
            raise RuntimeError('captured action family changed')
        if (self.engine.gpt_precision != 'fp32' or torch.is_autocast_enabled(inputs.device.type)
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise RuntimeError('captured strict precision settings changed')
        self._leaves = tuple(parameters.values())
        self._leaf_signature = tuple((v.data_ptr(),v._version,_layout(v)) for v in self._leaves)
        self.pending = True
        self.replays += 1
        generation = self.replays
        try:
            output = self.call(*self._leaves, *self.buffers.values(), inputs)
            output.register_hook(lambda gradient: self._backward_started(gradient, generation))
        except BaseException:
            self.discard()
            raise
        return output
