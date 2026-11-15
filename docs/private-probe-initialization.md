# New checkpoints start without an edit

**A trained shared backbone can give fresh random probes a harmful first edit.** `zero_write_left` instead zeros each private left write factor. Right write factors and read probes retain their seeded random initialization. Thus both initial edits, ΔR and ΔC, are exactly zero.

**The zero edit remains trainable when the shared backbone emits nonzero codes.** For an edit `L K Rᵀ`, setting only `L=0` leaves its derivative with respect to L nonzero in general. The implementation rejects a shared backbone whose effective codes are all zero, which would make this initialization unable to begin learning. After the first left-factor update, the other private factors can also receive gradients. Loading previously fitted probes never reapplies initialization.

Select [the initialization recipe](../recipes/zero_edit_private_probes.yaml) after loading a trained shared Chisel checkpoint. This is an optional initialization experiment; it changes neither the shrinkage reward nor the preservation constraints and makes no performance claim.
