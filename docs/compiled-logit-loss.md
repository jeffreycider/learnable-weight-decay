# Compile the student forward with its preservation loss

Set `compile_preservation_loss: true` to compile the edited GPT forward, raw-logit mean squared error, and maximum absolute logit difference together. The existing cached teacher remains detached. Cotangents still flow through GPT to Chisel; no mathematical objective, number of examples, or private/shared optimizer update changes.

This option requires raw MSE and disabled KL/ranking measurements. Incompatible configurations fail at setup. It is off by default until the actual workload's speed and numerical parity have been measured. It does not compile Adam or the entire offline training iteration into one kernel.
