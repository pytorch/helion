"""Selection emitters of the CuTe backend: top-k, sort and radix select.

Matchers (``topk``, ``row_topk``, ``radix_selection``, ``ordered_selection``)
prove a root is a selection and build its plan; the emitters (``topk_codegen``,
``row_topk_codegen``, ``radix_selection_codegen``) render it; the traced selection
programs (``selection_network``, ``selection_coarse`` over ``sorting_networks``) are
lowered through the register-program bridge in ``selection_lowering``.  The
fragment-resident forms live in ``..fragment_selection``.

Role: planner and special-purpose emitter (section 3.2 of ``../README.md``).
"""
