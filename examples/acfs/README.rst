Advanced Controls File (ACF) Examples
======================================

Advanced Controls Files (ACFs) are highly specialized PTXAS compiler configurations for specific hardware and use cases. 

Helion provides two ways to use ACFs:

- ``advanced_controls_file`` in ``helion.Config``
- ``autotune_search_acf``

The ``advanced_controls_file`` parameter is used to specify a single ACF file to be applied to a kernel.
The ``autotune_search_acf`` parameter is used to specify a list of ACF files to be considered during autotuning.

This section contrains examples demonstrating how to use ACFs with Helion.

.. note::
   This feature is still highly experimental. It could cause incorrect results or runtime errors if ACFs are used with the wrong hardware or use case.

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:
   :glob:

   *
