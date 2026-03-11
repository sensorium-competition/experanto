Getting Started with Experanto
==============================

Experanto helps align temporal data from neuroscience experiments. It can handle visual stimuli, behavior, and neural responses.

It fixes common issues such as:

- Different sampling rates for different devices
- Different formats for data types such as videos and images in the same experiments

Basic Usage
-----------

To start using Experanto, you can load an experiment and interpolate data across different devices:

.. code-block:: python

   from experanto.experiment import Experiment
   import numpy as np

   # Load your experiment
   exp = Experiment("/path/to/experiment")

   # Query data at specific time points
   times = np.linspace(0, 10, 100)
   
   # Get interpolated data from all devices
   data = exp.interpolate(times)