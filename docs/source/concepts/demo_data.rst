.. _generating_sample_data:

Experanto data format
======================

The **Experanto** data format is designed to handle individual experiments, each of which can include multiple modalities.
Currently, supported modalities include:

- **Screen** (e.g., visual stimuli presentation)
- **Responses** (e.g., recorded subject neural responses)
- **Behaviors** (e.g., treadmill movement, eye tracking data)

Each modality has its own dedicated folder containing the relevant data and metadata.

File structure
--------------
A typical experiment follows this structured hierarchy:

.. code-block:: text

    experiment/
      ├── screen/
      │   ├── data/
      │   │   ├── 00000.npy
      │   │   ├── 00001.mp4
      │   │   ├── ...
      │   │   ├── means.npy
      │   │   ├── std.npy
      │   ├── meta/
      │   │   ├── 00000.yml
      │   │   ├── 00001.yml
      │   │   ├── ...      
      │   ├── meta.yml
      │   ├── timestamps.npy
      ├── responses/
      │   ├── data.mem
      │   ├── meta.yml
      │   ├── meta/
      │   │   ├── means.npy
      │   │   ├── std.npy
      ├── behaviors/ (similar structure to responses)

Generating sample data
----------------------
A sample dataset for testing can be generated using the **allen_exporter** library, which provides neurological experiment data on mice in the **Experanto** format.

The `allen_exporter <https://github.com/sensorium-competition/allen-exporter>`_ repository contains tools to generate datasets that include all the modalities mentioned above.

For more details, visit the repository and follow the installation and usage instructions.


