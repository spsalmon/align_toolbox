# ALIGN toolbox!

This package implements most of the functions used in the ALIGN pipeline, a modular image analysis workflow adapted for longitudinal, high-troughput microscopy. Developed by the Towbin Lab of the University of Bern.

This package goes hand in hand with our modular pipelining tool : <https://github.com/spsalmon/align_pipeline>

Documentation : <https://align-toolbox.readthedocs.io/en/latest/index.html>
## Install the package using pip

Simply run the following command:

```bash
pip3 install align_toolbox
```

## Build the package and install it

1. First, make sure build is installed:

   ```bash
   pip3 install build

   ```

2. Go to the package directory, eg:

   ```bash
   cd ~/align_toolbox

   ```

3. Build the package:

   ```bash
   python3 -m build

   ```

4. Install the package you just built:

   ```bash
   pip3 install dist/*.whl
   ```
