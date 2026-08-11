# LAMMPS build options (reference)

This records the previously used CMake options. It is not a complete current
Jupiter/connector installation recipe; see `SETUP_ENV.md`,
`../md_setup/README.md`, and the historical `CONNECTOR_REBUILD.md`.

```bash
cmake ../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DPKG_PLUGIN=ON \
  -DPKG_MOLECULE=ON \
  -DPKG_RIGID=ON \
  -DPKG_KSPACE=ON \
  -DPKG_GPU=ON \
  -DGPU_API=cuda \
  -DGPU_ARCH=sm_90
```
