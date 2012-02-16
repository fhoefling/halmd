halmd_add_library(halmd_mdsim_gpu_fields
  constant_force.cpp
  constant_force_kernel.cu
)
halmd_add_modules(
  libhalmd_mdsim_gpu_fields_constant_force
)
