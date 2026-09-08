# Source And Toolchain Audit

- overall_status: `PASS`
- selected_source: `clean`

## Vendor Source
- pre_rename_path: `cfd_applications_cleanroom/vendor/basilisk`
- post_rename_path: `cfd_applications_cleanroom/vendor/basilisk_clean`
- src_hashes_unchanged: `True`

## Native qcc Build
- config_symlink_target: `config.gcc`
- make_qcc_rc: `0`
- native_qcc_exists: `True`
- file_qcc: `/mnt/e/Research/PDE/MLP-field/PINN/cfd_applications_cleanroom/build/basilisk_arm64/src/qcc: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=4b4818b7359081643b71111f424d742a1f13707f, for GNU/Linux 3.2.0, with debug_info, not stripped`

## Native No-NN Canary Compile
- compile_rc: `0`
- binary_exists: `True`
- compile_log: `cfd_applications_cleanroom/results/raw/toolchain/native_no_nn_canary_clean_qcc.compile.log`
