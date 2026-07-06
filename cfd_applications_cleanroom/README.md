# CFD Applications Cleanroom

This route rebuilds the CFD application evidence pipeline from a clean Basilisk
source audit upward. It keeps official solver source immutable, treats
`/private/tmp/basilisk/src` as untrusted, and separates implementation, smoke
execution, contract validation, benchmark evidence, and accepted evidence.

Steps 1-4 below do not require plotting dependencies. Step 5 is fixed to the
Python figure backend; this project rejects any non-Python figure backend.

```bash
# 1. Audit source/toolchain only. No solver result is interpreted here.
PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/bin/conda run -n pinn --no-capture-output \
  python -m cfd_applications_cleanroom.cfd_apps.cli audit

# 2. Run the cheap local canary ladder three times.
PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/bin/conda run -n pinn --no-capture-output \
  python -m cfd_applications_cleanroom.cfd_apps.cli reproduce --tier canary --repeat 3

# 3. Run the accepted local matrix three times after canary gates pass.
PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/bin/conda run -n pinn --no-capture-output \
  python -m cfd_applications_cleanroom.cfd_apps.cli reproduce --tier accepted-local --repeat 3

# 4. Summarize all raw evidence for the selected run.
PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/bin/conda run -n pinn --no-capture-output \
  python -m cfd_applications_cleanroom.cfd_apps.cli summarize --latest

# 5. Plot with the fixed Python Nature-figure backend.
PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/bin/conda run -n pinn --no-capture-output \
  python -m cfd_applications_cleanroom.cfd_apps.cli figures --latest --figure-backend python
```

The initial shell only proves that the route is callable. It makes no CFD,
curvature, or performance claim until later source, schema, manifest, repeat,
and benchmark gates produce evidence bundles.
