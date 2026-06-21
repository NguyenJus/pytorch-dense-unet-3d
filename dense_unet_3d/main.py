"""Legacy entry point — thin wrapper around the console CLI.

The original ``cp <main.py>; python main.py`` workflow and the
hardcoded config path have been removed.

Use the ``dense-unet-3d`` console script instead::

    dense-unet-3d train  --config <path/to/config.yaml>
    dense-unet-3d eval   --config <path/to/config.yaml> --checkpoint <ckpt.pt>
    dense-unet-3d predict --config <path/to/config.yaml> --checkpoint <ckpt.pt> \\
                          --input <vol.nii.gz> --output <seg.nii.gz>

This file is retained only for backwards compatibility with any script that
does ``python -m dense_unet_3d.main``.  It delegates to ``cli.main()``
immediately.
"""

from dense_unet_3d.cli import main

if __name__ == "__main__":
    main()
