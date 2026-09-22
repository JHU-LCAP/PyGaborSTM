<!--
The summary for the NEXT release. Rewrite it when you bump the version in
pyproject.toml; the release workflow puts it at the top of the release page
and collapses the generated PR and commit lists underneath.

Describe what the release does for someone installing it. The generated
lists already cover what changed mechanically, so this should not repeat
them.

If this file is empty or only comments, the release falls back to the
generated notes alone.
-->

0.1.1 fixes the GPU install and tightens input checking.

- `pip install 'pygaborstm[cuda12]'` and `[cuda13]` now also install the CUDA libraries from PyPI, so a GPU install needs only the NVIDIA driver. Before, CuPy fell back to CPU on a machine without a system CUDA toolkit.
- Stereo or other multi-channel audio is rejected with a clear error instead of being flattened into interleaved samples.
