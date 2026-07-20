## Profiling Memorry
To collect aggregate memory usage across your script and its worker processes, use mprof (comes bundled with the memory_profiler package):

1. Install it (if not already available):

`uv pip install memory_profiler matplotlib`

(matplotlib is needed for the plotting step later.)

2. Run your script under mprof with the multiprocess flag:

`uv run mprof run --multiprocess -o prof/mprofile.dat  python train_bpe_tinystories.py `

This runs your script normally but samples RSS memory over time for the main process and all its children (e.g., the workers spawned via num_processes), writing the data to a .dat file (e.g., mprofile_<timestamp>.dat).

3. Visualize it:
`mprof plot`
`uv run mprof plot -o prof/memory.png prof/mprofile.dat`