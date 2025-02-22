# Learn CPU

Profiling

```bash
sudo perf stat -d ./main
```

```
CPU max theoretical FLOP/s = per-core FLOP/s x num cores
                           = per-core FLOP/cycle x clock rate x num cores
```

Without pipelining (instruction-level parallelism - ILP), FLOP/cycle ~ inverse latency of a particular instruction. With ILP, we can look at instruction per cycle (IPC) instead.
- E.g. [_mm256_fmadd_ps](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmadd_ps) has cycle per instruction (CPI) of 0.5, and it does 16 FLOPs (8 multiply and 8 add) -> FLOP/cycle = 16 / 0.5 = 32
- You can also get throughput for floating point ops at https://www.agner.org/optimize/instruction_tables.pdf or https://www.uops.info/table.html
- Using vector instructions may lower clock speed.
- VFMADD_PS has ~4 cycles latency -> we have to schedule 4 / 0.5 = 8 independent VFMADD_PS instructions to exploit ILP.
