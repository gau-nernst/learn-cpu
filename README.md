# Learn CPU

Most examples can be run with

```bash
clang++ *.cpp -O3 -o main -std=c++17 && ./main

# on macOS
clang++ *.cpp -O3 -o main -std=c++17 -framework Accelerate -DACCELERATE_NEW_LAPACK && ./main
```
