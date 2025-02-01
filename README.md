# Learn CPU

Most examples can be run with

```bash
clang++ *.cpp -O3 -o main -std=c++17 && ./main

# on macOS, using Apple Clang
clang++ *.cpp -O3 -o main -std=c++17 -framework Accelerate -DACCELERATE_NEW_LAPACK && ./main

# on macOS, using Homebrew Clang, with OpenMP
$(brew --prefix llvm)/bin/clang++ *.cpp -O3 -o main -std=c++17 -fopenmp -framework Accelerate -DACCELERATE_NEW_LAPACK && ./main
```
