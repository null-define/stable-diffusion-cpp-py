# stable_diffusion_cpp_py

python bind for stable_diffusion_cpp by pybind11

## install

- if you use nvidia gpu, execute `CMAKE_ARGS="-DSD_CUBLAS=on" pip install git+https://github.com/null-define/stable-diffusion-cpp-py` install
- if you use amd gpu, execute `CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ CMAKE_ARGS="-DSD_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1100" pip install git+https://github.com/null-define/stable-diffusion-cpp-py` install


## References
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [pybind11](https://github.com/pybind/pybind11)