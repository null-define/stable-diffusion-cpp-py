# stable-diffusion-cpp-py

python bind for stable-diffusion.cpp

## description

a simple stable_diffusion_cpp python wrapper by using pybind11 and magic_enum, aim to provide a python interface for stable-diffusion.cpp to build upper application

## install

- if you use nvidia gpu, execute `CMAKE_ARGS="-DSD_CUBLAS=on" pip install git+https://github.com/null-define/stable-diffusion-cpp-py` install
- if you use amd gpu, execute `CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ CMAKE_ARGS="-DSD_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1100" pip install git+https://github.com/null-define/stable-diffusion-cpp-py` install

- To upgrade and rebuild stable-diffusion-cpp-py add --upgrade --force-reinstall --no-cache-dir flags to the pip install command to ensure the package is rebuilt from source.

## References

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [pybind11](https://github.com/pybind/pybind11)
- [stable-diffusion-cpp-py](https://github.com/abetlen/llama-cpp-python)
