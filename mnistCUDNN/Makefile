# mnistCUDNN — direct build (no CMake/FreeImage subdir needed).
# Uses the system cudnn/cublas/cudart + libfreeimage-dev. Orin / aarch64, CUDA 12.6.
NVCC   ?= nvcc
CCBIN  ?= g++
TARGET  = mnistCUDNN
SRCS    = mnistCUDNN.cpp fp16_dev.cu fp16_emu.cpp
LIBS    = -lcudnn -lcublas -lcudart -lfreeimage
CFLAGS  = -I/usr/include -Xcompiler -fpermissive

$(TARGET): $(SRCS)
	$(NVCC) -ccbin $(CCBIN) $(CFLAGS) -o $(TARGET) $(SRCS) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
