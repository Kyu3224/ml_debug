import pynvml


def gpu_memory_left(msg=""):
    """
        Display detailed GPU memory status for all available devices.

        Args:
            msg (str): Optional tag or context message for the output (e.g., "Before training", "After inference").

        Behavior:
            - Initializes NVML and queries each available GPU.
            - Prints total, used, and free memory in megabytes for each GPU.
            - Gracefully shuts down NVML after querying.

        Example:
            gpu_memory_left("After loading model")

        Output Example:
            Displaying GPU Memory Left After loading model
            GPU 0 (NVIDIA RTX 4090):
              Total Memory : 24576.00 MB
              Used Memory  : 3024.25 MB
              Free Memory  : 21551.75 MB
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"Displaying GPU Memory Left {msg}")
        print(f"GPU {i} ({name.encode()}):")
        print(f"  Total Memory : {mem_info.total / 1024 ** 2:.2f} MB")
        print(f"  Used Memory  : {mem_info.used / 1024 ** 2:.2f} MB")
        print(f"  Free Memory  : {mem_info.free / 1024 ** 2:.2f} MB\n")
    pynvml.nvmlShutdown()


class GpuMemoryTracer:
    """
        Context manager to trace GPU memory usage before and after a code block.

        Args:
            device_index (int): GPU index to trace (default: 0).
            label (str): Descriptive label for logging (e.g., "Model Allocation").

        Behavior:
            - On __enter__, records the used GPU memory.
            - On __exit__, records memory again and computes the difference.
            - Prints memory before, after, and delta in MB.

        Example:
            with GpuMemoryTracer(label="Model Allocation"):
                model = MyModel().cuda()
                input = torch.randn(1, 3, 224, 224).cuda()
                output = model(input)

        Output Example:
            [Model Allocation] GPU memory before: 1452.13 MB
            [Model Allocation] GPU memory after:  2480.50 MB
            [Model Allocation] GPU memory change: 1028.37 MB
    """
    def __init__(self, device_index=0, label=""):
        self.device_index = device_index
        self.label = label

    def __enter__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.used_before = mem_info.used
        print(f"[{self.label}] GPU memory before: {self.used_before / 1024**2:.2f} MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.used_after = mem_info.used
        print(f"[{self.label}] GPU memory after:  {self.used_after / 1024**2:.2f} MB")
        diff = self.used_after - self.used_before
        print(f"[{self.label}] GPU memory change: {diff / 1024**2:.2f} MB\n")
        pynvml.nvmlShutdown()
