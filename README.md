# cuImageProcessor: CUDA Bitmap Image Processor

cuImageProcessor is a C/C++ library for bitmap image processing, supporting operations such as resizing, cropping, grayscale conversion, rotation, and horizontal or vertical flipping. cuImageProcessor uses NVIDIA CUDA to accelerate image processing tasks.

## Installation
1. Download the `.so` library file and `.cuh` header files.
1. Include the library file and header files in the build process.
1. Add include statements in the C/C++ file.

## Operations
### Resize
#### Parameters
- filename: char*
- width: int
- height: int
#### Example
```cpp
#include "cuImageProcessor/bitmap.cuh"
#include "cuImageProcessor/resize.cuh"
int main()
{
    resize("input.bmp", 1024, 1024);
    return 0;
}
```

### Crop
#### Parameters
- filename: char*
- start_x: int
- start_y: int
- end_x: int
- end_y: int

#### Example
```cpp
#include "cuImageProcessor/bitmap.cuh"
#include "cuImageProcessor/crop.cuh"
int main()
{
    crop("input.bmp", 200, 200, 500, 500);
    return 0;
}
```

### Grayscale
#### Parameters
- filename: char*

#### Example
```cpp
#include "cuImageProcessor/bitmap.cuh"
#include "cuImageProcessor/grayscale.cuh"
int main()
{
    grayscale("input.bmp");
    return 0;
}
```

### Rotate
#### Parameters
- filename: char*
#### Example
```cpp
#include "cuImageProcessor/bitmap.cuh"
#include "cuImageProcessor/rotate.cuh"
int main()
{
    rotate("input.bmp");
    return 0;
}
```

### Horizontal Flip
#### Parameters
- filename: char*
#### Example
```cpp
#include "cuImageProcessor/bitmap.cuh"
#include "cuImageProcessor/flip.cuh"
int main()
{
    flip_horizontal("input.bmp");
    return 0;
}
```

### Vertical Flip
#### Parameters
- filename: char*
#### Example
```cpp
#include "cuImageProcessor/bitmap.cuh"
#include "cuImageProcessor/flip.cuh"
int main()
{
    flip_vertical("input.bmp");
    return 0;
}
```



## Example Code
- [cuImageProcessor-examples](https://github.com/YeonguChoe/cuImageProcessor-examples)

## Supported Operating System
- Linux
- Windows

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright © 2025 Yeongu Choe.