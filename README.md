# CUDA Flocking

* Author: Keyan Zhai ([LinkedIn](https://www.linkedin.com/in/keyanzhai), [personal website](https://keyanzhai.github.io/), [twitter](https://twitter.com/KeyanZhai31533))
* Tested on: Windows 10 Pro, AMD Ryzen 7 5800X 8-Core Processor 3.80 GHz, RTX 3080 10GB (personal)

---

Flocking simulation in C++/CUDA based on the Reynolds Boids algorithm, along with two levels of
optimization: 
1. a scattered uniform grid 
2. a uniform grid with semi-coherent memory access

## Demo: 50,000 Boids Flocking

| Naive Neighbor Search (~28 fps) | Scattered Uniform Grid (~430 fps) | Coherent Uniform Grid (~925 fps) |
| --- | --- | --- |
| ![](images/naive.gif) | ![](images/scattered.gif) | ![](images/coherent.gif) |

![](images/boids.png)

In the boids flocking simulation, particles representing birds or fish
(boids) move around the simulation space according to three rules:

1. cohesion - boids move towards the perceived center of mass of their neighbors
2. separation - boids avoid getting to close to their neighbors
3. alignment - boids generally try to move with the same direction and speed as
their neighbors


## Performance Analysis

### Questions

* Q: For each implementation, how does changing the number of boids affect performance? Why do you think this is?
  * A: For all the 3 implementations, increasing the number of boids lowers the performance. Because more boids requires more computation for their velocity and position.

    | Number of boids | Naive (visualization enabled / disabled)| Scattered Uniform Grid (visualization enabled / disabled) | Coherent Uniform Grid (visualization enabled / disabled) | 
    | ----------- | ----------- | ----------- | ----------- |
    | 5000    | ~445 fps / ~510 fps  |  ~1180 fps / ~1650 fps  |   ~1300 fps / ~1980 fps    |
    | 10000   | ~210 fps / ~225 fps  | ~930 fps / ~1200 fps  | ~1170 fps / ~1730 fps    |
    | 20000   | ~106 fps / ~110 fps  | ~690 fps / ~870 fps   | ~1090 fps / ~1560 fps    |
    | 50000   | ~28 fps / ~28 fps  | ~430 fps / ~500 fps  | ~925 fps / ~1200 fps    | 
    | 100000  | ~8.6 fps / ~8.6 fps  | ~150 fps / ~155 fps   |  ~570 fps / ~820 fps     |
    | 200000  | ~2.4 fps / ~2.4 fps   | ~65 fps / ~70 fps  | ~310 fps / ~440 fps     |
    | 500000  | ~0 fps / ~0 fps   | ~10.8 fps / ~10.8 fps  | ~105 fps / ~130 fps     |
    | 1000000 | ~0 fps / ~0 fps   | ~1 fps / ~1 fps  | ~38 fps / ~40 fps    |

    Note: visualization disabled was tested on Microsoft Remote Desktop.

* Q: For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

  * A: For all the 3 implementations, changing the block count and block size does not have a significant influence on the performance. This is because different block sizes or different block counts are just different arrangement of the threads. The number of threads and how they are executed are still the same. There is also no shared memory use or synchronization between the threads within a block, so how you organize the threads into blocks does not affect the performance.
  
    | Block Size | Block Count (boids) | Block Count (cells) | Naive | Scattered Uniform grid | Coherent Uniform Grid | 
    | ----------- | ----------- | ----------- | ----------- | --- | --- |
    | 16 | 625 |  666   |  ~228 fps   | ~1330 fps  |   ~1780 fps |
    | 32 (warp size) | 313 | 333 | ~232  fps  | ~1233 fps  |  ~1700 fps    |
    | 64 | 157 | 167 | ~230  fps  | ~1224 fps  | ~1710 fps    |
    | 128 | 79 | 84  |  ~225 fps  |  ~1200 fps  |   ~1730 fps    |
    | 165 (not 2's power) | 61 | 65 | ~236 fps  | ~1230 fps  | ~1720 fps |
    | 256 | 40 | 42 | ~238 fps  | ~1200 fps  | ~1720 fps    | 
    | 512  | 20 | 21 | ~217 fps  |  ~1140 fps | ~1710 fps |
    | 1024 (max) | 10 | 11 | ~150 fps | ~1070 fps | ~1600 fps | 

    Note: visualization disabled, tested on Microsoft Remote Desktop.

    Block count is computed with block size. The block size by default is 128, which means there are 128 threads in one block. Changing the block size, the block count can be calculated accordingly. 

    For kernels that execute once for each boid:
    ```C++
    dim3 boidBlocksPerGrid((num_boids + blockSize - 1) / blockSize);
    ```

    For kernels that execute once for each grid cell:
    ```C++
    dim3 cellBlocksPerGrid((num_cells + blockSize - 1) / blockSize);
    ```

    Here we use 10000 boids for testing. By defualt, number of cells = 22 * 22 * 22 = 10648. So we have:
    
    ```
    Block Count (boids) = (10000 + blockSize - 1) / blockSize
    ```

    ```
    Block Count (cells) = (10648 + blockSize - 1) / blockSize
    ```

* Q: For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
  * A: (Question: How do you define "more coherent uniform grid"? What's the baseline?) Coherent is referring to the memory access to the boids' data `dev_pos` and `dev_vel1`. There is significant performance improvements with the more coherent uniform grid compared to the scattered uniform grid implementation. The outcome is as expected because there is less global memory access with the coherent uniform grid.


* Q: Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
  * A: Cell width by default is 2 times the neighboring distance `d`, and 8 neighboring cells will be checked at maximum. Changing the cell width to 1 neighboring distance, 27 neighboring cells will be checked, but the performance improves.
  Changing cell width and checking 27 vs 8 neighboring cells did affect performance. The reason is that checking 8 neighboring cells instead of all the 27 neighboring cells will cause branch divergence for the threads within a warp, which will lower the performance.

    | Cell Width | Naive | Scattered Uniform Grid | Coherent Uniform Grid | 
    | ----------- | ----------- | ----------- | ----------- |
    | 2 * d   | / |  ~1200 fps  |   ~1730 fps    |
    | 1 * d   | / |  ~2040 fps  | ~2180 fps    |

    Note: 10000 boids, visualization disabled, tested on Microsoft Remote Desktop.

    ---

    **University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**