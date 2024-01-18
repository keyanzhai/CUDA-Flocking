#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount; // Number of total cells in the grid (3d), `gridSideCount` cubed
int gridSideCount; // Number of cells on one side of the grid (1d)
float gridCellWidth; // Width of the cell, defined as double the neighborhood distance
float gridInverseCellWidth; // Inverse of the cell width
glm::vec3 gridMinimum; // The minimum xyz coordinates of the grid, used to compute the cell index of a boid

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount; // Number of cells on one side of the grid (1d)

  gridCellCount = gridSideCount * gridSideCount * gridSideCount; // Number of total cells in the grid (3d), `gridSideCount` cubed
  gridInverseCellWidth = 1.0f / gridCellWidth; // Inverse of the cell width
  float halfGridWidth = gridCellWidth * halfSideCount;

  // The minimum xyz coordinates of the grid
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* Velocity change contribution by rule 1 Cohesion:
* Boids try to fly towards the center of mass of neighboring boids.
* 
*/
__device__ glm::vec3 computeVelocityChangeRule1(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
  glm::vec3 perceived_center(0.0f, 0.0f, 0.0f);
  int num_neighbors = 0;

  for (int b = 0; b < N; ++b) {
    if ((b != iSelf) && (glm::distance(pos[iSelf], pos[b]) < rule1Distance)) {
      perceived_center += pos[b];
      num_neighbors++;
    }
  }

  perceived_center /= num_neighbors;

  return (perceived_center - pos[iSelf]) * rule1Scale;
}

/**
* Velocity change contribution by rule 2 Seperation:
* Boids try to keep a small distance away from other objects (including other boids).
* 
*/
__device__ glm::vec3 computeVelocityChangeRule2(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
  glm::vec3 c(0.0f, 0.0f, 0.0f);

  for (int b = 0; b < N; ++b) {
    if ((b != iSelf) && (glm::distance(pos[iSelf], pos[b]) < rule2Distance)) {
      c -= (pos[b] - pos[iSelf]);
    }
  }

  return c * rule2Scale;
}

/**
* Velocity change contribution by rule 3 Alignment:
* Boids try to match velocity with near boids.
*/
__device__ glm::vec3 computeVelocityChangeRule3(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
  glm::vec3 perceived_velocity(0.0f, 0.0f, 0.0f);
  int num_neighbors = 0;

  for (int b = 0; b < N; ++b) {
    if ((b != iSelf) && (glm::distance(pos[iSelf], pos[b]) < rule3Distance)) {
      perceived_velocity += vel[b];
      num_neighbors++;
    }
  }

  perceived_velocity /= num_neighbors;
  return perceived_velocity * rule3Scale;
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  glm::vec3 overall_vel_change(0.0f, 0.0f, 0.0f);

  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  glm::vec3 rule1_vel_change = computeVelocityChangeRule1(N, iSelf, pos, vel);

  // Rule 2: boids try to stay a distance d away from each other
  glm::vec3 rule2_vel_change = computeVelocityChangeRule2(N, iSelf, pos, vel);
  
  // Rule 3: boids try to match the speed of surrounding boids
  glm::vec3 rule3_vel_change = computeVelocityChangeRule3(N, iSelf, pos, vel);
  overall_vel_change += rule1_vel_change + rule2_vel_change + rule3_vel_change;

  return overall_vel_change;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {

  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
      return;
  }

  // Compute a new velocity based on pos and vel1
  glm::vec3 curr_vel = vel1[index];
  glm::vec3 vel_change = computeVelocityChange(N, index, pos, vel1);
  glm::vec3 new_vel = curr_vel + vel_change;

  // Clamp the speed
  new_vel.x = new_vel.x < -maxSpeed ? -maxSpeed : new_vel.x;
  new_vel.y = new_vel.y < -maxSpeed ? -maxSpeed : new_vel.y;
  new_vel.z = new_vel.z < -maxSpeed ? -maxSpeed : new_vel.z;

  new_vel.x = new_vel.x > maxSpeed ? maxSpeed : new_vel.x;
  new_vel.y = new_vel.y > maxSpeed ? maxSpeed : new_vel.y;
  new_vel.z = new_vel.z > maxSpeed ? maxSpeed : new_vel.z;
  
  // Record the new velocity into vel2. Question: why NOT vel1?
  vel2[index] = new_vel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

/**
* Compute the cell index for each boid as well as the pointer (array index) to the actual boid data in pos and vel1/vel2.
* Parameters:
* - `gridResolution`: number of cells along each side of the grid, i.e. `gridSideCount`.
* 
*/
__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  // Label each boid with the index of its grid cell. Store in `gridIndices`, i.e. `dev_particleGridIndices`.
  glm::vec3 boid_pos = pos[index]; // Position of current boid

  // Calculate the cell index of the current boid
  int iX = floor((boid_pos.x - gridMin.x) * inverseCellWidth);
  int iY = floor((boid_pos.y - gridMin.y) * inverseCellWidth);
  int iZ = floor((boid_pos.z - gridMin.z) * inverseCellWidth);
  int index1D = gridIndex3Dto1D(iX, iY, iZ, gridResolution);

  // Store the cell index in `gridIndices`.
  gridIndices[index] = index1D;

  // Set up a parallel array of integer indices as pointers to the actual
  // boid data in pos and vel1/vel2. Store in `indices`, i.e. `dev_particleArrayIndices`.
  // Initially (before sorting), the pointer (array index) to the actual data in `dev_pos` and `dev_vel1/vel2` should be the same as the `index`.
  indices[index] = index;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
// This kernel sets all the elements in `intBuffer` to `value`. 
// We can use this to set the `value` to -1 to indicate an empty cell.
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

/**
* Unroll the loop for finding the start and end indices of each cell's data pointers in the array of boid indices
* Parameters:
* - `particleGridIndices`: the list of boid indices sorted by the cell index, i.e. `dev_particleGridIndices` after sorting
* - `gridCellStartIndices`: the starting index for each cell in `dev_particleGridIndices`
* - `gridCellEndIndices`: the ending index for each cell in `dev_particleGridIndices`
* 
*/
__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  // For each cell, get its starting and ending index in `dev_particleGridIndices`
  int curr_cell_idx = particleGridIndices[index];

  // If the first cell, we can't look at the previous one, just store the start index for it
  if (index == 0) { 
    gridCellStartIndices[curr_cell_idx] = index;
    return;
  }
  // Else, compare current cell with previous cell
  else {
    int prev_cell_idx = particleGridIndices[index - 1];

    // If they are different, we are at the border of two different cells
    // The start and end indices are inclusive.
    if (curr_cell_idx != prev_cell_idx) {
      gridCellEndIndices[prev_cell_idx] = index;
      gridCellStartIndices[curr_cell_idx] = index;
      return;
    }
    else {
      return;
    }
  }

  return;
}

__device__ glm::vec3 computeVelocityChangeRule1UniformGrid(const int N, const int iSelf, const int startIdx, 
                                                           const int* particleArrayIndices, 
                                                           const glm::vec3* pos, const glm::vec3* vel) {
  glm::vec3 perceived_center(0.0f, 0.0f, 0.0f);
  int num_neighbors = 0;

  for (int i = startIdx; i < startIdx + N; ++i) {
    int boid_idx = particleArrayIndices[i];

    if ((boid_idx != iSelf) && (glm::distance(pos[boid_idx], pos[iSelf]) < rule1Distance)) {
      perceived_center += pos[boid_idx];
      num_neighbors++;
    }
  }
  if (num_neighbors != 0) {
    perceived_center /= num_neighbors;
    return (perceived_center - pos[iSelf]) * rule1Scale;
  }
  else {
    return perceived_center;
  }
  
}

__device__ glm::vec3 computeVelocityChangeRule2UniformGrid(const int N, const int iSelf, const int startIdx,
  const int* particleArrayIndices,
  const glm::vec3* pos, const glm::vec3* vel) {

  glm::vec3 c(0.0f, 0.0f, 0.0f);

  for (int i = startIdx; i < startIdx + N; ++i) {
    int boid_idx = particleArrayIndices[i];

    if ((boid_idx != iSelf) && (glm::distance(pos[boid_idx], pos[iSelf]) < rule2Distance)) {
      c -= (pos[boid_idx] - pos[iSelf]);
    }
  }

  return c * rule2Scale;
}

__device__ glm::vec3 computeVelocityChangeRule3UniformGrid(const int N, const int iSelf, const int startIdx,
  const int* particleArrayIndices,
  const glm::vec3* pos, const glm::vec3* vel) {
  glm::vec3 perceived_velocity(0.0f, 0.0f, 0.0f);
  int num_neighbors = 0;

  for (int i = startIdx; i < startIdx + N; ++i) {
    int boid_idx = particleArrayIndices[i];

    if ((boid_idx != iSelf) && (glm::distance(pos[boid_idx], pos[iSelf]) < rule3Distance)) {
      perceived_velocity += vel[boid_idx];
      num_neighbors++;
    }
  }

  if (num_neighbors != 0) {
    perceived_velocity /= num_neighbors;
  }
  
  return perceived_velocity * rule3Scale;
}


/**
* Kernel that calculates the new velocity for a boid using the uniform grid data structure.
* Parameters:
* - `gridCellStartIndices`: i.e. `dev_gridCellStartIndices`
* - `gridCellEndIndices`: i.e. `dev_gridCellEndIndices`
* - `particleArrayIndices`: i.e. `dev_particleArrayIndices`
*/
__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  // Identify the grid cell that this particle is in
  glm::vec3 boid_pos = pos[index]; // Position of current boid
  glm::vec3 curr_vel = vel1[index];
  glm::vec3 overall_vel_change(0.0f, 0.0f, 0.0f);

  float max_rule_distance = cellWidth / 2.0f;
  float boid_x_min = boid_pos.x - max_rule_distance, boid_x_max = boid_x_min + cellWidth;
  float boid_y_min = boid_pos.y - max_rule_distance, boid_y_max = boid_y_min + cellWidth;
  float boid_z_min = boid_pos.z - max_rule_distance, boid_z_max = boid_z_min + cellWidth;

  // Calculate the cell index of the current boid
  int iX = floor((boid_pos.x - gridMin.x) * inverseCellWidth);
  int iY = floor((boid_pos.y - gridMin.y) * inverseCellWidth);
  int iZ = floor((boid_pos.z - gridMin.z) * inverseCellWidth);
  int index1D = gridIndex3Dto1D(iX, iY, iZ, gridResolution);

  float cell_x_min = gridMin.x + iX * cellWidth, cell_x_max = cell_x_min + cellWidth;
  float cell_y_min = gridMin.y + iY * cellWidth, cell_y_max = cell_y_min + cellWidth;
  float cell_z_min = gridMin.z + iZ * cellWidth, cell_z_max = cell_z_max + cellWidth;

  // Identify which cells may contain neighbors. This isn't always 8.
  // Loop through all the neighboring cells
  // Consider the indexing order in `gridIndex3Dto1D`:
  // x + y * gridResolution + z * gridResolution * gridResolution
  // If you use nested loops to walk over a "chunk" of the uniform grid, what order should you nest?
  // Goal is to keep access as contiguous as possible!
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        // Get the neighboring cell's cell index
        int neighbor_iX = iX + dx, neighbor_iY = iY + dy, neighbor_iZ = iZ + dz;
        int neighbor_index1D = gridIndex3Dto1D(neighbor_iX, neighbor_iY, neighbor_iZ, gridResolution);
        float neighbor_x_min = cell_x_min + dx * cellWidth, neighbor_x_max = cell_x_min + cellWidth;
        float neighbor_y_min = cell_y_min + dy * cellWidth, neighbor_y_max = cell_y_min + cellWidth;
        float neighbor_z_min = cell_z_min + dz * cellWidth, neighbor_z_max = cell_z_min + cellWidth;

        // Check if this neighboring cell may contain neighbor boids of current boid
        // Skip the current cell if it's not within the neighboring distance of the current boid
        if (neighbor_x_min >= boid_x_max || neighbor_x_max <= boid_x_min || 
            neighbor_y_min >= boid_y_max || neighbor_y_max <= boid_y_min || 
            neighbor_z_min >= boid_z_max || neighbor_z_max <= boid_z_min) {
          continue;
        }
        else {
          // Current neighbor cell contain neighbor boids of the current boid
          // Get all the boid data of the current neighboring cell
          // `particleArrayIndices[start_boid_idx, end_boid_idx]` contains all the boids data inside this cell
          int start_boid_idx = gridCellStartIndices[neighbor_index1D];
          int end_boid_idx = gridCellEndIndices[neighbor_index1D];
          int num_cell_boids = end_boid_idx - start_boid_idx + 1;

          // Skip the current cell if there are no boids inside
          if (start_boid_idx == -1 || end_boid_idx == -1) {
            continue;
          }

          // Compute the velocity change by all the boids inside the current cell
          glm::vec3 rule1_vel_change = computeVelocityChangeRule1UniformGrid(num_cell_boids, index, start_boid_idx, 
                                                                             particleArrayIndices, pos, vel1);

          glm::vec3 rule2_vel_change = computeVelocityChangeRule2UniformGrid(num_cell_boids, index, start_boid_idx,
                                                                             particleArrayIndices, pos, vel1);

          glm::vec3 rule3_vel_change = computeVelocityChangeRule3UniformGrid(num_cell_boids, index, start_boid_idx,
                                                                             particleArrayIndices, pos, vel1);
          overall_vel_change += rule1_vel_change + rule2_vel_change + rule3_vel_change;
        }
      }
    }
  }

  // Get the new velocity
  glm::vec3 new_vel = curr_vel + overall_vel_change;

  // Clamp the speed change before putting the new speed in vel2
  new_vel.x = new_vel.x < -maxSpeed ? -maxSpeed : new_vel.x;
  new_vel.y = new_vel.y < -maxSpeed ? -maxSpeed : new_vel.y;
  new_vel.z = new_vel.z < -maxSpeed ? -maxSpeed : new_vel.z;

  new_vel.x = new_vel.x > maxSpeed ? maxSpeed : new_vel.x;
  new_vel.y = new_vel.y > maxSpeed ? maxSpeed : new_vel.y;
  new_vel.z = new_vel.z > maxSpeed ? maxSpeed : new_vel.z;

  vel2[index] = new_vel;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
  int N = numObjects;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
  
  // Boids examine neighboring boids to determine new velocity `dev_vel2`
  kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (N, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  // Boids update position "dev_pos" based on the current velocity `dev_vel1` and change in time `dt`
  kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (N, dt, dev_pos, dev_vel1);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // Update the current velocity `dev_vel1` to the new velocity `dev_vel2`
  dev_vel1 = dev_vel2;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

  int num_boids = numObjects;
  dim3 boidBlocksPerGrid((num_boids + blockSize - 1) / blockSize);

  // Label each boid with its array index as well as its grid index.
  // Call this kernel `kernComputeIndices()`.
  kernComputeIndices << <boidBlocksPerGrid, blockSize >> > 
    (num_boids, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  // Sort based on key (cell index)
  // Key list: `dev_particleGridIndices`
  // Value list: `dev_particleArrayIndices`
  // Number of elements should be the number of boids, i.e. `num_boids`
  // Both has length `num_boids`, i.e. number of total boids.
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices); // Key list
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices); // Value list
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + num_boids, dev_thrust_particleArrayIndices);

  // Now `dev_particleGridIndices` and `dev_particleArrayIndices` are sorted based on the cell index order.
  // Unroll the loop for finding the start and end indices of each cell's data pointers in the array of boid indices `dev_particleArrayIndices`.
  int num_cells = gridCellCount;
  dim3 cellBlocksPerGrid((num_cells + blockSize - 1) / blockSize);

  // Invoke kernel `kernResetIntBuffer()` to initialize all the start and end indices of each cell to -1
  kernResetIntBuffer << <cellBlocksPerGrid, blockSize >> > (num_cells, dev_gridCellStartIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer for dev_gridCellStartIndices failed!");

  kernResetIntBuffer << <cellBlocksPerGrid, blockSize >> > (num_cells, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer for dev_gridCellEndIndices failed!");

  // Invoke kernel `kernIdentifyCellStartEnd()` to identify the start point of each cell in the gridIndices array.
  kernIdentifyCellStartEnd << <boidBlocksPerGrid, blockSize >> > (num_boids, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // Perform velocity updates using neighbor search
  // Invoke kernel `kernUpdateVelNeighborSearchScattered()`
  kernUpdateVelNeighborSearchScattered << <boidBlocksPerGrid, blockSize >> >
    (num_boids, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, 
      dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, 
      dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  // Update positions
  kernUpdatePos << <boidBlocksPerGrid, blockSize >> > (num_boids, dt, dev_pos, dev_vel1);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // Ping-pong buffers as needed
  dev_vel1 = dev_vel2;
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.
  std::cout << "Part 1.2 finished" << std::endl;
  std::cout << "Number of boids = " << numObjects << std::endl;

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
