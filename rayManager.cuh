#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "shapes.cuh"

#include "vectorCU.cuh"

#include "rays.cuh"

__global__ void initRays(Ray* r, const int length, const int height, const float fov) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= length * height) { return; }

	const int x = idx % length;
	const int y = idx / length;


	r[idx].origin = { (float)x, (float)y, 0 };
	r[idx].direction = { (float)x * fov, (float)y * fov, 1 };
	r[idx].closestDistance = 1e100;
}

__global__ void marchRaysUntilDone(Ray* r, const int numRays, const int stepLimit, const VectorCU<Shape> shapes) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= numRays) { return; }

	r[idx].marchUntilDone(stepLimit, shapes);
}

struct rayGrid {
	int length, height;

	float fov;

	Ray* rays;

	__host__ rayGrid(const int length, const int height, const float fov) : length(length), height(height), fov(fov) {
		cudaMalloc(&rays, sizeof(Ray) * length * height);
		initRays << <512, (length * height / 512) + 1 >> > (rays, length, height, fov);
	}

	__host__ void resetRayGrid() {
		initRays << <512, (length * height / 512) + 1 >> > (rays, length, height, fov);
	}

	__host__ void raymarch(const VectorCU<Shape> shapes, const int stepLimit) {
		marchRaysUntilDone << <512, (length * height / 512) + 1 >> > (rays, length * height, stepLimit, shapes);
	}

	__host__ ~rayGrid() {
		cudaFree(rays);
	}
};