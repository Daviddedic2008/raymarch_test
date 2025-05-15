#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__device__ inline float distance(const float3 p1, const float3 p2) {
	const float xdiff = p2.x - p1.x;
	const float ydiff = p2.y - p1.y;
	const float zdiff = p2.z - p1.z;

	return __fsqrt_rn(__fmaf_rn(xdiff, xdiff, __fmaf_rn(ydiff, ydiff, __fmul_rn(zdiff, zdiff))));

	
}

// base shape
struct Shape {
	float (*sdf)(float3 point);
	float3 position;

	float3 color;

	__host__ __device__ Shape(float (*sdf)(float3 point), float3 position, float3 color) : sdf(sdf), position(position), color(color){}
};

struct Sphere : Shape {
	float radius;

	__device__ inline float SphereSdf(float3 point) {
		return distance(point, position) - radius;
	}
};