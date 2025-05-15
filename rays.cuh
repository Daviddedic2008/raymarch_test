#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "shapes.cuh"

__device__ inline float smoothMin(const float d1, const float d2, const float k) {
	float h = fmaxf(k - fabsf(d1 - d2), 0.0f) / k;
	return fminf(d1, d2) - h * h * h * k * (1.0f / 3.0f);
}

struct Ray {
	float3 origin, direction;

	float closestDistance = 1e100;

	__host__ __device__ Ray(const float3& origin, const float3& direction) : origin(origin), direction(direction){}

	inline __device__ void resetRay(const float3& o) {
		origin = o;
		closestDistance = 1e100;
	}

	inline __device__ void checkDistance(const Shape& s) {
		const float td = s.sdf(origin);
		closestDistance = (td < closestDistance) * td + (td >= closestDistance) * closestDistance;
	}

	inline __device__ void lerpCheckDistance(const Shape& s1, const Shape& s2, const float blendConstant) {
		const float td = smoothMin(s1.sdf(origin), s2.sdf(origin), blendConstant);
		closestDistance = (td < closestDistance) * td + (td >= closestDistance) * closestDistance;
	}

	inline __device__ bool march() {
		if (closestDistance < 1e-7) {
			return true;
		}

		origin.x += direction.x * closestDistance;
		origin.y += direction.y * closestDistance;
		origin.z += direction.z * closestDistance;

		return false;
	}

	inline void marchUntilDone(const int stepLimit, const ) {
		for (int s = 0; s < stepLimit; s++) {

		}
	}
};