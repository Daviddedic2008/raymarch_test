#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

template<typename T>

struct vectorCU {
	void* allocatedMemory;
	unsigned int allocatedElements;
	unsigned int currentIndex = 0;

	__host__ vectorCU(const int initialSize = 0) {
		cudaMalloc(&allocatedMemory, sizeof(T) * initialSize);
		allocatedElements = initialSize;
	}

	__device__ inline T& operator[](const int index) const {
		return ((T*)allocatedMemory)[index];
	}

	__host__ inline void append(const T& t) {
		if (allocatedElements > currentIndex) {
			((T*)allocatedMemory)[currentIndex] = t;
			currentIndex++;
			return;
		}

		allocatedElements++;

		void* tmpPointer;
		cudaMalloc(&tmpPointer, sizeof(T) * allocatedElements);

		cudaMemcpy(tmpPointer, allocatedMemory, sizeof(T) * currentIndex, cudaMemcpyDeviceToDevice);

		cudaFree(allocatedMemory);

		allocatedMemory = tmpPointer;

		((T*)allocatedMemory)[currentIndex] = t;
		currentIndex++;
	}

	__host__ inline void popNoDealloc() {
		if (currentIndex > 0) {
			currentIndex--;
		}
	}

	__host__ inline void pop() {
		void* tmpPointer;

		if (allocatedElements <= 0) { return; }

		currentIndex -= !(currentIndex < allocatedElements);
		allocatedElements--;

		cudaMalloc(&tmpPointer, sizeof(T) * allocatedElements);

		cudaMemcpy(tmpPointer, allocatedMemory, sizeof(T) * currentIndex);

		free(allocatedMemory);

		allocatedMemory = tmpPointer;
	}

	__host__ ~vectorCU() {
		cudaFree(allocatedMemory);
	}
};