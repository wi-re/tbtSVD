//#define NO_CUDA_SUPPORT
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <random>
#include <atomic>
#include <mutex>
#ifndef NO_CUDA_SUPPORT
#include <cuda.h>
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#define __forceinline__
#endif
#ifdef WIN32
#define NOMINMAX
#include <Windows.h>
#endif

#include "SVD.h"
// Function used to calculate the error by reconstructing A from U * S * V^T. Caclulates the absolute, maximum and off diagonal error as a tuple.
__host__ __device__ __forceinline__ auto error(const SVD::Mat3x3& A, const SVD::Mat3x3& U, const SVD::Mat3x3& S, const SVD::Mat3x3& V) {
	auto B = U * S * V.transpose();
	float c00 = B.m_00 - A.m_00; c00 *= c00;
	float c01 = B.m_01 - A.m_01; c01 *= c01;
	float c02 = B.m_02 - A.m_02; c02 *= c02;
	float c10 = B.m_10 - A.m_10; c10 *= c10;
	float c11 = B.m_11 - A.m_11; c11 *= c11;
	float c12 = B.m_12 - A.m_12; c12 *= c12;
	float c20 = B.m_20 - A.m_20; c20 *= c20;
	float c21 = B.m_21 - A.m_21; c21 *= c21;
	float c22 = B.m_22 - A.m_22; c22 *= c22;

	float diagonal_error = std::abs(S.m_01) + std::abs(S.m_02) + std::abs(S.m_10) + std::abs(S.m_12) + std::abs(S.m_20) + std::abs(S.m_21);
	float abs_error = c00 + c01 + c02 + c10 + c11 + c12 + c20 + c21 + c22;
	float max_abs_error = std::max(c00, std::max(c01, std::max(c02, std::max(c10, std::max(c11, std::max(c12, std::max(c20, std::max(c21, c22))))))));
	return std::make_tuple(abs_error, max_abs_error, diagonal_error, A.det(), B.det());
}
// Wrapping function for testing purposes, calls the SVD on input Matrix A and stores the errors in entry i of the given arrays
__device__ __host__ auto testSVD(int32_t i, float* abs, float* max, float* diagonal, float* matrices, int32_t tests) {
	auto A = SVD::Mat3x3::fromPtr(matrices, i, tests);
	auto usv = SVD::svd(A);
	auto err = error(A, usv.U, usv.S, usv.V);
	abs[i] = std::get<0>(err);
	max[i] = std::get<1>(err);
	diagonal[i] = std::get<2>(err);	
}
#ifndef NO_CUDA_SUPPORT
// Wrapper to test code on GPU. Executes on a single thread 
__global__ void gtestSVD(float* abs, float* max, float* diagonal, float* matrices, int32_t tests) {
	int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= tests) return;
	testSVD(i, abs, max, diagonal, matrices, tests);
}
#endif
namespace testUtility {
	// Returns a random Matrix built using std::uniform_real_distribution with norm 1
	auto randomMatrix() {
		static std::random_device rd;
		static std::mt19937_64 gen(rd());
		static std::uniform_real_distribution<float> dis(-1.f, 1.f);
		SVD::Mat3x3 M(
			dis(gen), dis(gen), dis(gen),
			dis(gen), dis(gen), dis(gen),
			dis(gen), dis(gen), dis(gen)
		);
		float norm = sqrtf(
			M.m_00 * M.m_00 + M.m_10 * M.m_10 + M.m_20 * M.m_20
			+ M.m_01 * M.m_01 + M.m_11 * M.m_11 + M.m_21 * M.m_21
			+ M.m_02 * M.m_02 + M.m_12 * M.m_12 + M.m_22 * M.m_22
		);

		return M * (1.f / norm);
	}
	unsigned int getIntLength(int x) {
		if (x == 0)
			return 1;
		else return std::log10(std::abs(x)) + 1;
	}
	template<class T>
	constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
		return v < lo ? lo : hi < v ? hi : v;
	}
	auto vectorStats(const std::vector<float>& data, int32_t buckets = 120, int32_t height = 64) {
		auto sam = data;
		std::nth_element(sam.begin(), sam.begin() + sam.size() / 2, sam.end(),
			[](auto lhs, auto rhs) { return lhs < rhs; });
		auto median = sam[sam.size() / 2];
		auto min = FLT_MAX;
		auto max = -FLT_MAX;
		auto average = 0.f;
		auto sum = 0.f;
		float ctr = (float) data.size();
		for (auto s : data) {
			auto val = s;
			min = std::min(min, val);
			max = std::max(max, val);
			sum += val;
		}
		average = sum / ctr;
		auto stddev = 0.f;
		for (auto s : data) {
			auto val = s;
			auto diff = (val - average) * (val - average);
			stddev += diff;
		}
		stddev /= ctr - 1;
		stddev = sqrt(stddev);
#define COUT_FLOAT_HEX(x) std::scientific /*<< std::setprecision(6)*/ << x << " [" << std::hexfloat << x << "]" << std::defaultfloat
		std::cout << "Data entries: " << ((int32_t)ctr) <<
			", avg:" << COUT_FLOAT_HEX(average) <<
			", median: " << COUT_FLOAT_HEX(median) <<
			", min: " << COUT_FLOAT_HEX(min) <<
			", max: " << COUT_FLOAT_HEX(max) <<
			", stddev: " << COUT_FLOAT_HEX(stddev) <<
			std::endl;
		std::cout << std::defaultfloat;
		if (height == 0)
			return;
		std::vector<int32_t> histogram;
		histogram.resize(buckets);
		for (auto s : data) {
			auto bucket = clamp(static_cast<int32_t>((s - min) / (max - min) * (float)buckets), 0, buckets - 1);
			histogram[bucket]++;
		}
		int32_t minBar = INT_MAX;
		int32_t maxBar = INT_MIN;
		for (auto b : histogram) {
			maxBar = std::max(b, maxBar);
			minBar = std::min(b, minBar);
		}
		for (int32_t i = height - 1; i >= 0; --i) {
			float scaledHeight = (float)i * 1.f / ((float)height);
			int32_t data = scaledHeight * (maxBar - minBar) + minBar;
			std::cout << std::left << std::setw(8) << data << " ";
			for (int32_t j = 0; j < buckets; ++j) {
				auto entries = histogram[j];
				auto scaledBar = ((float)entries - (float)minBar) / ((float)maxBar - (float)minBar);
				std::cout << (scaledBar > scaledHeight ? "x" : " ");
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	void progressBar(int32_t step, int32_t numSteps) {
		static std::mutex m;
		std::lock_guard<std::mutex> lg(m);
		static auto startOverall = std::chrono::high_resolution_clock::now();
		if (numSteps == 0) {
			startOverall = std::chrono::high_resolution_clock::now();
			return;
		}
		auto now = std::chrono::high_resolution_clock::now();
		int barWidth = 120;
		float progress = ((float)step) / ((float)numSteps);
		std::cout << std::setw(getIntLength(numSteps) + 1) << step << "/" << numSteps << " ";
		std::cout << "[";
		int pos = barWidth * progress;
		for (int i = 0; i < barWidth; ++i) {
			if (i < pos) std::cout << "=";
			else if (i == pos) std::cout << ">";
			else std::cout << " ";
		}
		std::cout << "] " << std::setw(3) << int(progress * 100.0) << "% ";
		if (step != 0) {
			auto duration = now - startOverall;
			auto estimated = duration / progress - duration;
			double s = (double)std::chrono::duration_cast<std::chrono::microseconds>(duration).count() * 0.001 *0.001;
			std::cout << std::fixed << std::setw(14) << std::setprecision(2) << ((double)step) / s << " elems/sec ";
			auto printTime = [](auto tp) {
				std::stringstream sstream;
				auto h = std::chrono::duration_cast<std::chrono::hours>(tp).count();
				auto m = std::chrono::duration_cast<std::chrono::minutes>(tp).count() - h * 60;
				auto s = std::chrono::duration_cast<std::chrono::seconds>(tp).count() - h * 3600 - m * 60;
				auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp).count() - s * 1000 - h * 3600 * 1000 - m * 60 * 1000;
				sstream << std::setw(2) << h << "h " << std::setw(2) << m << "m " << std::setw(2) << s << "s " << std::setw(4) << ms << "ms";
				return sstream.str();
			};
			std::cout << " Elapsed: " << printTime(duration) << " ETA: " << printTime(estimated);
			std::cout << "     ";
		}
		std::cout << "\r";
		std::cout.flush();
	}
	void printVectors(std::vector<float> vec, float* ptr, std::string descr, int32_t tests) {
		std::ios cout_state(nullptr);
		cout_state.copyfmt(std::cout);
		std::cout << descr << ": " << std::endl;
#ifndef NO_CUDA_SUPPORT
#ifndef NO_CPU_SUPPORT
		vectorStats(vec, 200, 0);
#endif
		cudaMemcpy(vec.data(), ptr, sizeof(float) * tests, cudaMemcpyDeviceToHost);
#endif
		vectorStats(vec, 200, 16);
		std::cout.copyfmt(cout_state);
	}
}
int32_t main() {
	std::ios cout_state(nullptr);
#ifdef WIN32
	cout_state.copyfmt(std::cout);
	HWND console = GetConsoleWindow();
	RECT r;
	GetWindowRect(console, &r);
	MoveWindow(console, 0, 0, 1920, 1200, TRUE);
#endif
	constexpr int32_t tests = 1024 * 1024 * 16;
	std::vector<float> abs_errors(tests), max_errors(tests), diag_errors(tests);

	float* host_a = abs_errors.data(), *host_m = max_errors.data(), *host_d = diag_errors.data();
	float* dev_a = nullptr, *dev_m = nullptr, *dev_d = nullptr;
	
#ifndef NO_CUDA_SUPPORT
	cudaMalloc(&dev_a, sizeof(float) * tests);
	cudaMalloc(&dev_m, sizeof(float) * tests);
	cudaMalloc(&dev_d, sizeof(float) * tests);
#endif
	float* host_matrices = (float*)malloc(sizeof(float) * 9 * tests);
	float* dev_matrices = nullptr;
	std::cout << "Generating random test data" << std::endl;
	std::atomic<int32_t> ctr(0);
#pragma omp parallel for
	for (int32_t i = 0; i < tests; ++i) {
		auto M = testUtility::randomMatrix();
		M.toPtr(host_matrices, i, tests);
		auto c = ctr++;
		if (c % std::max(32, (tests / 100)) == 0)
			testUtility::progressBar(c, tests);
	}
#ifndef NO_CUDA_SUPPORT
	cudaMalloc(&dev_matrices, sizeof(float) * 9 * tests);
	cudaMemcpy(dev_matrices, host_matrices, sizeof(float) * 9 * tests, cudaMemcpyHostToDevice);
#endif
	std::cout << std::endl;
#ifndef NO_CPU_SUPPORT
	std::cout << "Running test on CPU" << std::endl;
	testUtility::progressBar(0, 0);
	ctr = 0;
#pragma omp parallel for
	for (int32_t i = 0; i < tests; ++i) {
		testSVD(i, host_a, host_m, host_d, host_matrices, tests);
		auto c = ctr++;
		if (c % std::max(32, (tests / 100)) == 0)
			testUtility::progressBar(c, tests);
	}
	testUtility::progressBar(tests, tests);
	std::cout << std::endl;
#endif
#ifndef NO_CUDA_SUPPORT
	std::cout << "Running test on GPU" << std::endl;
	auto blockSize = 64;
	auto blocks = tests / blockSize + (tests % blockSize != 0 ? 1 : 0);
	testUtility::progressBar(0, 0);
	cudaDeviceSynchronize();
	gtestSVD <<<blocks, blockSize >>> (dev_a, dev_m, dev_d, dev_matrices, tests);
	cudaDeviceSynchronize();
	testUtility::progressBar(tests, tests);
#endif

	std::cout.copyfmt(cout_state);
	std::cout << std::endl;

	testUtility::printVectors(abs_errors, dev_a, "Absolute Errors", tests);
	testUtility::printVectors(max_errors, dev_m, "Maximum Errors", tests);
	testUtility::printVectors(diag_errors, dev_d, "Diagonal Errors", tests);

#ifndef NO_CUDA_SUPPORT
	cudaFree(dev_a);
	cudaFree(dev_m);
	cudaFree(dev_d);
	cudaFree(dev_matrices);
#endif
	free(host_matrices);
	getchar();
}