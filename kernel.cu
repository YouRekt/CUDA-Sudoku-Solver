#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>

#ifdef __INTELLISENSE__
#include "intelisense.h" // Fixes intellisense not recognizing atomicAdd();
#endif

#define CUDACheckError(cudaStatus) \
	if(cudaStatus != cudaSuccess) {	\
		std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "; \
		std::cerr << cudaGetErrorName(cudaStatus) << ": " << cudaGetErrorString(cudaStatus) << std::endl; \
	} \

#define THREADS_PER_BLOCK 1024 // Anything above just breaks
#define NUM_OF_BLOCKS 128

#define MAX_BOARDS (INT_MAX >> 4)
#define BFS_DEPTH 30
#define N 9
#define SQRT_N 3
#define BOARD_SIZE 81

// Host helper functions
void printDeviceArray(const char* name, int* d_array, int size) {
	int* h_array = new int[size];
	CUDACheckError(cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost));

	std::cout << "\n" << name << ":\n";
	for (int i = 0; i < size; i++) {
		if (i % BOARD_SIZE == 0 && size > 1) {
			std::cout << "#" << (i / BOARD_SIZE) + 1 << std::endl;
		}

		std::cout << h_array[i] << " ";

		// Add a newline after every row
		if ((i + 1) % N == 0) {
			std::cout << "\n";
		}

		// Add an extra newline after printing a full board (NxN grid)
		if ((i + 1) % BOARD_SIZE == 0) {
			std::cout << std::endl;
		}
	}
	delete[] h_array;
}

int countEmptySpaces(int* board) {
	int count = 0;
	for (int i = 0; i < BOARD_SIZE; i++) {
		if (board[i] == 0)
			count++;
	}
	return count;
}

void print(int* board) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			std::cout << board[i * N + j] << " ";
		std::cout << std::endl;
	}
}

// Device helper functions
__device__ void printBoard(int* board, int thread) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("Thread %d: [%d,%d] = %d\n", thread, i, j, board[N * i + j]);
		}
		printf("\n");
	}
	printf("\n");
}

__device__ void clearSeen(bool* seen) {
	memset(seen, false, sizeof(bool) * N);
}

__device__ void copyBoard(const int* src, int* dst) {
	for (int i = 0; i < BOARD_SIZE; i++) {
		dst[i] = src[i];
	}
}

__device__ void initializeEmptyCellIndexes(int* board, int* emptyCellIndexes, int* emptyCellCounts) {
	int index = 0;
	for (int i = 0; i < BOARD_SIZE; i++) {
		if (board[i] == 0) {
			emptyCellIndexes[index] = i;
			index++;
		}
	}
	*emptyCellCounts = index;
}

__device__ bool isValid(int* d_board, int changedIndex) { // DFS version
	int changedRow = changedIndex / N;
	int changedColumn = changedIndex % N;

	if (d_board[changedIndex] < 1 || d_board[changedIndex]>9)
		return false;

	bool seen[N];

	// Check column
	clearSeen(seen);
	for (int i = 0; i < N; i++) {
		int number = d_board[i * N + changedColumn] - 1;
		if (number >= 0) {
			if (seen[number])
				return false;
			seen[number] = true;
		}
	}

	// Check row
	clearSeen(seen);
	for (int i = 0; i < N; i++) {
		int number = d_board[changedRow * N + i] - 1;
		if (number >= 0) {
			if (seen[number])
				return false;
			seen[number] = true;
		}
	}

	// Check box
	clearSeen(seen);
	int b = (SQRT_N * (changedRow / SQRT_N)) + (changedColumn / SQRT_N);
	for (int i = 0; i < SQRT_N; i++) {
		for (int j = 0; j < SQRT_N; j++) {
			int b_row = b / SQRT_N, b_col = b % SQRT_N;
			int number = d_board[(SQRT_N * b_row + i) * N + SQRT_N * b_col + j] - 1;
			if (number >= 0) {
				if (seen[number])
					return false;
				seen[number] = true;
			}
		}
	}

	return true;
}

__device__ bool isValid(int* d_board, int changedIndex, int value) { // BFS version
	int changedRow = changedIndex / N;
	int changedColumn = changedIndex % N;

	bool seen[N];

	// Check column
	clearSeen(seen);
	for (int i = 0; i < N; i++) {
		int number = d_board[i * N + changedColumn] - 1;
		if (number >= 0) {
			if (seen[number])
				return false;
			seen[number] = true;
		}
	}
	if (seen[value - 1])
		return false;

	// Check row
	clearSeen(seen);
	for (int i = 0; i < N; i++) {
		int number = d_board[changedRow * N + i] - 1;
		if (number >= 0) {
			if (seen[number])
				return false;
			seen[number] = true;
		}
	}
	if (seen[value - 1])
		return false;

	// Check box
	clearSeen(seen);
	int b = SQRT_N * (changedRow / SQRT_N) + changedColumn / SQRT_N;
	for (int i = 0; i < SQRT_N; i++) {
		for (int j = 0; j < SQRT_N; j++) {
			int b_row = b / SQRT_N, b_col = b % SQRT_N;
			int number = d_board[(SQRT_N * b_row + i) * N + SQRT_N * b_col + j] - 1;
			if (number >= 0) {
				if (seen[number])
					return false;
				seen[number] = true;
			}
		}
	}
	if (seen[value - 1])
		return false;

	return true;
}

__global__ void DFS(int* BFSearchedBoards, const int boardsCount, int* emptyCellIndexes, int* emptyCellCounts, bool* finished, int* solvedBoard) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	while (!*finished && index < boardsCount) {
		int* currentBoard = BFSearchedBoards + index * BOARD_SIZE;
		int* currentEmptyCells = emptyCellIndexes + index * BOARD_SIZE;
		int emptyCellIndex = 0;

		while (emptyCellIndex >= 0 && emptyCellIndex < emptyCellCounts[index]) {
			currentBoard[currentEmptyCells[emptyCellIndex]]++;

			if (currentBoard[currentEmptyCells[emptyCellIndex]] > 9) {
				currentBoard[currentEmptyCells[emptyCellIndex]] = 0;
				emptyCellIndex--;  // Backtrack
			}
			else if (isValid(currentBoard, currentEmptyCells[emptyCellIndex])) {
				emptyCellIndex++;  // Move forward
			}
		}

		if (emptyCellIndex == emptyCellCounts[index]) {
			*finished = true;
			copyBoard(currentBoard, solvedBoard);
		}

		index += gridDim.x * blockDim.x; // This ensures we dont mess up the work of other threads
	}
}

__global__ void BFS(int* BFSearchedBoards, int startIndex, int cutoffIndex, int* nextBoardIndex, int* emptyCellIndexes, int* emptyCellCounts) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < cutoffIndex - startIndex) {
		bool found = false;
		int offsetIndex = index + startIndex;

		for (int i = BOARD_SIZE * offsetIndex; i < BOARD_SIZE * (offsetIndex + 1) && !found; i++) {
			if (BFSearchedBoards[i] == 0) {
				found = true;
				int normalizedIndex = i - BOARD_SIZE * offsetIndex;

				for (int val = 1; val <= N; val++) { // Test each value for the empty cell and if it's valid add it to the boards array
					if (isValid(BFSearchedBoards + BOARD_SIZE * offsetIndex, normalizedIndex, val)) {
						int boardIndex = atomicAdd(nextBoardIndex, 1); // This ensures contingency of the boards array
						copyBoard(BFSearchedBoards + BOARD_SIZE * offsetIndex, BFSearchedBoards + BOARD_SIZE * boardIndex);
						BFSearchedBoards[BOARD_SIZE * boardIndex + normalizedIndex] = val;

						initializeEmptyCellIndexes(BFSearchedBoards + BOARD_SIZE * boardIndex, emptyCellIndexes + BOARD_SIZE * boardIndex, emptyCellCounts + boardIndex); // Update the information about empty cells for DFS
					}
				}
			}
		}

		index += blockDim.x * gridDim.x; // This ensures we dont mess up the work of other threads
	}
}

int main() {
	int* h_board = new int[BOARD_SIZE];

	std::ifstream in("easy.txt");
	char c;
	int i = 0;
	while (in >> c && i < BOARD_SIZE) {
		h_board[i] = c - '0';
		i++;
	}
	in.close();

	int* d_BFSearchedBoards;
	int* d_emptyCellCounts;
	int* d_emptyCellIndexes;
	int* d_nextBoardIndex;

	CUDACheckError(cudaMalloc((void**)&d_BFSearchedBoards, MAX_BOARDS * sizeof(int)));
	CUDACheckError(cudaMalloc((void**)&d_emptyCellIndexes, MAX_BOARDS * sizeof(int)));
	CUDACheckError(cudaMalloc((void**)&d_emptyCellCounts, (MAX_BOARDS / BOARD_SIZE + 1) * sizeof(int)));
	CUDACheckError(cudaMalloc((void**)&d_nextBoardIndex, sizeof(int)));

	CUDACheckError(cudaMemset(d_BFSearchedBoards, 0, MAX_BOARDS * sizeof(int)));
	CUDACheckError(cudaMemset(d_emptyCellIndexes, 0, MAX_BOARDS * sizeof(int)));

	// I tried calling cudaMemset(d_nextBoardIndex,1,sizeof(int)) but it outputs 16843009. Reason: https://forums.developer.nvidia.com/t/can-we-use-memset-for-non-zero-initial-value/4032
	int initValue = 1; // Hence this is a simple workaround.
	CUDACheckError(cudaMemcpy(d_nextBoardIndex, &initValue, sizeof(int), cudaMemcpyHostToDevice));

	// Copy the initial board to BFS boards
	CUDACheckError(cudaMemcpy(d_BFSearchedBoards, h_board, BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice));

	// Timing variables
	cudaEvent_t start, stop;
	float milliseconds = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	int startIndex = 0;
	int cutoffIndex = 1;

	BFS << <NUM_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_BFSearchedBoards, startIndex, cutoffIndex, d_nextBoardIndex, d_emptyCellIndexes, d_emptyCellCounts);

	int depth;

	if (countEmptySpaces(h_board) > BOARD_SIZE - 2 * N) { // If the board is almost empty it is actually faster to skip the BFS
		depth = 0;
	}
	else {
		depth = BFS_DEPTH;
	}

	// Generate more boards with more iterations of BFS
	for (int i = 0; i < depth; i++) {
		startIndex = cutoffIndex;
		CUDACheckError(cudaMemcpy(&cutoffIndex, d_nextBoardIndex, sizeof(int), cudaMemcpyDeviceToHost));
		BFS << <NUM_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_BFSearchedBoards, startIndex, cutoffIndex, d_nextBoardIndex, d_emptyCellIndexes, d_emptyCellCounts);
	}
	startIndex = cutoffIndex;
	CUDACheckError(cudaMemcpy(&cutoffIndex, d_nextBoardIndex, sizeof(int), cudaMemcpyDeviceToHost));

	int boardsCount = cutoffIndex - startIndex;
	bool* d_finished;
	int* d_solvedBoard;
	int* h_solvedBoard = new int[BOARD_SIZE];
	bool* h_finished = new bool;

	CUDACheckError(cudaMalloc((void**)&d_finished, sizeof(bool)));
	CUDACheckError(cudaMalloc((void**)&d_solvedBoard, BOARD_SIZE * sizeof(int)));

	CUDACheckError(cudaMemset(d_finished, false, sizeof(bool)));

	DFS << <NUM_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_BFSearchedBoards + BOARD_SIZE * startIndex, boardsCount, d_emptyCellIndexes + BOARD_SIZE * startIndex, d_emptyCellCounts + startIndex, d_finished, d_solvedBoard);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

	CUDACheckError(cudaMemcpy(h_finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
	if (*h_finished) {
		CUDACheckError(cudaMemcpy(h_solvedBoard, d_solvedBoard, BOARD_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
		print(h_solvedBoard);
	}
	else
		printf("No solutuion found");

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	delete[] h_board;
	delete[] h_solvedBoard;
	delete h_finished;
	cudaFree(d_BFSearchedBoards);
	cudaFree(d_emptyCellCounts);
	cudaFree(d_emptyCellIndexes);
	cudaFree(d_nextBoardIndex);
	cudaFree(d_finished);
	cudaFree(d_solvedBoard);

	return 0;
}
