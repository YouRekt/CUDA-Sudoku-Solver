#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>

#define N 9
#define SQRT_N 3
#define BOARD_SIZE 81
#define BFS_DEPTH 30

using Board = std::vector<int>;

// Function to print the Sudoku board
void printBoard(const Board& board) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << board[i * N + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// Check if placing a number at a specific position is valid
bool isValid(const Board& board, int index, int value) {
	int row = index / N;
	int col = index % N;

	// Check the row
	for (int i = 0; i < N; i++) {
		if (board[row * N + i] == value) return false;
	}

	// Check the column
	for (int i = 0; i < N; i++) {
		if (board[i * N + col] == value) return false;
	}

	// Check the subgrid
	int startRow = (row / SQRT_N) * SQRT_N;
	int startCol = (col / SQRT_N) * SQRT_N;
	for (int i = 0; i < SQRT_N; i++) {
		for (int j = 0; j < SQRT_N; j++) {
			if (board[(startRow + i) * N + (startCol + j)] == value) return false;
		}
	}

	return true;
}

// BFS function to populate the next generation of boards
void BFS(std::vector<Board>& currentBoards, std::vector<Board>& nextBoards) {
	for (const auto& board : currentBoards) {
		bool foundEmpty = false;

		for (int i = 0; i < BOARD_SIZE; i++) {
			if (board[i] == 0) {
				foundEmpty = true;

				// Try all possible values for the empty cell
				for (int value = 1; value <= N; value++) {
					if (isValid(board, i, value)) {
						Board newBoard = board;
						newBoard[i] = value;
						nextBoards.push_back(newBoard);
					}
				}

				break; // Only process the first empty cell
			}
		}

		// If no empty cells were found, add the board to the next generation
		if (!foundEmpty) {
			nextBoards.push_back(board);
		}
	}
}

// Solve a single board using DFS
bool DFS(Board& board) {
	for (int i = 0; i < BOARD_SIZE; i++) {
		if (board[i] == 0) {
			for (int value = 1; value <= N; value++) {
				if (isValid(board, i, value)) {
					board[i] = value;
					if (DFS(board)) return true;
					board[i] = 0; // Backtrack
				}
			}
			return false; // No valid value found
		}
	}
	return true; // Solved
}

int main() {
	// Read the Sudoku puzzle from a file
	Board initialBoard(BOARD_SIZE);
	std::ifstream inFile("hard.txt");
	if (!inFile) {
		std::cerr << "Failed to open the input file." << std::endl;
		return 1;
	}

	char c;
	for (int i = 0; i < BOARD_SIZE && inFile >> c; i++) {
		initialBoard[i] = c - '0';
	}
	inFile.close();

	std::cout << "Initial Board:" << std::endl;
	printBoard(initialBoard);

	// Start the total runtime measurement
	auto start = std::chrono::high_resolution_clock::now();

	// BFS Phase (to generate boards)
	std::vector<Board> currentBoards = { initialBoard };
	std::vector<Board> nextBoards;

	for (int depth = 0; depth < BFS_DEPTH; depth++) {
		nextBoards.clear();
		BFS(currentBoards, nextBoards);
		currentBoards = std::move(nextBoards);
	}

	// DFS Phase (to solve the board)
	bool solved = false;
	for (auto& board : currentBoards) {
		if (DFS(board)) {
			printBoard(board);
			solved = true;
			break;
		}
	}

	// End the total runtime measurement
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::milliseconds miliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "Execution time: " << miliseconds.count() << " ms" << std::endl;

	if (!solved) {
		std::cout << "No solution found." << std::endl;
	}

	return 0;
}
