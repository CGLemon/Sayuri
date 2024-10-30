#pragma once

#include <ctime>
#include <cstdint>
#include <vector>
#include <string>
#include <initializer_list>

// Split the pathname path into a vector.
std::vector<std::string> SplitPath(std::string path);

// Concatenate paths or filenames.
std::string ConcatPath(const std::string path_1, const std::string path_2);
std::string ConcatPath(std::initializer_list<std::string> list);

// Creates directory at a given path. Throws exception if cannot.
// Returns silently if already exists.
void TryCreateDirectory(const std::string& path);

// Returns true if the directory already exists.
bool IsDirectoryExist(const std::string& directory);

// Returns list of full paths of regular files in this directory.
// Silently returns empty vector on error.
std::vector<std::string> GetFileList(const std::string& directory);

// Returns list of full paths of directory in this directory.
// Silently returns empty vector on error.
std::vector<std::string> GetDirectoryList(const std::string& directory);

// Returns list of full paths of regular files in this directory tree.
// Silently returns empty vector on error.
std::vector<std::string> SearchFileTree(const std::string& directory, size_t *counter = nullptr);

// Returns size of a file, 0 if file doesn't exist or can't be read.
std::uint64_t GetFileSize(const std::string& filename);

// Returns modification time of a file, 0 if file doesn't exist or can't be read.
time_t GetFileTime(const std::string& filename);
