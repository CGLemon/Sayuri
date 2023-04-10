#include <stdexcept>
#include <algorithm>
#include "utils/filesystem.h"

#ifdef WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#endif

std::string ConcatPath(const std::string path_1, const std::string path_2) {
#ifdef WIN32
    return path_1 + "\\" + path_2;
#else
    return path_1 + "/" + path_2;
#endif
}

std::string ConcatPath(std::initializer_list<std::string> list) {
    if (list.size() == 0) {
        return std::string{};
    }

    auto next = std::begin(list);
    auto path = *next;
    while (true) {
        next+=1;
        if (next == std::end(list)) {
            break;
        }
        path = ConcatPath(path, *next);
    }
    return path;
}

void TryCreateDirectory(const std::string& path) {
#ifdef WIN32
    if (CreateDirectoryA(path.c_str(), nullptr)) return;
    if (GetLastError() != ERROR_ALREADY_EXISTS) {
        throw std::runtime_error("Cannot create directory: " + path);
    }
#else
    if (mkdir(path.c_str(), 0777) < 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + path);
    }
#endif
}

bool IsDirectoryExist(const std::string& directory) {
#ifdef WIN32
    WIN32_FIND_DATAA dir;
    const auto handle = FindFirstFileA((directory + "\\*").c_str(), &dir);
    if (handle == INVALID_HANDLE_VALUE) return false;
#else
    DIR* dir = opendir(directory.c_str());
    if (!dir) return false;
#endif
    return true;
}

std::vector<std::string> GetFileList(const std::string& directory) {
    std::vector<std::string> result;
#ifdef WIN32
    WIN32_FIND_DATAA dir;
    const auto handle = FindFirstFileA((directory + "\\*").c_str(), &dir);
    if (handle == INVALID_HANDLE_VALUE) return result;
    do {
        if ((dir.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
            result.emplace_back(dir.cFileName);
        }
    } while (FindNextFile(handle, &dir) != 0);
    FindClose(handle);
#else
    DIR* dir = opendir(directory.c_str());
    if (!dir) return result;
    while (auto* entry = readdir(dir)) {
        bool exists = false;
        if (entry->d_type == DT_REG) {
            exists = true;
        } else if (entry->d_type == DT_LNK) {
            const std::string filename = directory + "/" + entry->d_name;
            struct stat s;
            exists = stat(filename.c_str(), &s) == 0 && (s.st_mode & S_IFMT) == S_IFREG;
        }
        if (exists) result.push_back(entry->d_name);
    }
    closedir(dir);
#endif
    return result;
}

std::vector<std::string> GetDirectoryList(const std::string& directory) {
    std::vector<std::string> result;
#ifdef WIN32
    WIN32_FIND_DATAA dir;
    const auto handle = FindFirstFileA((directory + "\\*").c_str(), &dir);
    if (handle == INVALID_HANDLE_VALUE) return result;
    do {
        if ((dir.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
            result.emplace_back(dir.cFileName);
        }
    } while (FindNextFile(handle, &dir) != 0);
    FindClose(handle);
#else
    DIR* dir = opendir(directory.c_str());
    if (!dir) return result;
    while (auto* entry = readdir(dir)) {
        bool exists = false;
        if (entry->d_type == DT_DIR) {
            exists = true;
        }
        if (exists) result.push_back(entry->d_name);
    }
    closedir(dir);
#endif
    result.erase(std::remove_if(std::begin(result),
                                std::end(result),
                                [](auto in) -> bool {
                                    return in == "." ||  in == "..";
                                }),
                 std::end(result));
    return result;
}

std::vector<std::string> SearchFileTree(const std::string& directory, size_t *counter) {
    size_t root_counter = 0;
    if (counter == nullptr) {
        counter = &root_counter;
    }
    auto result = GetFileList(directory);
    *counter += result.size();

    if (*counter > 9999) {
        return result;
    }

    auto dir_list = GetDirectoryList(directory);
    for (const auto &dir : dir_list) {
        auto fullname = std::string{};
#ifdef WIN32
        fullname = directory + "\\" + dir;
#else
        fullname = directory + "/" + dir;
#endif

        auto sub_result = SearchFileTree(fullname, counter);
        for (auto &sub : sub_result) {
#ifdef WIN32
            sub = dir + "\\" + sub;
#else
            sub = dir + "/" + sub;
#endif
        }

        result.insert(std::end(result),
                          std::begin(sub_result),
                          std::end(sub_result));
    }
    return result;
}

std::uint64_t GetFileSize(const std::string& filename) {
#ifdef WIN32
    WIN32_FILE_ATTRIBUTE_DATA s;
    if (!GetFileAttributesExA(filename.c_str(), GetFileExInfoStandard, &s)) {
        return 0;
    }
    return (static_cast<std::uint64_t>(s.nFileSizeHigh) << 32) + s.nFileSizeLow;
#else
    struct stat s;
    if (stat(filename.c_str(), &s) < 0) {
        return 0;
    }
    return s.st_size;
#endif
}

time_t GetFileTime(const std::string& filename) {
#ifdef WIN32
    WIN32_FILE_ATTRIBUTE_DATA s;
    if (!GetFileAttributesExA(filename.c_str(), GetFileExInfoStandard, &s)) {
        return 0;
    }
    return (static_cast<std::uint64_t>(s.ftLastWriteTime.dwHighDateTime) << 32) +
               s.ftLastWriteTime.dwLowDateTime;
#else
    struct stat s;
    if (stat(filename.c_str(), &s) < 0) {
        return 0;
    }
#ifdef __APPLE__
    return s.st_mtimespec.tv_sec;
#else
    return s.st_mtim.tv_sec;
#endif
#endif
}
