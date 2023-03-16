#include "utils/gzip_helper.h"

#include <stdexcept>
#include <memory>
#include <cstring>

#ifdef USE_ZLIB

#include "zlib.h"

void SaveGzip(std::string filename, std::string &buffer) {
    filename += ".gz";
    auto out = gzopen(filename.c_str(), "wb9");

    auto in_buff_size = buffer.size();
    auto in_buff = std::make_unique<char[]>(in_buff_size);
    std::memcpy(in_buff.get(), buffer.data(), in_buff_size);

    auto comp_size = gzwrite(out, in_buff.get(), in_buff_size);
    if (!comp_size) {
        throw "Error in gzip output";
    }
    gzclose(out);
}

#else

void SaveGzip(std::string /* filename */, std::string & /* buffer */ ) {
    throw "No gzip library";
}

#endif


bool IsGzipValid() {
#ifdef USE_ZLIB
    return true;
#else
    return false;
#endif
}
