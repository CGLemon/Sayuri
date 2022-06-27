#ifdef USE_CUDA
#include "neural/cuda/cuda_forward_pipe.h"
#endif

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_OPENBLAS
#ifndef __APPLE__
#include <cblas.h>
#endif
#endif

#include "config.h"
#include "neural/blas/blas_forward_pipe.h"
#include "game/symmetry.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/encoder.h"
#include "utils/log.h"
#include "utils/random.h"
#include "utils/format.h"

#include <random>
#include <sstream>
#include <iomanip>

void Network::Initialize(const std::string &weightsfile) {
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);

    LOGGING << "BLAS Core:" << ' ' << openblas_get_corename() << std::endl;
#endif
#ifdef USE_MKL
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);

    LOGGING << "BLAS core: MKL" << ' ' << Version.Processor << std::endl;
#endif
#endif

#ifdef USE_EIGEN
    LOGGING << "BLAS Core: Eigen" << ' '
                << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.' << EIGEN_MINOR_VERSION << ' '
                << "library." << std::endl;
#endif

#ifdef USE_CUDA
    using backend = CudaForwardPipe;
#else
    using backend = BlasForwardPipe;
#endif

    pipe_ = std::make_unique<backend>();
    auto dnn_weights = std::make_shared<DNNWeights>();

    DNNLoder::Get().FromFile(dnn_weights, weightsfile);

    if (!dnn_weights->loaded) {
        dnn_weights.reset();
        dnn_weights = nullptr;
    }

    pipe_->Initialize(dnn_weights);
    SetCacheSize(GetOption<int>("playouts"));
}

void Network::SetCacheSize(int playouts) {
    playouts = std::max(128, playouts);
    nn_cache_.SetCapacity(playouts * GetOption<int>("cache_buffer_factor"));
}

void Network::ClearCache() {
    nn_cache_.Clear();
}

Network::Result Network::DummyForward(const Network::Inputs& inputs) const {
    Network::Result result{};

    auto rng = Random<kXoroShiro128Plus>::Get();
    auto dis = std::uniform_real_distribution<float>(0, 1);

    const auto boardsize = inputs.board_size;
    const auto num_intersections = boardsize * boardsize;

    result.board_size = boardsize;
    for (int idx = 0; idx < 3; ++idx) {
        result.wdl[idx] = dis(rng);
    }

    for (int idx = 0; idx < num_intersections; ++idx) {
        result.probabilities[idx] = dis(rng);
    }
    result.pass_probability = dis(rng);

    return result;
}

Network::Result
Network::GetOutputInternal(const GameState &state, const int symmetry) {
    Network::Result out_result;
    Network::Result result;

    auto inputs = Encoder::Get().GetInputs(state, symmetry);

    if (pipe_->Valid()) {
        result = pipe_->Forward(inputs);
    } else {
        result = DummyForward(inputs);
    }
    out_result = result;

    const auto boardsize = inputs.board_size;
    const auto num_intersections = boardsize * boardsize;

    auto probabilities_buffer = std::vector<float>(num_intersections);
    auto ownership_buffer = std::vector<float>(num_intersections);

    for (int idx = 0; idx < num_intersections; ++idx) {
        probabilities_buffer[idx] = result.probabilities[idx];
        ownership_buffer[idx] = result.ownership[idx];
    }

    // Probabilities, ownership
    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto symm_index = Symmetry::Get().TransformIndex(boardsize, symmetry, idx);
        out_result.probabilities[symm_index] = probabilities_buffer[idx];
        out_result.ownership[symm_index] = std::tanh(ownership_buffer[idx]);
    }

    // Final score
    out_result.final_score = 20 * result.final_score;

    // winrate
    auto wdl_buffer = std::vector<float>(3);
    wdl_buffer[0] = result.wdl[0];
    wdl_buffer[1] = result.wdl[1];
    wdl_buffer[2] = result.wdl[2];
    wdl_buffer = Softmax(wdl_buffer, 1);

    out_result.wdl[0] = wdl_buffer[0];
    out_result.wdl[1] = wdl_buffer[1];
    out_result.wdl[2] = wdl_buffer[2];
    out_result.wdl_winrate = wdl_buffer[0] - wdl_buffer[2];
    out_result.wdl_winrate = (out_result.wdl_winrate + 1.f) / 2;
    out_result.stm_winrate = (std::tanh(out_result.stm_winrate) + 1.f) / 2;

    return out_result;
}

bool Network::ProbeCache(const GameState &state,
                         Network::Result &result) {
    if (LookupCache(nn_cache_, state.GetHash(), result)) {
        if (result.board_size == state.GetBoardSize()) {
            return true;
        }
    }

    if (state.GetBoardSize() >= state.GetMoveNumber() &&
            GetOption<bool>("early_symm_cache")) {
        for (int symm = Symmetry::kIdentitySymmetry+1; symm < Symmetry::kNumSymmetris; ++symm) {
            if (LookupCache(nn_cache_, state.ComputeSymmetryHash(symm), result)) {
                if (result.board_size != state.GetBoardSize()) {
                    break;
                }
                const int boardsize = result.board_size;
                const int num_intersections = state.GetNumIntersections();
                    
                auto probabilities_buffer = std::vector<float>(num_intersections);
                auto ownership_buffer = std::vector<float>(num_intersections);

                // copy result to buffer
                std::copy(std::begin(result.probabilities),
                              std::begin(result.probabilities) + num_intersections,
                              std::begin(probabilities_buffer));
                std::copy(std::begin(result.ownership),
                              std::begin(result.ownership) + num_intersections,
                              std::begin(ownership_buffer));

                // transfer them
                for (int idx = 0; idx < num_intersections; ++idx) {
                    const auto symm_index = Symmetry::Get().TransformIndex(boardsize, symm, idx);
                    result.probabilities[idx] = probabilities_buffer[symm_index];
                    result.ownership[idx] = ownership_buffer[symm_index];
                }
                return true;
            }
        }         
    }
    return false;
}

Network::Result
Network::GetOutput(const GameState &state,
                   const Ensemble ensemble,
                   const float temperature,
                   int symmetry,
                   const bool read_cache,
                   const bool write_cache) {
    Result result;
    if (ensemble == kNone) {
        symmetry = Symmetry::kIdentitySymmetry;
    } else if (ensemble == kDirect) {
        assert(symmetry >= 0 && symmetry < Symmetry::kNumSymmetris);
    } else if (ensemble == kRandom) {
        symmetry = Random<kXoroShiro128Plus>::Get().RandFix<Symmetry::kNumSymmetris>();
    }

    bool probed = false;

    // Get result from cache, if it is in the cache memory.
    if (read_cache) {
        if (ProbeCache(state, result)) {
            probed = true;
        }
    }

    if (!probed) {
        result = GetOutputInternal(state, symmetry);

        // Write result to cache, if it is not in the cache memory.
        if (write_cache) {
            nn_cache_.Insert(state.GetHash(), result);
        }
    }

    ActivatePolicy(result, temperature);

    return result;
}

std::string Network::GetOutputString(const GameState &state,
                                     const Ensemble ensemble,
                                     int symmetry) {
    const auto result = GetOutput(state, ensemble, 1.f, symmetry, false, false);
    const auto bsize = result.board_size;

    auto out = std::ostringstream{};
 
    out << "stm winrate: " << result.stm_winrate << std::endl;
    out << "wdl winrate: " << result.wdl_winrate << std::endl;
    out << "win probability: " << result.wdl[0] << std::endl;
    out << "draw probability: " << result.wdl[1] << std::endl;
    out << "loss probability: " << result.wdl[2] << std::endl;
    out << "final score: " << result.final_score << std::endl;

    out << "probabilities: " << std::endl;
    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            out << Format("%10.6f", result.probabilities[state.GetIndex(x,y)]);
        }
        out << std::endl;
    }
    out << Format("pass probabilities: %.6f\n", result.pass_probability);

    out << "ownership: " << std::endl;
    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            out << Format("%10.6f", result.ownership[state.GetIndex(x,y)]);
        }
        out << std::endl;
    }
    out << std::endl;

    return out.str();
}

std::vector<float> Network::Softmax(std::vector<float> &input, const float temperature) const {
    auto output = std::vector<float>{};
    output.reserve(input.size());

    const auto alpha = *std::max_element(std::begin(input), std::end(input));
    auto denom = 0.0f;

    for (const auto in_val : input) {
        auto val = std::exp((in_val - alpha) / temperature);
        denom += val;
        output.emplace_back(val);
    }

    for (auto &out : output) {
        out /= denom;
    }

    return output;
}

void Network::ActivatePolicy(Result &result, const float temperature) const {
    const auto boardsize = result.board_size;
    const auto num_intersections = boardsize * boardsize;

    auto probabilities_buffer = std::vector<float>(num_intersections+1);

    for (int idx = 0; idx < num_intersections; ++idx) {
        probabilities_buffer[idx] = result.probabilities[idx];
    }
    probabilities_buffer[num_intersections] = result.pass_probability;
    probabilities_buffer = Softmax(probabilities_buffer, temperature);

    for (int idx = 0; idx < num_intersections; ++idx) {
        result.probabilities[idx] = probabilities_buffer[idx];
    }
    result.pass_probability = probabilities_buffer[num_intersections];
}

int Network::GetBestPolicyVertex(const GameState &state,
                                 const bool allow_pass) {
    const auto result = GetOutput(state, kRandom);
    const auto boardsize = result.board_size;
    const auto num_intersections = boardsize * boardsize;

    int max_idx = -1;
    int max_vtx = kPass;

    for (int idx = 0; idx < num_intersections; ++idx) {
        if (max_idx == -1 ||
                result.probabilities[max_idx] < result.probabilities[idx]) {
            const auto x = idx % boardsize;
            const auto y = idx / boardsize;
            const auto vtx = state.GetVertex(x,y);

            if (state.IsLegalMove(vtx)) {
                max_idx = idx;
                max_vtx = vtx;
            }
        }
    }

    if (allow_pass &&
            result.probabilities[max_idx] < result.pass_probability) {
        return kPass;
    }

    return max_vtx;
}

void Network::Destroy() {
    if (pipe_) {
        pipe_->Destroy();
    }
}

void Network::Reload(int board_size) {
    if (pipe_) {
        pipe_->Reload(board_size);
    }
}
