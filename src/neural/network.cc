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

    DNNLoder::Get().FormFile(dnn_weights, weightsfile);

    if (!dnn_weights->loaded) {
        dnn_weights.reset();
        dnn_weights = nullptr;
    }

    pipe_->Initialize(dnn_weights);
    SetCacheSize(GetOption<int>("playouts"));
}

void Network::SetCacheSize(int playouts) {
    nn_cache_.SetCapacity(playouts * 20);
}

void Network::ClearCache() {
    nn_cache_.Clear();
}

Network::Result Network::DummyForward(const Network::Inputs& inputs) const {
    Network::Result result{};

    auto rng = Random<RandomType::kXoroShiro128Plus>::Get();
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
    const auto Softmax = [](std::vector<float> &input, float temperature) {
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
    };

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

    auto probabilities_buffer = std::vector<float>(num_intersections+1);
    for (int idx = 0; idx < num_intersections; ++idx) {
        probabilities_buffer[idx] = result.probabilities[idx];
    }
    probabilities_buffer[num_intersections] = result.pass_probability;
    probabilities_buffer = Softmax(probabilities_buffer, 1);


    // Probabilities, ownership
    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto symm_index = Symmetry::Get().TransformIndex(symmetry, idx);
        out_result.probabilities[symm_index] = probabilities_buffer[idx];
        out_result.ownership[symm_index] = std::tanh(out_result.ownership[idx]);
    }
    out_result.pass_probability = probabilities_buffer[num_intersections];

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
    out_result.wdl_winrate = std::tanh(wdl_buffer[0] - wdl_buffer[2]);
    out_result.wdl_winrate = (out_result.wdl_winrate + 1.f) / 2;
    out_result.stm_winrate = (out_result.stm_winrate + 1.f) / 2;

    return out_result;
}

bool Network::ProbeCache(const GameState &state,
                         Network::Result &result) {
    // TODO: Cache the all symmetry board in early game.

    if (LookupCache(nn_cache_, state.GetHash(), result) ) {
        return true;
    }
    return false;
}

Network::Result
Network::GetOutput(const GameState &state,
                   const Ensemble ensemble,
                   int symmetry,
                   const bool read_cache,
                   const bool write_cache) {
    Result result;
    if (ensemble == kNone) {
        symmetry = Symmetry::kIdentitySymmetry;
    } else if (ensemble == kDirect) {
        assert(symmetry >= 0 && symmetry < Symmetry::kNumSymmetris);
    } else if (ensemble == kRandom) {
        auto rng = Random<RandomType::kXoroShiro128Plus>::Get();
        symmetry = rng.RandFix<Symmetry::kNumSymmetris>();
    }

    // Get result from cache, if the it is in the cache memory.
    if (read_cache) {
        if (ProbeCache(state, result)) {
            return result;
        }
    }

    result = GetOutputInternal(state, symmetry);

    // Write result to cache, if the it is not in the cache memory.
    if (write_cache) {
        nn_cache_.Insert(state.GetHash(), result);
    }

    return result;
}

std::string Network::GetOutputString(const GameState &state,
                                     const Ensemble ensemble,
                                     int symmetry) {
    const auto result = GetOutput(state, ensemble, symmetry, false, false);
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
            auto idx = state.GetIndex(x,y);
            out << std::setw(9) << std::fixed << std::setprecision(6) << result.probabilities[idx] << " ";
        }
        out << std::endl;
    }
    out << "pass probabilities: " << std::setw(9) << std::fixed << std::setprecision(6) << result.pass_probability << std::endl;
    out << std::endl;

    out << "ownership: " << std::endl;
    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            auto idx = state.GetIndex(x,y);
            out << std::setw(9) << std::fixed << std::setprecision(6) << result.ownership[idx] << " ";
        }
        out << std::endl;
    }
    out << std::endl;

    return out.str();
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
