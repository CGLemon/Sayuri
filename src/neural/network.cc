#include "neural/network.h"

#ifdef USE_CUDA
#include "neural/cuda/cuda_forward_pipe.h"
#endif

#ifdef USE_TENSORRT
#include "neural/trt/trt_forward_pipe.h"
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

#include <iomanip>
#include <random>
#include <sstream>

#include "config.h"
#include "game/symmetry.h"
#include "neural/blas/blas_forward_pipe.h"
#include "neural/encoder.h"
#include "neural/loader.h"
#include "utils/format.h"
#include "utils/log.h"
#include "utils/logits.h"
#include "utils/option.h"
#include "utils/random.h"

void Network::Initialize(const std::string& weightsfile) {
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
    LOGGING << "BLAS Core: Eigen" << ' ' << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.'
            << EIGEN_MINOR_VERSION << ' ' << "library." << std::endl;
#endif

#ifdef USE_TENSORRT
    using Backend = TrtForwardPipe;
#elif defined(USE_CUDA)
    using Backend = CudaForwardPipe;
#else
    using Backend = BlasForwardPipe;
#endif

    // Initialize the parameters.
    default_policy_offset_ =
        static_cast<PolicyBufferOffset>(GetOption<int>("policy_buffer_offset"));
    no_cache_ = GetOption<bool>("no_cache");
    early_symm_cache_ = GetOption<bool>("early_symm_cache");
    cache_memory_mib_ = 0;

    pipe_ = std::make_unique<Backend>();
    auto dnn_weights = std::make_shared<DNNWeights>();

    // Parse the NN weights file.
    DNNLoader::Get().FromFile(dnn_weights, weightsfile);

    // There is no weighs. Will disable the NN forward pipe.
    if (!dnn_weights->loaded) {
        dnn_weights.reset();
        dnn_weights = nullptr;
        no_cache_ = false; // Disable cache because it is not
                           // effect on dummy forwarding pipe.
    }

    // Initialize the NN forward pipe.
    pipe_->Initialize(dnn_weights);
    SetCacheSize(GetOption<int>("cache_memory_mib"));

#ifdef SELF_CHECK
    cpu_pipe_ = std::make_unique<BlasForwardPipe>();
    cpu_pipe_->Initialize(dnn_weights);
#endif

    ResetNumQueries();
}

size_t Network::SetCacheSize(size_t MiB) {
    const size_t mem_mib = std::min(std::max(size_t{5}, MiB), // min:   5 MB
                                    size_t{128 * 1024}        // max: 128 GB
    );

    const size_t entry_byte = nn_cache_.GetEntrySize();
    const size_t mem_byte = mem_mib * 1024 * 1024;
    size_t num_entries = mem_byte / entry_byte + 1;

    cache_memory_mib_ = mem_mib;
    nn_cache_.SetCapacity(num_entries);

    const double mem_used = static_cast<double>(num_entries * entry_byte) / (1024.f * 1024.f);
    if (no_cache_) {
        LOGGING << "Disable the NN cache.\n";
    } else {
        LOGGING << Format(
            "Allocated %.2f MiB memory for NN cache (%zu entries).\n", mem_used, num_entries);
    }
    return num_entries;
}

size_t Network::GetCacheMib() const {
    return cache_memory_mib_;
}

void Network::ClearCache() {
    nn_cache_.Clear();
}

std::string Network::GetName() const {
    return pipe_->GetName();
}

void Network::ResetNumQueries(size_t q) {
    num_queries_.store(q, std::memory_order_relaxed);
}

size_t Network::GetNumQueries() const {
    return num_queries_.load(std::memory_order_relaxed);
}

Network::Result Network::DummyForward(const Network::Inputs& inputs) const {
    Network::Result result{};

    auto rng = Random<>::Get();
    auto dist = std::uniform_real_distribution<float>(0, 1);

    const auto boardsize = inputs.board_size;
    const auto num_intersections = boardsize * boardsize;

    result.board_size = boardsize;
    for (int idx = 0; idx < 3; ++idx) {
        result.wdl[idx] = dist(rng);
    }
    result.stm_winrate = 0.f;

    for (int idx = 0; idx < num_intersections; ++idx) {
        result.probabilities[idx] = dist(rng);
    }
    result.pass_probability = dist(rng);

    return result;
}

Network::Result
Network::GetOutputInternal(const GameState& state, const int symmetry, PolicyBufferOffset offset) {
    auto inputs = Encoder::Get().GetInputs(state, symmetry, GetVersion());
    if (offset == PolicyBufferOffset::kDefault) {
        inputs.offset = default_policy_offset_;
    } else {
        inputs.offset = offset;
    }

    Network::Result result;
    if (pipe_->Valid()) {
        num_queries_.fetch_add(1, std::memory_order_relaxed);
        result = pipe_->Forward(inputs);
    } else {
        result = DummyForward(inputs);
    }
    TransformResult(result, symmetry);

#ifdef SELF_CHECK
    // TODO: Make self-check optional.
    if (pipe_->Valid() && cpu_pipe_->Valid()) {
        auto ref = cpu_pipe_->Forward(inputs);
        TransformResult(ref, symmetry);
        SelfCheck(result, ref);
    }
#endif

    return result;
}

bool Network::ProbeCache(const GameState& state, Network::Result& result) {
    if (nn_cache_.LookupItem(state.GetHash(), result)) {
        if (result.board_size == state.GetBoardSize()) {
            return true;
        }
    }

    if (state.GetBoardSize() >= state.GetMoveNumber() && early_symm_cache_) {
        for (int symm = Symmetry::kIdentitySymmetry + 1; symm < Symmetry::kNumSymmetris; ++symm) {
            if (nn_cache_.LookupItem(state.ComputeSymmetryHash(symm), result)) {
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

                // apply invert symmetry
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
Network::GetOutput(const GameState& state, const Ensemble ensemble, Network::Query query) {
    if (ensemble == kDirect) {
        if (query.symmetry < 0 || query.symmetry >= Symmetry::kNumSymmetris) {
            query.symmetry = Symmetry::kIdentitySymmetry;
        }
    } else if (ensemble == kRandom) {
        query.symmetry = Random<>::Get().RandFix(Symmetry::kNumSymmetris);
    }

    if (no_cache_ || ensemble == kAverage) {
        query.read_cache = false;
        query.write_cache = false;
    }

    Result result;
    if (query.read_cache && ProbeCache(state, result)) {
        ActivatePolicy(result, query.temperature);
        return result;
    }

    if (ensemble == kAverage) {
        for (int symm = Symmetry::kIdentitySymmetry; symm < Symmetry::kNumSymmetris; ++symm) {
            auto inter_result = GetOutputInternal(state, symm, query.offset);
            const auto boardsize = inter_result.board_size;
            const auto num_intersections = boardsize * boardsize;
            ActivatePolicy(inter_result, query.temperature);

            result.pass_probability += inter_result.pass_probability / Symmetry::kNumSymmetris;
            result.wdl_winrate += inter_result.wdl_winrate / Symmetry::kNumSymmetris;
            result.stm_winrate += inter_result.stm_winrate / Symmetry::kNumSymmetris;
            result.final_score += inter_result.final_score / Symmetry::kNumSymmetris;
            result.q_error += inter_result.q_error / Symmetry::kNumSymmetris;
            result.score_error += inter_result.score_error / Symmetry::kNumSymmetris;
            for (int idx = 0; idx < 3; ++idx) {
                result.wdl[idx] += inter_result.wdl[idx] / Symmetry::kNumSymmetris;
            }
            for (int idx = 0; idx < num_intersections; ++idx) {
                result.probabilities[idx] +=
                    inter_result.probabilities[idx] / Symmetry::kNumSymmetris;
                result.ownership[idx] += inter_result.ownership[idx] / Symmetry::kNumSymmetris;
            }
            if (symm == Symmetry::kIdentitySymmetry) {
                result.ImportQueryInfo(inter_result);
            }
        }
    } else {
        result = GetOutputInternal(state, query.symmetry, query.offset);
        if (query.write_cache) {
            nn_cache_.Insert(state.GetHash(), result);
        }
        ActivatePolicy(result, query.temperature);
    }
    return result;
}

std::string
Network::GetOutputString(const GameState& state, const Ensemble ensemble, Network::Query query) {
    query.read_cache = false;
    query.write_cache = false;
    const auto result = GetOutput(state, ensemble, query);
    const auto board_size = result.board_size;
    const auto num_intersections = state.GetNumIntersections();

    auto out = std::ostringstream{};

    out << "stm winrate: " << result.stm_winrate << std::endl;
    out << "wdl winrate: " << result.wdl_winrate << std::endl;
    out << "win probability: " << result.wdl[0] << std::endl;
    out << "draw probability: " << result.wdl[1] << std::endl;
    out << "loss probability: " << result.wdl[2] << std::endl;
    out << "final score: " << result.final_score << std::endl;
    out << "q error: " << result.q_error << std::endl;
    out << "score error: " << result.score_error << std::endl;

    out << "probabilities: " << std::endl;
    for (int idx = 0; idx < num_intersections; ++idx) {
        out << Format("%10.6f", result.probabilities[state.IndexToRowMajorIndex(idx)]);
        if ((idx + 1) % board_size == 0) {
            out << std::endl;
        }
    }
    out << Format("pass probabilities: %.6f\n", result.pass_probability);

    out << "ownership: " << std::endl;
    for (int idx = 0; idx < num_intersections; ++idx) {
        out << Format("%10.6f", result.ownership[state.IndexToRowMajorIndex(idx)]);
        if ((idx + 1) % board_size == 0) {
            out << std::endl;
        }
    }
    out << std::endl;

    return out.str();
}

void Network::SelfCheck(Network::Result result, Network::Result ref) {
    // Calculates L2-norm between result and ref.
    constexpr auto kMaxError = 0.2f;

    auto GetSquareError = [](double a, double b) -> double {
        const auto diff = a - b;
        return diff * diff;
    };

    ActivatePolicy(result, 1.0f);
    ActivatePolicy(ref, 1.0f);

    auto error = 0.0;
    const auto board_size = result.board_size;
    const auto num_intersections = board_size * board_size;

    for (int idx = 0; idx < num_intersections; ++idx) {
        error += GetSquareError(result.probabilities[idx], ref.probabilities[idx]);
    }
    error += GetSquareError(result.pass_probability, ref.pass_probability);
    error += GetSquareError(result.wdl_winrate, ref.wdl_winrate);
    error = std::sqrt(error);

    if (error > kMaxError || std::isnan(error)) {
        throw std::runtime_error(Format("GPU self-check mismatch (%10.6f).", error));
    }
}

void Network::TransformResult(Network::Result& result, const int symmetry) {
    const Network::Result result_buf = result;

    const auto boardsize = result.board_size;
    const auto num_intersections = boardsize * boardsize;

    auto probabilities_buffer = std::vector<float>(num_intersections);
    auto ownership_buffer = std::vector<float>(num_intersections);

    // copy result to buffer
    std::copy(std::begin(result_buf.probabilities),
              std::begin(result_buf.probabilities) + num_intersections,
              std::begin(probabilities_buffer));
    std::copy(std::begin(result_buf.ownership),
              std::begin(result_buf.ownership) + num_intersections,
              std::begin(ownership_buffer));

    // apply invert symmetry for probabilities, ownership
    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto symm_index = Symmetry::Get().TransformIndex(boardsize, symmetry, idx);
        result.probabilities[symm_index] = probabilities_buffer[idx];
        result.ownership[symm_index] = std::tanh(ownership_buffer[idx]);
    }

    // final score
    result.final_score = 20 * result_buf.final_score;

    // winrate
    auto wdl_buffer = std::vector<float>(3);
    wdl_buffer[0] = result_buf.wdl[0];
    wdl_buffer[1] = result_buf.wdl[1];
    wdl_buffer[2] = result_buf.wdl[2];
    wdl_buffer = Softmax(wdl_buffer, 1);

    result.wdl[0] = wdl_buffer[0];
    result.wdl[1] = wdl_buffer[1];
    result.wdl[2] = wdl_buffer[2];
    result.wdl_winrate = (wdl_buffer[0] - wdl_buffer[2] + 1.f) / 2;
    result.stm_winrate = (std::tanh(result_buf.stm_winrate) + 1.f) / 2;

    // error
    auto SoftplusSquare = [](float x) -> float {
        if (x <= 20.f) {
            x = std::log(1.f + std::exp(x));
        }
        return (x * x) / 4.f;
    };

    result.q_error = 0.25 * SoftplusSquare(result_buf.q_error);
    result.score_error = 150 * SoftplusSquare(result_buf.score_error);
}

void Network::ActivatePolicy(Result& result, const float temperature) const {
    const auto board_size = result.board_size;
    const auto num_intersections = board_size * board_size;

    auto probabilities_buffer = std::vector<float>(num_intersections + 1);

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

int Network::GetVertexWithPolicy(const GameState& state,
                                 const float temperature,
                                 const bool allow_pass) {
    const auto result = GetOutput(state, kRandom, Query::Get().SetTemperature(temperature));
    const auto boardsize = result.board_size;
    const auto num_intersections = boardsize * boardsize;

    auto select_vtx = kNullVertex;
    auto accum = float{0.0f};
    auto accum_vector = std::vector<std::pair<float, int>>{};

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto vtx = state.IndexToVertex(idx);
        if (state.IsLegalMove(vtx)) {
            accum += result.probabilities[idx];
            accum_vector.emplace_back(accum, vtx);
        }
    }

    if (accum_vector.empty() || allow_pass) {
        accum += result.pass_probability;
        accum_vector.emplace_back(std::pair<float, int>(accum, kPass));
    }

    auto distribution = std::uniform_real_distribution<float>{0.0, accum};
    auto pick = distribution(Random<>::Get());
    auto size = accum_vector.size();

    for (auto idx = size_t{0}; idx < size; ++idx) {
        if (pick < accum_vector[idx].first) {
            select_vtx = accum_vector[idx].second;
            break;
        }
    }

    return select_vtx;
}

PolicyBufferOffset Network::GetDefaultPolicyOffset() const {
    return default_policy_offset_;
}

int Network::GetVersion() const {
    return pipe_->GetVersion();
}

int Network::GetNumWorkers() const {
    return pipe_->GetNumWorkers();
}

bool Network::Valid() const {
    if (pipe_) {
        return pipe_->Valid();
    }
    return false;
}

void Network::Destroy() {
    if (pipe_) {
        pipe_->Destroy();
    }
}

void Network::Reconstruct(Network::Option option) {
    if (pipe_) {
        pipe_->Construct(option, nullptr);
    }
}
