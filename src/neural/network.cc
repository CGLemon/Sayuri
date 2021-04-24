#ifdef USE_CUDA
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

#include "game/symmetry.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "utils/log.h"
#include "utils/random.h"

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
#else
    using backend = BlasForwardPipe;
#endif

    pipe_ = std::make_unique<backend>();
    auto dnn_weights = std::make_shared<DNNWeights>();

    DNNLoder::Get().FormFile(dnn_weights, weightsfile);
    pipe_->Initialize(dnn_weights);

    if (dnn_weights->loaded) {
        // Do nothing...
    }
}

Network::Result
Network::GetOutputInternal(const GameState &state, const bool symmetry) {
/*
    auto policy_out = std::vector<float>(POLICYMAP * INTERSECTIONS);
    auto winrate_out = std::vector<float>(WINRATELAYER);

    auto input_planes = Model::gather_planes(position, symmetry);
    auto input_features = Model::gather_features(position);

    if (m_forward->valid()) {
        m_forward->forward(input_planes, input_features, policy_out, winrate_out);
    } else {
        // If we didn't load the network yet, output the random result.
        dummy_forward(policy_out, winrate_out);
    }

    // TODO: Remove "softmax_pol_temp" and "softmax_wdl_temp" to UCCI Option.
    const auto result = Model::get_result(policy_out,
                                          winrate_out,
                                          option<float>("softmax_pol_temp"),
                                          option<float>("softmax_wdl_temp"),
                                          symmetry);

    return result;
 */
    return Result{};
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
                   const Ensemble ensemble, int symmetry) {
    Result result;
    if (ensemble == NONE) {
        symmetry = Symmetry::kIdentitySymmetry;
    } else if (ensemble == DIRECT) {
        assert(symmetry >= 0 && symmetry < Symmetry::kNumSymmetris);
    } else if (ensemble == RANDOM_SYMMETRY) {
        auto rng = Random<RandomType::kXoroShiro128Plus>::Get();
        symmetry = rng.RandFix<Symmetry::kNumSymmetris>();
    }


    // Get result from cache, if the it is in the cache memory.
    if (ProbeCache(state, result)) {
        return result;
    }

    // result = get_output_internal(state, symm);


    // Write result to cache, if the it is not in the cache memory.
    nn_cache_.Insert(state.GetHash(), result);

    return result;
}
