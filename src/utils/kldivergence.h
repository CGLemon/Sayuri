#pragma once

#include <vector>
#include <type_traits>

template<
    typename T,
    typename = std::enable_if_t<
                   std::is_floating_point<T>::value
               >
>
bool ComputeKlDivergence(const std::vector<T> &p,
                         const std::vector<T> &q,
                         T& kld_result,
                         double buff=1e-8) {
    const size_t psize = p.size();
    const size_t qsize = q.size();
    kld_result = 0.f;

    if (psize != qsize) {
        return false;
    }

    for (size_t i = 0; i < psize; ++i) {
        double p_val = std::max(buff, (double)p[i]);
        double q_val = std::max(buff, (double)q[i]);
        kld_result += p_val * std::log(p_val / q_val);
    }

    return true;
}

template<
    typename T,
    typename = std::enable_if_t<
                   std::is_floating_point<T>::value
               >
>
T GetKlDivergence(const std::vector<T> &p,
                  const std::vector<T> &q,
                  double buff=1e-8) {
    T kld_result;
    ComputeKlDivergence(p, q, kld_result, buff);
    return kld_result;
}
