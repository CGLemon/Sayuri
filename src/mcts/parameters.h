#pragma once

#include "config.h"

class Parameters {
public:
    Parameters() = default;

    void Reset() {
        threads = GetOption<int>("threads");
        playouts = GetOption<int>("playouts");
        visits = GetOption<int>("visits");

        resign_threshold = GetOption<float>("resign_threshold");
        fpu_root_reduction = GetOption<float>("fpu_root_reduction");
        fpu_reduction = GetOption<float>("fpu_reduction");

        cpuct_init = GetOption<float>("cpuct_init");
        cpuct_root_init = GetOption<float>("cpuct_root_init");

        cpuct_base = GetOption<float>("cpuct_base");
        cpuct_root_base = GetOption<float>("cpuct_root_base");

        draw_factor = GetOption<float>("draw_factor");
        draw_factor = GetOption<float>("draw_root_factor");
    }

    int threads;
    int visits;
    int playouts;

    float resign_threshold;

    float fpu_root_reduction;
    float fpu_reduction;
    float cpuct_init;
    float cpuct_root_init;
    float cpuct_base;
    float cpuct_root_base;
    float draw_factor;
    float draw_root_factor;
};


