#include "neural/loader.h"
#include "neural/network_basic.h"
#include "utils/splitter.h"
#include "utils/log.h"
#include "utils/format.h"
#include "utils/parse_float.h"
#include "utils/option.h"
#include "config.h"

#include <iostream>
#include <sstream>

#ifdef USE_FAST_PARSER
#include "fast_float.h"
#endif 

DNNLoder& DNNLoder::Get() {
    static DNNLoder lodaer;
    return lodaer;
}

void DNNLoder::FromFile(std::shared_ptr<DNNWeights> weights, std::string filename) {
    auto file = std::ifstream{};
    auto buffer = std::stringstream{};
    auto line = std::string{};

    if (filename.empty()) {
        LOGGING << "There is no weights file." << std::endl;
        return;
    }

    file.open(filename, std::ifstream::binary | std::ifstream::in);

    if (!file.is_open()) {
        LOGGING << "Couldn't open file:" << ' ' << filename << '!' << std::endl;
        return;
    }

    char c;
    while (file.get(c)) {
        // Copy the file data to buffer.
        buffer << c;
    }

    file.close();

    try {
        Parse(weights, buffer);
    } catch (const char *err) {
        // Should be not happned.

        LOGGING << "Fail to load the network file!" << std::endl
                    << Format("    Cause: %s.", err) << std::endl;
    }
}

void DNNLoder::Parse(std::shared_ptr<DNNWeights> weights, std::istream &buffer) {
   /**
    * get main
    * get info
    *   (About the network information.)
    * end
    *
    * get struct
    *   (About the network struct.)
    * end
    *
    * get parameters
    *   (The network weights are here. It is must be in)
    *    the last scope.)
    * end
    * end
    *
    */
    auto line = std::string{};

    if (std::getline(buffer, line)) {
        const auto spt = Splitter(line, 2);
        if (spt.GetWord(0)->Get<>() != "get" ||
                spt.GetWord(1)->Get<>() != "main") {
            throw "Weights file format is not acceptable";
        }
    } else {
        throw "Weights file is empty";
    }

    auto netinfo = NetInfo{};
    auto netstack = NetStack{};
    auto netstruct = NetStruct{};

    while (std::getline(buffer, line)) {
        const auto spt = Splitter(line, 2);
        if (spt.GetWord(0)->Get<>() == "get") {
            if (spt.GetWord(1)->Get<>() == "info") {
                ParseInfo(netinfo, buffer);
            } else if (spt.GetWord(1)->Get<>() == "stack") {
                ParseStack(netstack, buffer);
            } else if (spt.GetWord(1)->Get<>() == "struct") {
                ParseStruct(netstruct, buffer);
            } else if (spt.GetWord(1)->Get<>() == "parameters") {
                break;
            }
        } else if (spt.GetWord(0)->Get<std::string>() == "end") {
            // do nothing...
        }
    }

    buffer.clear();
    buffer.seekg(0, std::ios::beg);

    CkeckMisc(netinfo, netstack, netstruct);

    // Now start to parse the weights.
    while (std::getline(buffer, line)) {
        const auto spt = Splitter(line);
        if (spt.GetWord(0)->Get<>() == "get") {
            if (spt.GetWord(1)->Get<>() == "parameters") {
                FillWeights(netinfo, netstack, netstruct, weights, buffer);
            }
        }
    }
}

void DNNLoder::ParseInfo(NetInfo &netinfo, std::istream &buffer) const {
    auto line = std::string{};
    while (std::getline(buffer, line)) {
        const auto spt = Splitter(line);
        if (spt.GetWord(0)->Get<>()[0] == '#') {
            continue;
        } else if (spt.GetWord(0)->Get<>() == "end") {
            break;
        }
        netinfo.emplace(spt.GetWord(0)->Get<>(),
                            spt.GetWord(1)->Get<>());
    }

    const auto NotFound = [](NetInfo &netinfo, std::string target) -> bool {
        return std::end(netinfo) == netinfo.find(target);
    };

    if (NotFound(netinfo, "InputChannels")) {
        throw "InputChannels must be provided";
    }
    if (NotFound(netinfo, "ResidualBlocks")) {
        throw "ResidualBlocks must be provided";
    }
    if (NotFound(netinfo, "ResidualChannels")) {
        throw "ResidualChannels must be provided";
    }
    if (NotFound(netinfo, "PolicyExtract")) {
        throw "PolicyExtract must be provided";
    }
    if (NotFound(netinfo, "ValueExtract")) {
        throw "ValueExtract must be provided";
    }
}

void DNNLoder::ParseStack(NetStack &netstack, std::istream &buffer) const {
    auto line = std::string{};
    while (std::getline(buffer, line)) {
        const auto spt = Splitter(line);
        if (spt.GetWord(0)->Get<>()[0] == '#') {
            continue;
        } else if (spt.GetWord(0)->Get<>() == "end") {
            break;
        }
        netstack.emplace_back(spt.GetWord(0)->Get<>());
    }
}

void DNNLoder::ParseStruct(NetStruct &netstruct, std::istream &buffer) const {
    auto line = std::string{};
    auto cnt = size_t{0};
    while (std::getline(buffer, line)) {
        const auto spt = Splitter(line);
        if (spt.GetWord(0)->Get<>()[0] == '#') {
            continue;
        } else if (spt.GetWord(0)->Get<>() == "end") {
            break;
        }
            
        netstruct.emplace_back(LayerShape{});
        for (auto i = size_t{1}; i < spt.GetCount(); ++i) {
            const auto s = spt.GetWord(i)->Get<int>();
            netstruct[cnt].emplace_back(s);
        }

        const auto layer_name = spt.GetWord(0)->Get<>();
        if (layer_name == "FullyConnect") {
            if (netstruct[cnt].size() != 2) {
                throw "The FullyConnect Layer shape is error";
            }
        } else if (layer_name == "Convolution") {
            if (netstruct[cnt].size() != 3) {
                throw "The Convolution Layer shape is error";
            }
        } else if (layer_name == "BatchNorm") {
            if (netstruct[cnt].size() != 1) {
                throw "The BatchNorm layer shape is error";
            }
        } else {
            throw "The layer shape is error";
        }
        cnt++;
    }
}

void DNNLoder::CkeckMisc(NetInfo &netinfo, NetStack &netstack, NetStruct &netstruct) {
    const auto NotFound = [](NetInfo &netinfo, std::string target) -> bool {
        return std::end(netinfo) == netinfo.find(target);
    };

    use_binary_ = false;
    version_ = 1;

    if (!NotFound(netinfo, "FloatType")) {
        if (netinfo["FloatType"] == "float32bin") {
            use_binary_ = true;
        }
    }

    if (!NotFound(netinfo, "Version")) {
        version_ = std::stoi(netinfo["Version"]);

        // v1: Base format.
        //
        // v2: Fixed the batch normalize layer weights
        //     format. There are some error in the gammas
        //     compression process.

        if (version_ > 2) {
            throw "Do not support this version";
        }
    }

    if (!NotFound(netinfo, "NNType")) {
        // Not used now.
    }

    // Build the stack if it is not in weights file. It only
    // supports for residual block with SE.
    if (netstack.empty()) {
        const auto residuals = std::stoi(netinfo["ResidualBlocks"]);

        // First convolution layer.
        const auto inputs_cnt = 2;

        // The head layers.
        const auto heads_cnt = 10;

        auto inner_cnt = 0;
        for (int b = 0; b < residuals; ++b) {
            auto block_type = std::string{"ResidualBlock"}; 
            inner_cnt += 4;
            if (netstruct[inner_cnt+inputs_cnt].size() == 2 /* fullyconnect layer */) {
                block_type += "-SE";
                inner_cnt += 2;
            }
            netstack.emplace_back(block_type);
        }

        if ((int)netstruct.size() != heads_cnt + inner_cnt + inputs_cnt) {
            throw "Do not support this weights format";
        }
    }

    for (auto &block_type : netstack) {
        for (auto &c: block_type) {
            if (c == '-') {
                c = ' ';
            }
        }
        const auto spt = Splitter(block_type);
        for (int i = 0; i < (int)spt.GetCount(); ++i) {
            const auto component = spt.GetWord(i)->Get<>();

            if (component == "ResidualBlock" || component == "SE") {
                // do nothing...
            } else {
                throw Format("Do not support this block type [%s].",
                                 block_type.c_str());
            }
        }
    }
}

void DNNLoder::DumpInfo(std::shared_ptr<DNNWeights> weights) const {
    auto out = std::ostringstream{};

    out << "Network Verison: " << version_ << '\n';
    out << "Input Channels: " << weights->input_channels << '\n';
    out << "Residual Blocks: " << weights->residual_blocks << '\n';
    out << "Residual Channels: " << weights->residual_channels << '\n';

    for (int i = 0; i < weights->residual_blocks; ++i) {
        out << "  block " << i+1 << ':';
        if (weights->tower[i].apply_se) {
            out << " ResidualBlock-SE" << '\n';
        } else {
            out << " ResidualBlock" << '\n';
        }
    }

    out << "Policy Head Channels: " << weights->policy_extract_channels << '\n';
    out << "Value Head Channels: " << weights->value_extract_channels << '\n';

    LOGGING << out.str();
}

void DNNLoder::FillWeights(NetInfo &netinfo,
                               NetStack &netstack,
                               NetStruct &netstruct,
                               std::shared_ptr<DNNWeights> weights,
                               std::istream &buffer) const {

    weights->input_channels = std::stoi(netinfo["InputChannels"]);

    weights->residual_blocks = std::stoi(netinfo["ResidualBlocks"]);
    weights->residual_channels = std::stoi(netinfo["ResidualChannels"]);

    weights->policy_extract_channels = std::stoi(netinfo["PolicyExtract"]);
    weights->value_extract_channels = std::stoi(netinfo["ValueExtract"]);

    if (weights->input_channels != kInputChannels) {
        throw "The number of input channels is wrong.";
    }

    const auto SplitterFound = [](const Splitter &spt, std::string key) -> bool {
        if (const auto res = spt.Find(key)) {
            return true;
        }
        return false;
    };

    // There are three types layer. Each layer has
    // two line weights. Here they are.
    //
    // a). Fully connect layer
    //   1. weights
    //   2. biases
    // b). Convolution layer
    //   1. weights
    //   2. biases
    // c). Batch normalize layer (v2)
    //   1. mean
    //   2. standard deviation 

    // input layer
    const auto inputs_cnt = 2;
    const auto input_conv_shape = netstruct[0];
    FillConvolutionLayer(weights->input_conv,
                         buffer,
                         input_conv_shape[0],
                         input_conv_shape[1],
                         input_conv_shape[2]);
        
    const auto input_bn_shape = netstruct[1];
    FillBatchnormLayer(weights->input_bn,
                       buffer,
                       input_bn_shape[0]);

    if (weights->residual_channels != input_conv_shape[1] ||
            weights->residual_channels != input_bn_shape[0] ||
            input_conv_shape[2] != 3) {
        throw "The input layers are wrong";
    }

    const auto residuals = weights->residual_blocks;
    auto se_cnt = 0;
    for (int b = 0; b < residuals; ++b) {
        const auto block_spt = Splitter(netstack[b]);
        if (SplitterFound(block_spt, "ResidualBlock")) {
            weights->tower.emplace_back(ResidualBlock{});
        } else {
            throw "Miss the ResidualBlock";
        }

        const auto t_offset = 4 * b + 2 * se_cnt + inputs_cnt;
        const auto res_conv1_shape = netstruct[t_offset];
        const auto res_bn1_shape = netstruct[t_offset+1];
        const auto res_conv2_shape = netstruct[t_offset+2];
        const auto res_bn2_shape = netstruct[t_offset+3];
        auto tower_ptr = weights->tower.data() + b;
        
        FillConvolutionLayer(tower_ptr->conv1,
                             buffer,
                             res_conv1_shape[0],
                             res_conv1_shape[1],
                             res_conv1_shape[2]);

        FillBatchnormLayer(tower_ptr->bn1,
                           buffer,
                           res_bn1_shape[0]);

        if (weights->residual_channels != res_conv1_shape[0] ||
                weights->residual_channels != res_conv1_shape[1] ||
                weights->residual_channels != res_bn1_shape[0] || 
                res_conv1_shape[2] != 3) {
            throw "The Residual Block(1) is wrong";
        }

        FillConvolutionLayer(tower_ptr->conv2,
                             buffer,
                             res_conv2_shape[0],
                             res_conv2_shape[1],
                             res_conv2_shape[2]);

        FillBatchnormLayer(tower_ptr->bn2,
                           buffer,
                           res_bn2_shape[0]);

        if (weights->residual_channels != res_conv2_shape[0] ||
                weights->residual_channels != res_conv2_shape[1] ||
                weights->residual_channels != res_bn2_shape[0] ||
                res_conv2_shape[2] != 3) {
            throw "The Residual Block(2) is wrong";
        }

        if (SplitterFound(block_spt, "SE")) {
            tower_ptr->apply_se = true;
            se_cnt++;
            const auto se_squeeze_shape = netstruct[t_offset+4];
            const auto se_excite_shape = netstruct[t_offset+5];
            FillFullyconnectLayer(tower_ptr->squeeze,
                                  buffer,
                                  se_squeeze_shape[0],
                                  se_squeeze_shape[1]);
            FillFullyconnectLayer(tower_ptr->excite,
                                  buffer,
                                  se_excite_shape[0],
                                  se_excite_shape[1]);

            if (se_squeeze_shape[1] != se_excite_shape[0]) {
                throw "The SE Unit size is wrong (1).";
            }
            if (se_squeeze_shape[0] != 3 * weights->residual_channels) {
                throw "The SE Unit size is wrong (2).";
            }
            if (se_excite_shape[1] != 2 * weights->residual_channels) {
                throw "The SE Unit size is wrong (3).";
            }
                
            tower_ptr->se_size = se_squeeze_shape[1];
        } else {
            tower_ptr->apply_se = false;
        }
    }

    const auto h_offset = 4 * residuals + 2 * se_cnt + inputs_cnt;

    // policy head
    const auto p_ex_conv_shape = netstruct[h_offset + 0];
    FillConvolutionLayer(weights->p_ex_conv,
                         buffer,
                         p_ex_conv_shape[0],
                         p_ex_conv_shape[1],
                         p_ex_conv_shape[2]);

    const auto p_ex_bn_shape = netstruct[h_offset + 1];
    FillBatchnormLayer(weights->p_ex_bn,
                       buffer,
                       p_ex_bn_shape[0]);


    const auto p_inter_fc_shape = netstruct[h_offset + 2];
    FillFullyconnectLayer(weights->p_inter_fc,
                          buffer,
                          p_inter_fc_shape[0],
                          p_inter_fc_shape[1]);

    const auto prob_conv_shape = netstruct[h_offset + 3];
    FillConvolutionLayer(weights->prob_conv,
                         buffer,
                         prob_conv_shape[0],
                         prob_conv_shape[1],
                         prob_conv_shape[2]);

    const auto pass_fc_shape = netstruct[h_offset + 4];
    FillFullyconnectLayer(weights->pass_fc,
                          buffer,
                          pass_fc_shape[0],
                          pass_fc_shape[1]);

    if (p_ex_conv_shape[2] != 1 || prob_conv_shape[2] != 1) {
        throw "The policy convolution kernel size is wrong";
    }
    if (prob_conv_shape[1] != kOuputProbabilitiesChannels) {
        throw "The number of policy ouput size is wrong";
    }
    if (p_inter_fc_shape[1] != pass_fc_shape[0] ||
            p_inter_fc_shape[0] != 3 * weights->policy_extract_channels ||
            p_inter_fc_shape[1] != 1 * weights->policy_extract_channels) {
        throw "The number of policy fully connect size is wrong";
    }
    if (pass_fc_shape[1] != kOuputPassProbability) {
        throw "The number of pass ouput size is wrong";
    }

    // value head
    const auto v_ex_conv_shape = netstruct[h_offset + 5];
    FillConvolutionLayer(weights->v_ex_conv,
                         buffer,
                         v_ex_conv_shape[0],
                         v_ex_conv_shape[1],
                         v_ex_conv_shape[2]);

    const auto v_ex_bn_shape = netstruct[h_offset  + 6];
    FillBatchnormLayer(weights->v_ex_bn,
                       buffer,
                       v_ex_bn_shape[0]);

    const auto v_inter_fc_shape = netstruct[h_offset + 7];
    FillFullyconnectLayer(weights->v_inter_fc,
                          buffer,
                          v_inter_fc_shape[0],
                          v_inter_fc_shape[1]);

    const auto v_os_conv_shape = netstruct[h_offset + 8];
    FillConvolutionLayer(weights->v_ownership,
                         buffer,
                         v_os_conv_shape[0],
                         v_os_conv_shape[1],
                         v_os_conv_shape[2]);

    const auto misc_fc_shape = netstruct[h_offset + 9];
    FillFullyconnectLayer(weights->v_misc,
                          buffer,
                          misc_fc_shape[0],
                          misc_fc_shape[1]);
    if (v_ex_conv_shape[2] != 1 || v_os_conv_shape[2] != 1) {
        throw "The value convolution kernel size is wrong";
    }
    if (v_os_conv_shape[1] != kOuputOwnershipChannels) {
        throw "The number of ownership ouput size is wrong";
    }
    if (v_inter_fc_shape[1] != misc_fc_shape[0] ||
            v_inter_fc_shape[0] != 3 * weights->value_extract_channels ||
            v_inter_fc_shape[1] != 3 * weights->value_extract_channels) {
        throw "The number of value fully connect size is wrong";
    }
    if (misc_fc_shape[1] != kOuputValueMisc) {
        throw "The misc value layer size is wrong";
    }

    auto line = std::string{};
    std::getline(buffer, line);
    const auto spt = Splitter(line);
    if (spt.GetWord(0)->Get<std::string>() != "end") {
        throw "Not end? Weights file format is not acceptable";
    }
    weights->loaded = true;
    DumpInfo(weights);
    ProcessWeights(weights);
    weights->winograd = GetOption<bool>("winograd");
}

void DNNLoder::ProcessWeights(std::shared_ptr<DNNWeights> weights) const {
    // input layer
    for (auto idx = size_t{0}; idx < weights->input_conv.GetBiases().size(); ++idx) {
        weights->input_conv.GetBiases()[idx] -= weights->input_bn.GetMeans()[idx];
        weights->input_bn.GetMeans()[idx] = 0.0f;

        size_t stride = weights->input_conv.GetWeights().size() /
                            weights->input_conv.GetBiases().size();
        auto scale = weights->input_bn.GetStddevs()[idx];
        for (auto k = size_t{0}; k < stride; ++k) {
            weights->input_conv.GetWeights()[stride * idx + k] *= scale;
        }
        weights->input_conv.GetBiases()[idx] *= scale;
        weights->input_bn.GetStddevs()[idx] = 1.0f;
    }

    // residual tower
    for (auto &residual : weights->tower) {
        // 1st layer
        for (auto idx = size_t{0}; idx < residual.conv1.GetBiases().size(); ++idx) {
            residual.conv1.GetBiases()[idx] -= residual.bn1.GetMeans()[idx];
            residual.bn1.GetMeans()[idx] = 0.0f;

            size_t stride1 = residual.conv1.GetWeights().size() /
                                residual.conv1.GetBiases().size();
            auto scale1 = residual.bn1.GetStddevs()[idx];
            for (auto k = size_t{0}; k < stride1; ++k) {
                residual.conv1.GetWeights()[stride1 * idx + k] *= scale1;
            }
            residual.conv1.GetBiases()[idx] *= scale1;
            residual.bn1.GetStddevs()[idx] = 1.0f;
        }

        // 2nd layer
        for (auto idx = size_t{0}; idx < residual.conv2.GetBiases().size(); ++idx) {
            residual.conv2.GetBiases()[idx] -= residual.bn2.GetMeans()[idx];
            residual.bn2.GetMeans()[idx] = 0.0f;

            size_t stride2 = residual.conv2.GetWeights().size() /
                                residual.conv2.GetBiases().size();
            auto scale2 = residual.bn2.GetStddevs()[idx];
            for (auto k = size_t{0}; k < stride2; ++k) {
                residual.conv2.GetWeights()[stride2 * idx + k] *= scale2;
            }
            residual.conv2.GetBiases()[idx] *= scale2;
            residual.bn2.GetStddevs()[idx] = 1.0f;
        }
    }
    // policy head
    for (auto idx = size_t{0}; idx < weights->p_ex_conv.GetBiases().size(); ++idx) {
        weights->p_ex_conv.GetBiases()[idx] -= weights->p_ex_bn.GetMeans()[idx];
        weights->p_ex_bn.GetMeans()[idx] = 0.0f;

        size_t stride = weights->p_ex_conv.GetWeights().size() /
                            weights->p_ex_conv.GetBiases().size();
        auto scale = weights->p_ex_bn.GetStddevs()[idx];
        for (auto k = size_t{0}; k < stride; ++k) {
            weights->p_ex_conv.GetWeights()[stride * idx + k] *= scale;
        }
        weights->p_ex_conv.GetBiases()[idx] *= scale;
        weights->p_ex_bn.GetStddevs()[idx] = 1.0f;
    }
    // value head
    for (auto idx = size_t{0}; idx < weights->v_ex_conv.GetBiases().size(); ++idx) {
        weights->v_ex_conv.GetBiases()[idx] -= weights->v_ex_bn.GetMeans()[idx];
        weights->v_ex_bn.GetMeans()[idx] = 0.0f;

        size_t stride = weights->v_ex_conv.GetWeights().size() /
                            weights->v_ex_conv.GetBiases().size();
        auto scale = weights->v_ex_bn.GetStddevs()[idx];
        for (auto k = size_t{0}; k < stride; ++k) {
            weights->v_ex_conv.GetWeights()[stride * idx + k] *= scale;
        }
        weights->v_ex_conv.GetBiases()[idx] *= scale;
        weights->v_ex_bn.GetStddevs()[idx] = 1.0f;
    }
}

void DNNLoder::GetWeightsFromBuffer(std::vector<float> &weights, std::istream &buffer) const {
    weights.clear();

    if (use_binary_) {
        while (true) {
            // Get the next float.
            float w = ParseBinFloat32(buffer);

            if (MatchFloat32(w, 0xffffffff)) {
                // It means the end of line.
                break;
            }

            weights.emplace_back(w);
        }
    } else {
        auto line = std::string{};
        if (std::getline(buffer, line)) {
            // On MacOS, if the numeric is too small, stringstream
            // can not parse the number to float, but double is ok.
            double weight;

#ifdef USE_FAST_PARSER
            auto start_ptr = line.data();
            auto end_ptr = line.data();
            auto line_size = line.size();
            auto finish_ptr = line.data() + line_size;
            weights.reserve(line_size / 12);

            while (*end_ptr == ' ') {
                end_ptr++;
                if (end_ptr == finish_ptr) break;
            }
            start_ptr = end_ptr;

            while (start_ptr != finish_ptr) {
                while (*end_ptr != ' ') {
                    end_ptr++;
                    if (end_ptr == finish_ptr) break;
                }
                const auto is_ok = fast_float::from_chars<double>(start_ptr, end_ptr, weight);
                if (is_ok.ec != std::errc()) {
                    throw "There is non-numeric in parameters";
                }

                weights.emplace_back(weight);

                while (*end_ptr == ' ') {
                    end_ptr++;
                    if (end_ptr == finish_ptr) break;
                }
                start_ptr = end_ptr;
            }
#else 
            std::stringstream line_buffer(line);
            while(line_buffer >> weight) {
                weights.emplace_back(weight);
            }
#endif
        }
    }
}

void DNNLoder::FillFullyconnectLayer(LinearLayer &layer,
                                         std::istream &buffer,
                                         const int in_size,
                                         const int out_size) const {
    auto weights = std::vector<float>{};
    layer.Set(in_size, out_size);    

    GetWeightsFromBuffer(weights, buffer);
    layer.LoadWeights(weights);

    GetWeightsFromBuffer(weights, buffer);
    layer.LoadBiases(weights);
}

void DNNLoder::FillBatchnormLayer(BatchNormLayer &layer,
                                      std::istream &buffer,
                                      const int channels) const {
    auto weights = std::vector<float>{};
    layer.Set(channels);

    GetWeightsFromBuffer(weights, buffer);
    layer.LoadMeans(weights);

    GetWeightsFromBuffer(weights, buffer);
    layer.LoadStddevs(weights, version_==1);
}

void DNNLoder::FillConvolutionLayer(ConvLayer &layer,
                                        std::istream &buffer,
                                        const int in_channels,
                                        const int out_channels,
                                        const int kernel_size) const {
    auto weights = std::vector<float>{};    
    layer.Set(in_channels, out_channels, kernel_size);

    GetWeightsFromBuffer(weights, buffer);
    layer.LoadWeights(weights);
    
    GetWeightsFromBuffer(weights, buffer);
    layer.LoadBiases(weights);
}
