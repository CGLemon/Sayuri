#include "neural/activation.h"
#include "neural/loader.h"
#include "neural/network_basic.h"
#include "utils/log.h"
#include "utils/format.h"
#include "utils/filesystem.h"
#include "utils/parse_float.h"
#include "utils/option.h"
#include "utils/time.h"
#include "config.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef USE_FAST_PARSER
#include "fast_float.h"
#endif

DNNLoader& DNNLoader::Get() {
    static DNNLoader lodaer;
    return lodaer;
}

void DNNLoader::FromFile(std::shared_ptr<DNNWeights> weights, std::string filename) {
    auto file = std::ifstream{};
    auto buffer = std::stringstream{};
    auto line = std::string{};

    weights_ = weights.get();
    Timer timer;

    if (filename.empty()) {
        LOGGING << "There is no weights file." << std::endl;
        return;
    }

    file.open(filename, std::ifstream::binary | std::ifstream::in);

    if (!file.is_open()) {
        LOGGING << Format("Couldn't open weights file from %s!", filename.c_str())
                    << std::endl;
        return;
    }

    buffer << file.rdbuf();
    file.close();

    try {
        LOGGING << Format("Load the weights file from: %s.", filename.c_str())
                    << std::endl;
        Parse(buffer);
        LOGGING << Format("Done! Load the weights file in %.2f sec.",
                              timer.GetDurationMilliseconds()/1000.f)
                    << std::endl;
       auto pathvec = SplitPath(filename);
       if (!pathvec.empty()) {
           weights->name = *std::rbegin(pathvec);
       } else {
           weights->name = "network";
       }
    } catch (const std::exception& e) {
        LOGGING << "Fail to load the network file!" << std::endl
                    << Format("    Cause: %s", e.what()) << std::endl;
    }
}

void DNNLoader::Parse(std::istream &buffer) {
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
    *   (The network weights are here. It is must be in
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
            throw std::runtime_error{"weights file format is not acceptable"};
        }
    } else {
        throw std::runtime_error{"weights file is empty"};
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
                // load the parameters later...
                break;
            }
        } else if (spt.GetWord(0)->Get<std::string>() == "end") {
            // do nothing...
        }
    }
    CheckMisc(netinfo, netstack, netstruct);

    // Now start to parse the weights.
    FillWeights(netinfo, netstack, netstruct, buffer);
}

void DNNLoader::ParseInfo(NetInfo &netinfo, std::istream &buffer) const {
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
}

void DNNLoader::ParseStack(NetStack &netstack, std::istream &buffer) const {
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

void DNNLoader::ParseStruct(NetStruct &netstruct, std::istream &buffer) const {
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
                throw std::runtime_error{"FullyConnect Layer shape is error"};
            }
        } else if (layer_name == "Convolution") {
            if (netstruct[cnt].size() != 3) {
                throw std::runtime_error{"Convolution Layer shape is error"};
            }
        } else if (layer_name == "DepthwiseConvolution") {
            if (netstruct[cnt].size() != 3) {
                throw std::runtime_error{"DepthwiseConvolution Layer shape is error"};
            }
        } else if (layer_name == "BatchNorm") {
            if (netstruct[cnt].size() != 1) {
                throw std::runtime_error{"BatchNorm layer shape is error"};
            }
        } else {
            throw std::runtime_error{"layer shape is error"};
        }
        cnt++;
    }
}

void DNNLoader::CheckMisc(NetInfo &netinfo, NetStack &netstack, NetStruct &netstruct) {
    const auto NotFound = [](NetInfo &netinfo, std::string target) -> bool {
        return std::end(netinfo) == netinfo.find(target);
    };

    use_binary_ = false;
    weights_->version = 1;

    if (!NotFound(netinfo, "FloatType")) {
        if (netinfo["FloatType"] == "float32bin") {
            use_binary_ = true;
        }
    }

    if (!NotFound(netinfo, "Version")) {
        weights_->version = std::stoi(netinfo["Version"]);

        // v1: First network structure. Support for ResidualBlock
        //     and BottleneckBlock.
        //
        // v2: Fixed the batch normalize layer weights
        //     format. There are some error in the gammas
        //     compression process.
        //
        // v3: Add more output heads.
        //
        // v4: Add support for MixerBlock and more activation
        //     function.
        //
        // v5: Add support for NestedBottleneckBlock and RepLK
        //     policy head.
    }

    if (weights_->version >= 6) {
        throw std::runtime_error{"do not support this version"};
    } else if (weights_->version >= 3) {
        // v5 ~ v3
        weights_->input_channels = 43;
        weights_->probabilities_channels = 5;
        weights_->pass_probability_outputs = 5;
        weights_->ownership_channels = 1;
        weights_->value_misc_outputs = 15;
    } else {
        // v2 ~ v1
        weights_->input_channels = 38;
        weights_->probabilities_channels = 1;
        weights_->pass_probability_outputs = 1;
        weights_->ownership_channels = 1;
        weights_->value_misc_outputs = 5;
    }

    if (!NotFound(netinfo, "NNType")) {
        // Not used.
    }

    if (NotFound(netinfo, "PolicyHeadType")) {
        weights_->policy_head_type = PolicyHeadType::kNormal;
    } else {
        auto policy_head_type = netinfo["PolicyHeadType"];
        for (char &c: policy_head_type) {
            c = std::tolower(c);
        }
        if (policy_head_type == "normal") {
            weights_->policy_head_type = PolicyHeadType::kNormal;
        } else if (policy_head_type == "replk") {
            weights_->policy_head_type = PolicyHeadType::kRepLK;
        } else {
            throw std::runtime_error{"unknown policy head type"};
        }
    }

    if (NotFound(netinfo, "ActivationFunction")) {
        weights_->default_act = Activation::kReLU;
    } else {
        weights_->default_act = StringToAct(netinfo["ActivationFunction"]);
    }

    // Construct the stack if it is not in weights file. Under this condition,
    // it only supports for residual block with SE.
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
            throw std::runtime_error{"do not support this weights format"};
        }
    }

    // Confirm all components is supported.
    const auto supported_components = std::vector<std::string>{
        "ResidualBlock",
        "BottleneckBlock",
        "NestedBottleneckBlock",
        "MixerBlock",
        "SE",
        "FixUp"
    };

    for (auto &block_type : netstack) {
        for (auto &c: block_type) {
            if (c == '-') {
                c = ' ';
            }
        }
        const auto spt = Splitter(block_type);
        for (int i = 0; i < (int)spt.GetCount(); ++i) {
            const auto component = spt.GetWord(i)->Get<>();
            const auto it = std::find(
                std::begin(supported_components),
                std::end(supported_components), component);

            if (it == std::end(supported_components)) {
                throw std::runtime_error{
                          Format("do not support this block type [%s]", block_type.c_str())};
            }
        }
    }
}

void DNNLoader::DumpInfo() const {
    auto out = std::ostringstream{};

    out << "Network Version: " << weights_->version << '\n';
    out << "Input Channels: " << weights_->input_channels << '\n';
    out << "Residual Blocks: " << weights_->residual_blocks << '\n';
    out << "Residual Channels: " << weights_->residual_channels << '\n';

    for (int i = 0; i < weights_->residual_blocks; ++i) {
        auto tower_ptr =  weights_->tower[i].get();

        out << "  block " << i+1 << ": ";
        if (tower_ptr->IsResidualBlock()) {
            out << "ResidualBlock";
        } else if (tower_ptr->IsBottleneckBlock()) {
            out << "BottleneckBlock";
        } else if (tower_ptr->IsNestedBottleneckBlock()) {
            out << "NestedBottleneckBlock";
        } else if (tower_ptr->IsMixerBlock()) {
            out << "MixerBlock";
        } else {
            continue; // unknown block
        }
        if (tower_ptr->apply_se) {
            out << "-SE";
        }
        out << '\n';
    }

    auto pol_head_type = std::string{"Normal"};
    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        pol_head_type = "RepLK";
    }
    out << "Policy Head Type: " << pol_head_type << '\n';
    out << "Policy Head Channels: " << weights_->policy_head_channels << '\n';
    out << "Value Head Channels: " << weights_->value_head_channels << '\n';

    LOGGING << out.str();
}

int DNNLoader::FillBlock(int offset,
                         Splitter block_spt,
                         NetStruct &netstruct,
                         std::istream &buffer) const {
    auto SplitterFound = [](Splitter &spt, std::string key) {
        if (const auto res = spt.Find(key)) {
            return true;
        }
        return false;
    };

    // Push the basic block.
    if (SplitterFound(block_spt, "ResidualBlock")) {
        weights_->tower.emplace_back(std::make_unique<ResidualBlock>());
    } else if (SplitterFound(block_spt, "BottleneckBlock")) {
        weights_->tower.emplace_back(std::make_unique<BottleneckBlock>());
    } else if (SplitterFound(block_spt, "NestedBottleneckBlock")) {
        weights_->tower.emplace_back(std::make_unique<NestedBottleneckBlock>());
    } else if (SplitterFound(block_spt, "MixerBlock")) {
        weights_->tower.emplace_back(std::make_unique<MixerBlock>());
    } else {
        throw std::runtime_error{"need the ResidualBlock, BottleneckBlock, NestedBottleneckBlock or MixerBlock"};
    }
    auto tower_ptr = std::rbegin(weights_->tower)->get();
    tower_ptr->apply_se = SplitterFound(block_spt, "SE");

    if (tower_ptr->IsResidualBlock()) {
        // residual block
        // 1). Convolution layer with 3x3 kernel
        // 2). Batch normalize layer
        // 3). Convolution layer with 3x3 kernel
        // 4). Batch normalize layer

        // 1st layers.
        const auto res_conv1_shape = netstruct[offset++];
        const auto res_bn1_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->conv1, buffer,
            res_conv1_shape[0], res_conv1_shape[1], res_conv1_shape[2]);
        FillBatchnormLayer(tower_ptr->bn1, buffer, res_bn1_shape[0]);

        // 2nd layers.
        const auto res_conv2_shape = netstruct[offset++];
        const auto res_bn2_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->conv2, buffer,
            res_conv2_shape[0], res_conv2_shape[1], res_conv2_shape[2]);
        FillBatchnormLayer(tower_ptr->bn2, buffer, res_bn2_shape[0]);

        const auto channels = weights_->residual_channels;
        const auto kernel = 3;
        if (channels != res_conv1_shape[0] ||
                channels != res_conv1_shape[1] ||
                channels != res_bn1_shape[0] ||
                channels != res_conv2_shape[0] ||
                channels != res_conv2_shape[1] ||
                channels != res_bn2_shape[0]) {
            throw std::runtime_error{"the channels of residual block is wrong"};
        }
        if (kernel != res_conv1_shape[2] ||
                kernel != res_conv2_shape[2]) {
            throw std::runtime_error{"the kernel of residual block is wrong"};
        }
    } else if (tower_ptr->IsBottleneckBlock()) {
        // bottleneck block
        // 1). Convolution layer with 1x1 kernel
        // 2). Batch normalize layer
        // 3). Convolution layer with 3x3 kernel
        // 4). Batch normalize layer
        // 5). Convolution layer with 3x3 kernel
        // 6). Batch normalize layer
        // 7). Convolution layer with 1x1 kernel
        // 8). Batch normalize layer

        // pre-bottleneck layers
        const auto pre_btl_conv_shape = netstruct[offset++];
        const auto pre_btl_bn_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->pre_btl_conv, buffer,
            pre_btl_conv_shape[0], pre_btl_conv_shape[1], pre_btl_conv_shape[2]);
        FillBatchnormLayer(tower_ptr->pre_btl_bn, buffer, pre_btl_bn_shape[0]);

        // 1st layers.
        const auto res_conv1_shape = netstruct[offset++];
        const auto res_bn1_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->conv1, buffer,
            res_conv1_shape[0], res_conv1_shape[1], res_conv1_shape[2]);
        FillBatchnormLayer(tower_ptr->bn1, buffer, res_bn1_shape[0]);

        // 2nd layers.
        const auto res_conv2_shape = netstruct[offset++];
        const auto res_bn2_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->conv2, buffer,
            res_conv2_shape[0], res_conv2_shape[1], res_conv2_shape[2]);
        FillBatchnormLayer(tower_ptr->bn2, buffer, res_bn2_shape[0]);

        // post-bottleneck layers
        const auto post_btl_conv_shape = netstruct[offset++];
        const auto post_btl_bn_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->post_btl_conv, buffer,
                post_btl_conv_shape[0], post_btl_conv_shape[1], post_btl_conv_shape[2]);
        FillBatchnormLayer(tower_ptr->post_btl_bn, buffer, post_btl_bn_shape[0]);

        const auto outer_channels = weights_->residual_channels;
        const auto inner_channels = pre_btl_conv_shape[1];
        const auto kernel = 3;
        tower_ptr->bottleneck_channels = inner_channels;
        if (outer_channels != pre_btl_conv_shape[0] ||
                outer_channels != post_btl_conv_shape[1] ||
                outer_channels != post_btl_bn_shape[0]) {
            throw std::runtime_error{"the outer channels of bottleneck block is wrong"};
        }
        if (inner_channels != pre_btl_bn_shape[0] ||
                inner_channels != res_conv1_shape[0] ||
                inner_channels != res_conv1_shape[1] ||
                inner_channels != res_bn1_shape[0] ||
                inner_channels != res_conv2_shape[0] ||
                inner_channels != res_conv2_shape[1] ||
                inner_channels != res_bn2_shape[0]) {
            throw std::runtime_error{"the inner channels of bottleneck block is wrong"};
        }
        if (kernel != res_conv1_shape[2] ||
                kernel != res_conv2_shape[2] ||
                1 != pre_btl_conv_shape[2] ||
                1 != post_btl_conv_shape[2]) {
            throw std::runtime_error{"the kernel of bottleneck block is wrong"};
        }
    } else if (tower_ptr->IsNestedBottleneckBlock()) {
        // nested bottleneck block
        // 1).  Convolution layer with 1x1 kernel
        // 2).  Batch normalize layer
        // ---- 1st Residual Block ----
        //     3).  Convolution layer with 3x3 kernel
        //     4).  Batch normalize layer
        //     5).  Convolution layer with 3x3 kernel
        //     6).  Batch normalize layer
        // ---- 2nd Residual Block ----
        //     7).  Convolution layer with 3x3 kernel
        //     8).  Batch normalize layer
        //     9).  Convolution layer with 3x3 kernel
        //     10). Batch normalize layer
        // 11). Convolution layer with 1x1 kernel
        // 12). Batch normalize layer

        // pre-bottleneck layers
        const auto pre_btl_conv_shape = netstruct[offset++];
        const auto pre_btl_bn_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->pre_btl_conv, buffer,
            pre_btl_conv_shape[0], pre_btl_conv_shape[1], pre_btl_conv_shape[2]);
        FillBatchnormLayer(tower_ptr->pre_btl_bn, buffer, pre_btl_bn_shape[0]);

        // 1st layers.
        const auto res_conv1_shape = netstruct[offset++];
        const auto res_bn1_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->conv1, buffer,
            res_conv1_shape[0], res_conv1_shape[1], res_conv1_shape[2]);
        FillBatchnormLayer(tower_ptr->bn1, buffer, res_bn1_shape[0]);

        // 2nd layers.
        const auto res_conv2_shape = netstruct[offset++];
        const auto res_bn2_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->conv2, buffer,
            res_conv2_shape[0], res_conv2_shape[1], res_conv2_shape[2]);
        FillBatchnormLayer(tower_ptr->bn2, buffer, res_bn2_shape[0]);

        // 3rd layers.
        const auto res_conv3_shape = netstruct[offset++];
        const auto res_bn3_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->conv3, buffer,
            res_conv3_shape[0], res_conv3_shape[1], res_conv3_shape[2]);
        FillBatchnormLayer(tower_ptr->bn3, buffer, res_bn3_shape[0]);

        // 4th layers.
        const auto res_conv4_shape = netstruct[offset++];
        const auto res_bn4_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->conv4, buffer,
            res_conv4_shape[0], res_conv4_shape[1], res_conv4_shape[2]);
        FillBatchnormLayer(tower_ptr->bn4, buffer, res_bn4_shape[0]);

        // post-bottleneck layers
        const auto post_btl_conv_shape = netstruct[offset++];
        const auto post_btl_bn_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->post_btl_conv, buffer,
                post_btl_conv_shape[0], post_btl_conv_shape[1], post_btl_conv_shape[2]);
        FillBatchnormLayer(tower_ptr->post_btl_bn, buffer, post_btl_bn_shape[0]);

        const auto outer_channels = weights_->residual_channels;
        const auto inner_channels = pre_btl_conv_shape[1];
        const auto kernel = 3;
        tower_ptr->bottleneck_channels = inner_channels;
        if (outer_channels != pre_btl_conv_shape[0] ||
                outer_channels != post_btl_conv_shape[1] ||
                outer_channels != post_btl_bn_shape[0]) {
            throw std::runtime_error{"the outer channels of nested bottleneck block is wrong"};
        }
        if (inner_channels != pre_btl_bn_shape[0] ||
                inner_channels != res_conv1_shape[0] ||
                inner_channels != res_conv1_shape[1] ||
                inner_channels != res_bn1_shape[0] ||
                inner_channels != res_conv2_shape[0] ||
                inner_channels != res_conv2_shape[1] ||
                inner_channels != res_bn2_shape[0] ||
                inner_channels != res_conv3_shape[0] ||
                inner_channels != res_conv3_shape[1] ||
                inner_channels != res_bn3_shape[0] ||
                inner_channels != res_conv4_shape[0] ||
                inner_channels != res_conv4_shape[1] ||
                inner_channels != res_bn4_shape[0]) {
            throw std::runtime_error{"the inner channels of nested bottleneck block is wrong"};
        }
        if (kernel != res_conv1_shape[2] ||
                kernel != res_conv2_shape[2] ||
                kernel != res_conv3_shape[2] ||
                kernel != res_conv4_shape[2] ||
                1 != pre_btl_conv_shape[2] ||
                1 != post_btl_conv_shape[2]) {
            throw std::runtime_error{"the kernel of nested bottleneck block is wrong"};
        }
    } else if (tower_ptr->IsMixerBlock()) {
        // mixer block
        // 1). Depthwise convolution layer with NxN kernel
        // 2). Batch normalize layer
        // 3). Convolution layer with 1x1 kernel
        // 4). Batch normalize layer
        // 5). Convolution layer with 1x1 kernel
        // 6). Batch normalize layer

        // depthwise conv layers
        const auto dw_conv_shape = netstruct[offset++];
        const auto dw_bn_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->dw_conv, buffer,
            dw_conv_shape[0], dw_conv_shape[1], dw_conv_shape[2]);
        FillBatchnormLayer(tower_ptr->dw_bn, buffer, dw_bn_shape[0]);

        // 1st feedforward layers.
        const auto ffn_conv1_shape = netstruct[offset++];
        const auto ffn_bn1_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->conv1, buffer,
            ffn_conv1_shape[0], ffn_conv1_shape[1], ffn_conv1_shape[2]);
        FillBatchnormLayer(tower_ptr->bn1, buffer, ffn_bn1_shape[0]);

        // 2nd feedforward layers.
        const auto ffn_conv2_shape = netstruct[offset++];
        const auto ffn_bn2_shape = netstruct[offset++];
        FillConvolutionLayer(tower_ptr->conv2, buffer,
            ffn_conv2_shape[0], ffn_conv2_shape[1], ffn_conv2_shape[2]);
        FillBatchnormLayer(tower_ptr->bn2, buffer, ffn_bn2_shape[0]);

        tower_ptr->feedforward_channels = ffn_conv1_shape[1];
        const auto channels = weights_->residual_channels;
        const auto kernel = 1;
        if (channels != dw_conv_shape[1] ||
                channels != dw_bn_shape[0] ||
                channels != ffn_conv1_shape[0] ||
                channels != ffn_conv2_shape[1] ||
                channels != ffn_bn2_shape[0]) {
            throw std::runtime_error{"the channels of mixer block is wrong"};
        }
        if (kernel != ffn_conv1_shape[2] ||
                kernel != ffn_conv2_shape[2]) {
            throw std::runtime_error{"the kernel of mixer block is wrong"};
        }
    }

    if (tower_ptr->apply_se) {
        // squeeze-and-excitation module
        // 1). Fully connect layer
        // 2). Fully connect layer
        const auto se_squeeze_shape = netstruct[offset++];
        const auto se_excite_shape = netstruct[offset++];

        FillFullyconnectLayer(tower_ptr->squeeze, buffer,
            se_squeeze_shape[0], se_squeeze_shape[1]);
        FillFullyconnectLayer(tower_ptr->excite, buffer,
            se_excite_shape[0], se_excite_shape[1]);
        tower_ptr->se_size = se_squeeze_shape[1];

        const auto channels = weights_->residual_channels;
        if (3 * channels != se_squeeze_shape[0] ||
                2 * channels != se_excite_shape[1]) {
            throw std::runtime_error{"the SE module size is wrong"};
        }
    }
    return offset;
}

void DNNLoader::FillWeights(NetInfo &netinfo,
                            NetStack &netstack,
                            NetStruct &netstruct,
                            std::istream &buffer) const {
    weights_->residual_blocks = std::stoi(netinfo["ResidualBlocks"]);
    weights_->residual_channels = std::stoi(netinfo["ResidualChannels"]);
    if (weights_->version >= 5) {
        weights_->policy_head_channels = std::stoi(netinfo["PolicyHeadChannels"]);
        weights_->value_head_channels = std::stoi(netinfo["ValueHeadChannels"]);
    } else {
        weights_->policy_head_channels = std::stoi(netinfo["PolicyExtract"]);
        weights_->value_head_channels = std::stoi(netinfo["ValueExtract"]);
    }
    if (weights_->input_channels != std::stoi(netinfo["InputChannels"])) {
        throw std::runtime_error{"the number of input channels is wrong"};
    }

    // There are three types layer. Each layer has
    // two line weights. Here they are:
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
    const auto input_conv_shape = netstruct[0];
    FillConvolutionLayer(weights_->input_conv,
                         buffer,
                         input_conv_shape[0],
                         input_conv_shape[1],
                         input_conv_shape[2]);

    const auto input_bn_shape = netstruct[1];
    FillBatchnormLayer(weights_->input_bn,
                       buffer,
                       input_bn_shape[0]);

    if (weights_->input_channels != input_conv_shape[0] ||
            weights_->residual_channels != input_conv_shape[1] ||
            weights_->residual_channels != input_bn_shape[0] ||
            input_conv_shape[2] != 3) {
        throw std::runtime_error{"the input layers are wrong"};
    }
    auto t_offset = 2; // include 'input_conv' and 'input_bn'

    // block tower
    for (int b = 0; b < weights_->residual_blocks; ++b) {
        const auto block_spt = Splitter(netstack[b]);
        t_offset = FillBlock(t_offset, block_spt, netstruct, buffer);
    }

    auto h_offset = t_offset;

    // policy head
    const auto p_hd_conv_shape = netstruct[h_offset++];
    FillConvolutionLayer(weights_->p_hd_conv,
                         buffer,
                         p_hd_conv_shape[0],
                         p_hd_conv_shape[1],
                         p_hd_conv_shape[2]);

    const auto p_hd_bn_shape = netstruct[h_offset++];
    FillBatchnormLayer(weights_->p_hd_bn,
                       buffer,
                       p_hd_bn_shape[0]);

    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        const auto dw_conv_shape = netstruct[h_offset++];
        const auto dw_bn_shape = netstruct[h_offset++];
        FillConvolutionLayer(weights_->p_dw_conv, buffer,
            dw_conv_shape[0], dw_conv_shape[1], dw_conv_shape[2]);
        FillBatchnormLayer(weights_->p_dw_bn, buffer, dw_bn_shape[0]);

        const auto pt_conv_shape = netstruct[h_offset++];
        const auto pt_bn_shape = netstruct[h_offset++];
        FillConvolutionLayer(weights_->p_pt_conv, buffer,
            pt_conv_shape[0], pt_conv_shape[1], pt_conv_shape[2]);
        FillBatchnormLayer(weights_->p_pt_bn, buffer, pt_bn_shape[0]);
    }

    const auto p_inter_fc_shape = netstruct[h_offset++];
    FillFullyconnectLayer(weights_->p_inter_fc,
                          buffer,
                          p_inter_fc_shape[0],
                          p_inter_fc_shape[1]);

    const auto prob_conv_shape = netstruct[h_offset++];
    FillConvolutionLayer(weights_->prob_conv,
                         buffer,
                         prob_conv_shape[0],
                         prob_conv_shape[1],
                         prob_conv_shape[2]);

    const auto pass_fc_shape = netstruct[h_offset++];
    FillFullyconnectLayer(weights_->pass_fc,
                          buffer,
                          pass_fc_shape[0],
                          pass_fc_shape[1]);

    if (p_hd_conv_shape[2] != 1 || prob_conv_shape[2] != 1) {
        throw std::runtime_error{"the policy convolution kernel size is wrong"};
    }
    if (prob_conv_shape[1] != weights_->probabilities_channels) {
        throw std::runtime_error{"the number of policy ouput size is wrong"};
    }
    if (p_inter_fc_shape[1] != pass_fc_shape[0] ||
            p_inter_fc_shape[0] != 3 * weights_->policy_head_channels ||
            p_inter_fc_shape[1] != 1 * weights_->policy_head_channels) {
        throw std::runtime_error{"the number of policy fully connect size is wrong"};
    }
    if (pass_fc_shape[1] != weights_->pass_probability_outputs) {
        throw std::runtime_error{"the number of pass ouput size is wrong"};
    }

    // value head
    const auto v_hd_conv_shape = netstruct[h_offset++];
    FillConvolutionLayer(weights_->v_hd_conv,
                         buffer,
                         v_hd_conv_shape[0],
                         v_hd_conv_shape[1],
                         v_hd_conv_shape[2]);

    const auto v_hd_bn_shape = netstruct[h_offset++];
    FillBatchnormLayer(weights_->v_hd_bn,
                       buffer,
                       v_hd_bn_shape[0]);

    const auto v_inter_fc_shape = netstruct[h_offset++];
    FillFullyconnectLayer(weights_->v_inter_fc,
                          buffer,
                          v_inter_fc_shape[0],
                          v_inter_fc_shape[1]);

    const auto v_os_conv_shape = netstruct[h_offset++];
    FillConvolutionLayer(weights_->v_ownership,
                         buffer,
                         v_os_conv_shape[0],
                         v_os_conv_shape[1],
                         v_os_conv_shape[2]);

    const auto misc_fc_shape = netstruct[h_offset++];
    FillFullyconnectLayer(weights_->v_misc,
                          buffer,
                          misc_fc_shape[0],
                          misc_fc_shape[1]);
    if (v_hd_conv_shape[2] != 1 || v_os_conv_shape[2] != 1) {
        throw std::runtime_error{"the value convolution kernel size is wrong"};
    }
    if (v_os_conv_shape[1] != weights_->ownership_channels) {
        throw std::runtime_error{"the number of ownership ouput size is wrong"};
    }
    if (v_inter_fc_shape[1] != misc_fc_shape[0] ||
            v_inter_fc_shape[0] != 3 * weights_->value_head_channels ||
            v_inter_fc_shape[1] != 3 * weights_->value_head_channels) {
        throw std::runtime_error{"the number of value fully connect size is wrong"};
    }
    if (misc_fc_shape[1] != weights_->value_misc_outputs) {
        throw std::runtime_error{"the misc value layer size is wrong."};
    }

    auto line = std::string{};
    std::getline(buffer, line);
    const auto spt = Splitter(line);
    if (spt.GetWord(0)->Get<std::string>() != "end") {
        throw std::runtime_error{"weights file format is not acceptable"};
    }
    weights_->winograd = GetOption<bool>("winograd");
    weights_->loaded = true;
    DumpInfo();
    ProcessWeights();
}

void DNNLoader::ProcessWeights() const {
    const auto ProcessConvBlock = [](ConvLayer &conv, BatchNormLayer &bn) {
        // Fuse the BatchNormLayer into the ConvLayer.
        for (auto idx = size_t{0};
                 idx < conv.GetBiases().size(); ++idx) {
            conv.GetBiases()[idx] -= bn.GetMeans()[idx];
            bn.GetMeans()[idx] = 0.0f;

            const size_t stride = conv.GetWeights().size() /
                                      conv.GetBiases().size();
            const auto scale = bn.GetStddevs()[idx];
            for (auto k = size_t{0}; k < stride; ++k) {
                conv.GetWeights()[stride * idx + k] *= scale;
            }
            conv.GetBiases()[idx] *= scale;
            bn.GetStddevs()[idx] = 1.0f;
        }
    };

    // input layers
    ProcessConvBlock(weights_->input_conv, weights_->input_bn);

    // block tower
    for (auto &block : weights_->tower) {
        if (block->IsResidualBlock()) {
            ProcessConvBlock(block->conv1, block->bn1);
            ProcessConvBlock(block->conv2, block->bn2);
        } else if (block->IsBottleneckBlock()) {
            ProcessConvBlock(block->pre_btl_conv, block->pre_btl_bn);
            ProcessConvBlock(block->conv1, block->bn1);
            ProcessConvBlock(block->conv2, block->bn2);
            ProcessConvBlock(block->post_btl_conv, block->post_btl_bn);
        } else if (block->IsNestedBottleneckBlock()) {
            ProcessConvBlock(block->pre_btl_conv, block->pre_btl_bn);
            ProcessConvBlock(block->conv1, block->bn1);
            ProcessConvBlock(block->conv2, block->bn2);
            ProcessConvBlock(block->conv3, block->bn3);
            ProcessConvBlock(block->conv4, block->bn4);
            ProcessConvBlock(block->post_btl_conv, block->post_btl_bn);
        } else if (block->IsMixerBlock()) {
            ProcessConvBlock(block->dw_conv, block->dw_bn);
            ProcessConvBlock(block->conv1, block->bn1);
            ProcessConvBlock(block->conv2, block->bn2);
        }
    }

    // policy head
    ProcessConvBlock(weights_->p_hd_conv, weights_->p_hd_bn);
    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        ProcessConvBlock(weights_->p_dw_conv, weights_->p_dw_bn);
        ProcessConvBlock(weights_->p_pt_conv, weights_->p_pt_bn);
    }

    // value head
    ProcessConvBlock(weights_->v_hd_conv, weights_->v_hd_bn);
}

void DNNLoader::GetWeightsFromBuffer(std::vector<float> &weights, std::istream &buffer) const {
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
            // On macOS, stringstream struggles to parse extremely
            // small numeric values into a float; however, parsing
            // them as a double works correctly
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
                    throw std::runtime_error{"non-numeric in parameters"};
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

void DNNLoader::FillFullyconnectLayer(LinearLayer &layer,
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

void DNNLoader::FillBatchnormLayer(BatchNormLayer &layer,
                                  std::istream &buffer,
                                  const int channels) const {
    auto weights = std::vector<float>{};
    layer.Set(channels);

    GetWeightsFromBuffer(weights, buffer);
    layer.LoadMeans(weights);

    GetWeightsFromBuffer(weights, buffer);
    layer.LoadStddevs(weights, weights_->version==1);
}

void DNNLoader::FillConvolutionLayer(ConvLayer &layer,
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
