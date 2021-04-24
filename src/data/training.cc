#include "data/training.h"

void ArrayStreamOut(std::ostream &out, std::vector<float> arr) {
    const auto size = arr.size();
    for (size_t i = 0; i < size; ++i) {
        out << arr[i];
        if (i != size-1) out << ' ';
    }
    out << std::endl;
}

void TrainingBuffer::StreamOut(std::ostream &out) {
    out << version << std::endl;
    out << mode << std::endl;
    out << board_size << std::endl;
    out << komi << std::endl;

    out << side_to_move << std::endl;

    ArrayStreamOut(out, planes);
    ArrayStreamOut(out, probabilities);
    ArrayStreamOut(out, auxiliary_probabilities);
    ArrayStreamOut(out, ownership);

    out << result << std::endl;
    out << final_score << std::endl;
}

std::ostream& operator<<(std::ostream& out, TrainingBuffer &buf) {
    buf.StreamOut(out);
    return out;
}
