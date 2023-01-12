#include "version.h"
#include "utils/format.h"

#include <vector>

const std::string kProgram = "Sayuri";

const std::vector<std::string> kReMember = { // R∃/MEMBER
    "2019-3-6",
    "ME & CREED <nZkv>" // 8. ~
};

// for the version 0.x.x ...
const std::vector<std::string> kMikazukiNoKoukai = { // 新月航程 / ミカヅキの航海
    "2017-5-17",
    "Mikazuki",                           // 1.  三日月 / ミカヅキ
    "Heikousen",                          // 2.  平行線 
    "Juu Oku Nen",                        // 3.  十億年
    "Cake o Yaku",                        // 4.  烤蛋糕 / ケーキを焼
    "Furaregai Girl",                     // 5.  被甩的男男女女 / フラレガイガール
    "Hachi to Misemono",                  // 6.  蜜蜂與馬戲團 / 蜂と見世物 
    "Ru-Rararu-Ra-Rurararu-Ra",           // 7.  嚕啦啦嚕拉嚕拉拉嚕拉 / るーららるーらーるららるーらー
    "Odd Eye",                            // 8.  異色瞳 / オッドアイ
    "Sore wa Chiisa na Hikari no You na", // 9.  宛如渺小的微光 / それは小さな光のような
    "Raise de Aou",                       // 10. 來世再會吧 / 来世で会おう
    "Knot",                               // 11. ~
    "Anonymous",                          // 12. 匿名者 / アノニマス
    "Natsu",                              // 13. 夏
    "Birthday Song"                       // 14. ~ 
};

// for the version 1.x.x ...
const std::vector<std::string> kSanketsuGirl = { // 酸欠少女
    "2022-8-10",
    "Sanketsu Girl",   // 1.  酸欠少女
    "Tower of Flower", // 2.  花之塔 / 花の塔
    "About a Voyage",  // 3.  航海之歌 / 航海の唄
    "DAWN DANCE",      // 4.  ~
    "World Secret",    // 5.  世界的祕密 / 世界の秘密
    "Aoibridge",       // 6.  葵橋
    "Moon & Bouquet",  // 7.  月與花束 / 月と花束
    "Kamisama",        // 8.  神 
    "Summer Bug",      // 9.  夏蟲
    "Dawn",            // 10. 黎明
    "Nejiko"           // 11. ~
};

constexpr bool kDevVersion = false;
constexpr size_t kVersionMajor = 0;
constexpr size_t kVersionMinor = 4;
constexpr size_t kVersionPatch = 0;

std::string GetProgramName() {
    return kProgram;
}

std::string GetProgramVersion() {
    if (kDevVersion) {
        return "dev";
    }
    return Format("%d.%d.%d", kVersionMajor, kVersionMinor, kVersionPatch);
}

std::string GetVersionName() {
    if (kDevVersion) {
        return kReMember[1];
    }
    if (kVersionMajor == 0) {
        if (kVersionMinor <= 0 ||
                kVersionMinor >= kMikazukiNoKoukai.size()) {
            return "N/A";
        }
        return kMikazukiNoKoukai[kVersionMinor];
    }
    else if (kVersionMajor == 1) {
        if (kVersionMinor <= 0 ||
                kVersionMinor >= kSanketsuGirl.size()) {
            return "N/A";
        }
        return kSanketsuGirl[kVersionMinor];
    }
    return "";
}
