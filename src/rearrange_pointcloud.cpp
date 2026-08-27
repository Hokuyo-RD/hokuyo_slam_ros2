// Copyright (C) 2024 Kiyoshi Irie
// MIT License
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream> 
#include <cmath>   
#include <iomanip> 
#include <cstring>
#include <cctype>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Geometry>

// JSONライブラリのインクルード (nlohmann/json を使用することを想定)
#include "json.hpp"
using json = nlohmann::json;

// ----------------------------------------------------------------------
// ヘルパー関数: 2つの (x, y) 座標間の距離を計算
// ----------------------------------------------------------------------
double calculate_distance_2d(double x1, double y1, double x2, double y2) {
    return std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
}

// ----------------------------------------------------------------------
// ヘルパー関数: 2つの (x, y, z) 座標間の距離を計算
// ----------------------------------------------------------------------
double calculate_distance_3d(double x1, double y1, double z1,
                             double x2, double y2, double z2) {
    return std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2) + std::pow(z1 - z2, 2));
}

// ----------------------------------------------------------------------
// ヘルパー関数: 引数を実数として安全に読み取る
//
// std::stod は変換に失敗すると例外を投げ、捕捉しないとプログラムが
// 異常終了 (abort) する。利用者にとって原因が分からない終わり方に
// なるため、ここで捕捉して分かりやすいメッセージを出す。
// ----------------------------------------------------------------------
bool parse_positive_double(const char *arg, const char *name, double &out) {
    try {
        size_t idx = 0;
        double v = std::stod(arg, &idx);
        // 数値の後ろにゴミが付いている場合 (例: "1.0m") も誤りとして扱う
        while (idx < std::strlen(arg) && std::isspace(static_cast<unsigned char>(arg[idx]))) ++idx;
        if (idx != std::strlen(arg)) {
            std::cerr << "エラー: " << name << " に数値として読めない値が指定されました: '"
                      << arg << "'" << std::endl;
            return false;
        }
        if (!std::isfinite(v) || v < 0.0) {
            std::cerr << "エラー: " << name << " には 0 以上の数値を指定してください: '"
                      << arg << "'" << std::endl;
            return false;
        }
        out = v;
        return true;
    } catch (const std::exception &) {
        std::cerr << "エラー: " << name << " に数値として読めない値が指定されました: '"
                  << arg << "'" << std::endl;
        return false;
    }
}

int main(int argc, char *argv[]) {
    // ----------------------------------------------------------------------
    // 1. 引数の確認と初期設定
    // ----------------------------------------------------------------------
    if (argc < 4) { 
        std::cerr << "使用法: " << argv[0] << " <ログファイル名> <PCD出力ファイル名のベース> <JSON出力ファイル名> [点群結合間隔(m)] [Waypoint設置間隔(m)]" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "エラー: ログファイル " << argv[1] << " を開けません。" << std::endl;
        return 1;
    }

    std::string line;
    
    // PCDの出力ベース名 (argv[2])
    std::string str = argv[2]; 
    std::string outpcd_name = str + "_Acord.pcd";

    // JSONの出力ファイル名 (argv[3]から直接読み込む)
    std::string outjson_name = argv[3]; 

    // 引数から間隔設定を読み込み (指定がない場合はデフォルト値を使用)
    //
    // pc_save_distance / min_distance_m はいずれも「距離 [m]」である。
    // lio_raw (src/pcd_tf_extractor.py) と意味をそろえてあるため、
    // 0.3 のような 1 未満の小数を指定して密な地図を作ることができる。
    double pc_save_distance = 1.0;  // デフォルト: 1.0m (点群を足し込む間隔)
    double min_distance_m = 4.0;    // デフォルト: 4.0m (Waypoint を置く間隔)

    if (argc >= 5) {
        if (!parse_positive_double(argv[4], "点群結合間隔 (pc_save_distance)", pc_save_distance)) {
            return 1;
        }
    }
    if (argc >= 6) {
        if (!parse_positive_double(argv[5], "Waypoint設置間隔 (wp_save_distance)", min_distance_m)) {
            return 1;
        }
    }

    std::cout << "  PointCloud Distance Filter: " << pc_save_distance << " m" << std::endl;
    std::cout << "  Waypoint Distance Filter: " << min_distance_m << " m" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Waypoint抽出用の変数
    double last_wp_x = 0.0;
    double last_wp_y = 0.0;
    json waypoints_list = json::array(); // JSON配列としてウェイポイントを格納
    
    double first_wp_x = 0.0;
    double first_wp_y = 0.0;

    int cnt = 0;

    // 点群の距離フィルタ用 (最初の 1 枚は必ず保存する)
    bool has_last_pcd = false;
    double last_pcd_x = 0.0, last_pcd_y = 0.0, last_pcd_z = 0.0;
    int saved_cloud_count = 0;

    // ----------------------------------------------------------------------
    // 2. メインループ: 点群の結合と Waypoint の抽出
    //
    // 点群と Waypoint は、それぞれ独立した距離フィルタで間引く。
    // (lio_raw と同じ考え方)
    // ----------------------------------------------------------------------
    while (std::getline(file, line)) {
        ++cnt;

        std::istringstream iss(line);
        std::string filename;
        float x, y, z, qx, qy, qz, qw, rx, ry, rz;

        // データ抽出 (11項目)
        if (!(iss >> filename >> x >> y >> z >> qx >> qy >> qz >> qw >> rx >> ry >> rz)) {
            // ログの形式が想定外の場合、スキップ
            std::cerr << "Warning: Skipping malformed log line: " << line << std::endl; 
            continue;
        }

        // ------------------------------------------------------------------
        // A. PCD結合処理 (距離フィルタ)
        //
        // 前回足し込んだ位置から pc_save_distance [m] 以上離れたときだけ
        // 点群を読み込んで結合する。0 を指定した場合はすべて結合する。
        // ------------------------------------------------------------------
        double cur_x = static_cast<double>(x);
        double cur_y = static_cast<double>(y);
        double cur_z = static_cast<double>(z);

        bool save_this_cloud =
            !has_last_pcd ||
            calculate_distance_3d(cur_x, cur_y, cur_z,
                                  last_pcd_x, last_pcd_y, last_pcd_z) >= pc_save_distance;

        if (save_this_cloud) {
            Eigen::Quaterniond quaternion(qw, qx, qy, qz);
            Eigen::Translation3d translation(x, y, z);
            Eigen::Affine3d transform = translation * quaternion;

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
                std::cerr << "ERROR: Couldn't read file " << filename << std::endl;
                continue;
            }

            pcl::transformPointCloud(*cloud, *cloud, transform);
            *merged_cloud += *cloud;

            has_last_pcd = true;
            last_pcd_x = cur_x;
            last_pcd_y = cur_y;
            last_pcd_z = cur_z;
            ++saved_cloud_count;
        }

        // ------------------------------------------------------------------
        // B. Waypoint 抽出処理
        // ------------------------------------------------------------------
        double current_x = cur_x;
        double current_y = cur_y;

        if (waypoints_list.empty() || 
            calculate_distance_2d(current_x, current_y, last_wp_x, last_wp_y) >= min_distance_m) 
        {
            // Waypointリストが空の場合、このポーズを最初の Waypoint として記録
            if (waypoints_list.empty()) {
                first_wp_x = current_x;
                first_wp_y = current_y;
            }
            
            // 相対座標を計算
            double relative_x = current_x - first_wp_x;
            double relative_y = current_y - first_wp_y;

            // 新しい Waypoint を追加
            json wp = json::array({
                // [x, y, 0.0] の形式で相対位置情報
                json::array({relative_x, relative_y, 0.0}), 
                // [0.0, 0.0, qz, qw] の形式で回転情報 
                json::array({0.0, 0.0, qz, qw}), 
                // デフォルトパラメータ
                json::object({
                    {"type", "normal"},
                    {"value", 0.0},
                    {"xy_tolerance", 1.0},
                    {"yaw_tolerance", 3.14}
                })
            });
            waypoints_list.push_back(wp);

            // 最後の Waypoint の絶対座標を更新 (次の距離計算のために使用)
            last_wp_x = current_x;
            last_wp_y = current_y;
        }
    }

    // ----------------------------------------------------------------------
    // 3. 結果の保存
    // ----------------------------------------------------------------------

    // PCDファイルの保存
    std::cout << "結合した点群: " << saved_cloud_count << " 枚 / " << cnt << " 枚中"
              << " (合計 " << merged_cloud->size() << " 点)" << std::endl;

    if (merged_cloud->empty()) {
        std::cerr << "エラー: 結合された点群が空です。地図は作成できません。" << std::endl;
        std::cerr << "       concat.txt の内容と、点群 (PCD) ファイルの有無を確認してください。"
                  << std::endl;
        return 1;
    }

    pcl::io::savePCDFileASCII(outpcd_name, *merged_cloud);
    std::cout << "PCDファイルを保存しました: " << outpcd_name << std::endl;

    // ----------------------------------------------------------------------
    // 🔧 修正: Waypointリストの最初2つと最後2つを削除
    // ----------------------------------------------------------------------
    size_t min_waypoints = 4; // 削除対象の合計数 (最初2 + 最後2)

    if (waypoints_list.size() > min_waypoints) {
        // 最初2つを削除 (0番目と1番目)
        // erase(開始イテレータ, 終了イテレータ)
        // waypoints_list.erase(waypoints_list.begin(), waypoints_list.begin() + 2);
        
        // 残りのリストの最後2つを削除
        // erase(最後尾から数えたイテレータ, 最後尾のイテレータ)
        // waypoints_list.erase(waypoints_list.end() - 2, waypoints_list.end());

        std::cout << "情報: Waypointリストの最初2つと最後2つの要素を削除しました。" << std::endl;
    } else if (waypoints_list.size() > 0) {
        std::cout << "警告: Waypointの数が少ないため、最初2つと最後2つの削除をスキップしました。"
                  << "（現在の要素数: " << waypoints_list.size() << "）" << std::endl;
    }
    
    // JSONファイルの保存
    std::ofstream ofs(outjson_name);
    if (ofs.is_open()) {
        // インデントをつけて読みやすい形式で保存
        ofs << std::setw(4) << waypoints_list << std::endl; 
        std::cout << "Waypointファイルを保存しました: " << outjson_name << std::endl;
    } else {
        std::cerr << "エラー: Waypointファイルを保存できませんでした: " << outjson_name << std::endl;
    }

    return 0;
}